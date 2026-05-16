# story_writer.py

# Import necessary libraries
import argparse
import difflib
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
from pathlib import Path
import re

# Named constant for magic number
MAX_KEY_EVENTS_PER_CHUNK = 5

def count_tokens(text: str) -> int:
    """
    Estimates the number of tokens in a string based on a word count.

    This is an approximation, as actual tokenization can vary between models.
    A common rule of thumb is that one word is roughly equal to one token.
    """
    return len(text.split())


def context_bar(used, total, width=30):
    pct = used / total
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{used:,} / {total:,} [{bar}] {pct*100:.1f}%"


def llm_invoke(llm, prompt, label, context_size):
    response = llm.invoke(prompt)
    input_tokens = 0
    if meta := getattr(response, 'response_metadata', {}):
        input_tokens = meta.get('token_usage', {}).get('prompt_tokens', 0)
    print(f"  {label}: {context_bar(input_tokens, context_size)}")
    return response


def get_incremented_filename(filename):
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    match = re.match(r"^(.*?)(\d+)?$", stem)
    base = match.group(1)
    num = match.group(2)
    candidate = filename
    if not path.exists():
        return filename
    # If it ends with a number, increment it
    if num:
        new_num = int(num) + 1
        new_stem = f"{base}{new_num}"
    else:
        new_stem = f"{stem}1"
    candidate = str(path.with_name(new_stem + suffix))
    # Keep incrementing if the candidate exists
    while Path(candidate).exists():
        match = re.match(r"^(.*?)(\d+)?$", Path(candidate).stem)
        base = match.group(1)
        num = match.group(2)
        if num:
            new_num = int(num) + 1
            new_stem = f"{base}{new_num}"
        else:
            new_stem = f"{Path(candidate).stem}1"
        candidate = str(Path(candidate).with_name(new_stem + suffix))
    return candidate


_EXTRACT_NAMES_PROMPT = (
    "/no_think\n"
    "List every character name and place name that appears in the text below.\n"
    "Output one name per line, nothing else. No explanations, no numbers, no bullet points.\n\n"
    "TEXT:\n{text}\n\n"
    "NAMES:"
)

_ORDER_CHECK_PROMPT = (
    "/no_think\n"
    "Check whether these story events are listed in a logical order.\n"
    "Identify any event stated before something it logically depends on "
    "(e.g. a battle outcome listed before the battle starts).\n\n"
    "KEY EVENTS:\n{key_events}\n\n"
    "Respond in EXACTLY this format:\n"
    "ISSUES:\n"
    "- <one issue per line, or 'None' if the order is fine>\n\n"
    "CORRECTED ORDER:\n"
    "1. <event text copied verbatim>\n"
    "2. <event text copied verbatim>\n"
    "(List ALL events in the correct logical order. Copy each line verbatim. Only reorder, do not change any wording.)\n"
)


def _extract_known_names(summary_context, llm, context_size):
    try:
        response = llm_invoke(llm, _EXTRACT_NAMES_PROMPT.format(text=summary_context), "Name extraction", context_size)
        names = []
        for line in response.content.strip().splitlines():
            name = line.strip().lstrip("-•*0123456789. ").strip()
            if len(name) > 1:
                names.append(name)
        return names
    except Exception:
        return []


def _find_name_typos(key_events, known_names):
    if not known_names:
        return list(key_events), []
    corrected = []
    issues = []
    for event in key_events:
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', event)
        corrected_event = event
        for word in dict.fromkeys(words):  # preserve order, deduplicate
            if word in known_names:
                continue
            matches = difflib.get_close_matches(word, known_names, n=1, cutoff=0.8)
            if matches:
                correct = matches[0]
                corrected_event = re.sub(r'\b' + re.escape(word) + r'\b', correct, corrected_event)
                issues.append(f"'{word}' looks like a misspelling of '{correct}'")
        corrected.append(corrected_event)
    return corrected, issues


def _check_event_order(key_events, llm, context_size):
    key_events_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(key_events))
    try:
        response = llm_invoke(llm, _ORDER_CHECK_PROMPT.format(key_events=key_events_str), "Order check", context_size)
        text = response.content.strip()
        issues = []
        corrected_order = list(key_events)

        if "ISSUES:" in text and "CORRECTED ORDER:" in text:
            parts = text.split("CORRECTED ORDER:", 1)
            issues_block = parts[0].replace("ISSUES:", "").strip()
            order_block = parts[1].strip()

            for line in issues_block.splitlines():
                line = line.strip().lstrip("-").strip()
                if line and line.lower() != "none":
                    issues.append(line)

            key_event_set = set(key_events)
            seen = set()
            corrected_lines = []
            for line in order_block.splitlines():
                m = re.match(r"^\d+\.\s+(.+)$", line.strip())
                if m:
                    candidate = m.group(1).strip()
                    if candidate in key_event_set and candidate not in seen:
                        corrected_lines.append(candidate)
                        seen.add(candidate)
            if len(corrected_lines) == len(key_events):
                corrected_order = corrected_lines
            else:
                print(f"  (Warning: could not parse corrected order — matched {len(corrected_lines)}/{len(key_events)} events)")

        return issues, corrected_order
    except Exception:
        return [], list(key_events)


def _apply_corrections_to_text(instructions_text, original_events, corrected_events):
    """Write corrected events back into the instructions file.
    If only text changed (no reorder), replace lines in-place to preserve blank-line grouping.
    If order changed, dump the events flat between the markers."""
    order_changed = [o for o in corrected_events] != [o for o in original_events if o in corrected_events and corrected_events.index(o) == original_events.index(o)]
    # Simpler check: same events in same sequence?
    order_changed = corrected_events != original_events and sorted(corrected_events) == sorted(original_events)

    lines = instructions_text.splitlines(keepends=True)
    result = []
    in_events = False
    event_idx = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "START OF KEY EVENTS:":
            in_events = True
            result.append(line)
            if order_changed:
                result.append("\n")
                for event in corrected_events:
                    result.append(event + "\n")
                result.append("\n")
        elif stripped == "END OF KEY EVENTS:":
            in_events = False
            result.append(line)
        elif in_events:
            if order_changed:
                pass  # already written above
            elif stripped:
                if event_idx < len(corrected_events):
                    ending = "\n" if line.endswith("\n") else ""
                    result.append(corrected_events[event_idx] + ending)
                    event_idx += 1
                else:
                    result.append(line)
            else:
                result.append(line)  # preserve blank lines
        else:
            result.append(line)

    return "".join(result)


_DEL   = "\033[41;97m"   # red background, bright white text  — removed chars
_ADD   = "\033[42;97m"   # green background, bright white text — added chars
_RESET = "\033[0m"


def _track_changes(old_text, new_text):
    """Return a single line showing old and new text inline, track-changes style."""
    old_tokens = re.findall(r"\S+|\s+", old_text)
    new_tokens = re.findall(r"\S+|\s+", new_text)
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    out = ""
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            out += "".join(old_tokens[a0:a1])
        elif op == "replace":
            out += _DEL + "".join(old_tokens[a0:a1]) + _RESET
            out += _ADD + "".join(new_tokens[b0:b1]) + _RESET
        elif op == "delete":
            out += _DEL + "".join(old_tokens[a0:a1]) + _RESET
        elif op == "insert":
            out += _ADD + "".join(new_tokens[b0:b1]) + _RESET
    return out


def _render_with_changes(instructions_text, key_events, name_corrected, final_corrected):
    """Render full instructions as one block with track-changes highlighting.
    Diffs name_corrected vs final_corrected so only order changes produce red/green lines.
    Equal entries are rendered with _track_changes(original, name_corrected) to show name fixes inline."""
    # Map name_corrected text → original text for looking up originals in the diff
    original_map = {nc: orig for orig, nc in zip(key_events, name_corrected)}

    diff = list(difflib.ndiff(name_corrected, final_corrected))
    rendered_events = []
    for entry in diff:
        tag = entry[:2]
        content = entry[2:]
        if tag == "? ":
            continue
        elif tag == "  ":
            # Unchanged position: show with inline name correction if any
            orig = original_map.get(content, content)
            rendered_events.append(_track_changes(orig, content) if orig != content else content)
        elif tag == "- ":
            # Line moved away from here: show original text on red background
            orig = original_map.get(content, content)
            rendered_events.append(_DEL + orig + _RESET)
        elif tag == "+ ":
            # Line moved to here: show corrected text on green background
            rendered_events.append(_ADD + content + _RESET)

    # Embed rendered events inside the instructions file structure
    lines = instructions_text.splitlines()
    result = []
    in_events = False
    for line in lines:
        stripped = line.strip()
        if stripped == "START OF KEY EVENTS:":
            in_events = True
            result.append(line)
            result.append("")
            for ev in rendered_events:
                result.append(ev)
            result.append("")
        elif stripped == "END OF KEY EVENTS:":
            in_events = False
            result.append(line)
        elif in_events:
            pass  # replaced by rendered_events above
        else:
            result.append(line)
    return "\n".join(result)


def run_preprocess(key_events, instructions_text, static_lore, summary_context, llm, instructions_path, context_size):
    print("\n" + "=" * 50)
    print("Pre-processor: continuity check...")
    print("=" * 50)

    known_names = _extract_known_names(static_lore, llm, context_size)

    print("  Checking for name misspellings...")
    name_corrected, name_issues = _find_name_typos(key_events, known_names)

    order_issues, order_corrected = _check_event_order(key_events, llm, context_size)

    all_issues = name_issues + order_issues

    if not all_issues:
        print("No issues found.")
        return key_events, instructions_text

    print(f"\nIssues found ({len(all_issues)}):")
    for issue in all_issues:
        print(f"  - {issue}")

    # Combine: apply name corrections on top of the order-corrected list
    name_map = {orig: corr for orig, corr in zip(key_events, name_corrected)}
    final_corrected = [name_map.get(e, e) for e in order_corrected]

    print()
    print(_render_with_changes(instructions_text, key_events, name_corrected, final_corrected))
    print()

    if final_corrected != key_events:
        answer = input(f"Apply corrections to {instructions_path}? [y/N]: ").strip().lower()
        if answer == "y":
            new_text = _apply_corrections_to_text(instructions_text, key_events, final_corrected)
            with open(instructions_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"Corrections applied and saved to {instructions_path}.")
            return final_corrected, new_text
    else:
        answer = input("Proceed with generation anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborting.")
            sys.exit(0)

    return key_events, instructions_text


def main():
    """
    Main function to parse arguments, load files, and execute the summarization and instruction chains.
    This version uses a 'refine' chain for a more detailed, longer output.
    """
    # Record the start time of the script
    start_time = time.time()
    
    # 1. Argument Parsing
    # ==============================================================================
    parser = argparse.ArgumentParser(
        description="Rewrite a large text file using LangChain and LM Studio with a 'refine' chain."
    )
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Path to the static lore/background text file (default: story_background.txt in working dir).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Path to the instructions file (default: instructions.txt in working dir).",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        dest="working_dir",
        help="Working directory containing story files (default: current directory).",
    )
    
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:1234/v1",
        help="The URL for the local LLM service (default: LM Studio port 1234).",
    )
    
    parser.add_argument(
        "--key_event_chunk_size",
        type=int,
        default=MAX_KEY_EVENTS_PER_CHUNK,
        help="Number of key events to process in each iteration. Lower number = longer story.",
    )
    
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Specify a chapter number to rebuild using previous summaries.",
    )

    parser.add_argument(
        "--regenerate",
        type=int,
        default=None,
        help="Regenerate a specific chapter, discarding its story and summary before rewriting.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="LLM top-p nucleus sampling.",
    )

    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=None,
        help="Penalises repeated tokens.",
    )

    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Penalises repeated topics.",
    )

    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="Minimum probability threshold for tokens.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="LLM top-k sampling.",
    )

    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=None,
        help="Multiplicative penalty for repeated tokens (llama.cpp style).",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=131072,
        dest="context_size",
        help="Model context window size in tokens (default: 131072 = 128k).",
    )

    parser.add_argument(
        "--min_tokens",
        type=int,
        default=None,
        help="Minimum tokens to generate per chunk before the model is allowed to stop.",
    )

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        default=False,
        dest="no_preprocess",
        help="Skip the instructions pre-processor continuity check.",
    )

    args = parser.parse_args()

    # Switch working directory if requested.
    # Do this before resolving any file defaults so all relative paths land here.
    if args.working_dir is not None:
        working_dir = Path(args.working_dir).resolve()
        if not working_dir.is_dir():
            print(f"Error: Working directory '{working_dir}' does not exist.")
            return
        os.chdir(working_dir)
        print(f"Working directory: {working_dir}")

    # Fill in defaults relative to (possibly changed) CWD.
    if args.story is None:
        args.story = "story_background.txt"
    if args.instructions is None:
        if args.regenerate is not None:
            args.instructions = f"chapter{args.regenerate}_instructions.txt"
        else:
            args.instructions = "instructions.txt"

    # Verify that the files exist
    if not (story_path := Path(args.story)).is_file():
        print(f"Error: The static lore file '{story_path}' was not found.")
        return
    if not (instructions_path := Path(args.instructions)).is_file():
        print(f"Error: The instructions file '{instructions_path}' was not found.")
        return

    # LLM Setup
    # ==============================================================================
    stop_tokens = ["\u0001", "<|end_of_text|>", "<|eot_id|>"]

    # Build kwargs only for parameters that were explicitly provided
    llm_kwargs = {
        "openai_api_base": args.api_url,
        "openai_api_key": "lm-studio",
        "model": "default",
        "max_tokens": -1,
    }
    if args.temperature is not None:
        llm_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        llm_kwargs["top_p"] = args.top_p
    if args.frequency_penalty is not None:
        llm_kwargs["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        llm_kwargs["presence_penalty"] = args.presence_penalty

    extra_body = {}
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.repeat_penalty is not None:
        extra_body["repeat_penalty"] = args.repeat_penalty
    if args.min_tokens is not None:
        extra_body["min_tokens"] = args.min_tokens
    if extra_body:
        llm_kwargs["extra_body"] = extra_body

    try:
        llm = ChatOpenAI(**llm_kwargs)
        print(f"Connected to LLM at {args.api_url}")
    except Exception as e:
        print(f"Failed to connect to the LLM service at {args.api_url}.")
        print(f"Error: {e}")
        return

    # Narrative Retelling Process
    # ==============================================================================
    # Load the static lore file
    with open(args.story, "r", encoding="utf-8") as f:
        static_lore = f.read()

    # Find and load all chapter summaries
    summary_files = []
    import glob
    for file in glob.glob("chapter*_summary.txt"):
        match = re.search(r"chapter(\d+)_summary\.txt", file)
        if match:
            chapter_num = int(match.group(1))
            summary_files.append((chapter_num, file))

    # Sort them by chapter number
    summary_files.sort()

    # Check if instructions file is a chapter regeneration request
    instructions_file_name = Path(args.instructions).name
    regenerate_match = re.match(r"^chapter(\d+)_instructions\.txt$", instructions_file_name)
    regenerating_chapter = False

    if regenerate_match and args.chapter is None:
        # Regenerating a specific chapter based on filename
        regenerate_chapter_num = int(regenerate_match.group(1))
        regenerating_chapter = True
        next_chapter_num = regenerate_chapter_num

        print(f"\nRegenerating chapter {regenerate_chapter_num}...")

        # Delete existing story and summary files
        story_file = Path(f"chapter{regenerate_chapter_num}_story.txt")
        summary_file = Path(f"chapter{regenerate_chapter_num}_summary.txt")

        if story_file.exists():
            story_file.unlink()
            print(f"Deleted existing {story_file.name}")

        if summary_file.exists():
            summary_file.unlink()
            print(f"Deleted existing {summary_file.name}")
    elif args.regenerate is not None:
        # --regenerate N: discard existing story and summary, then rewrite
        next_chapter_num = args.regenerate
        regenerating_chapter = True
        print(f"\nRegenerating chapter {next_chapter_num}...")

        story_file = Path(f"chapter{next_chapter_num}_story.txt")
        summary_file = Path(f"chapter{next_chapter_num}_summary.txt")

        if story_file.exists():
            story_file.unlink()
            print(f"Deleted existing {story_file.name}")

        if summary_file.exists():
            summary_file.unlink()
            print(f"Deleted existing {summary_file.name}")
    elif args.chapter is not None:
        # Explicit rebuild via --chapter flag
        next_chapter_num = args.chapter
        regenerating_chapter = True
        print(f"Rebuilding chapter {next_chapter_num} as requested.")
    else:
        # Create new chapter (default behavior)
        existing_chapters = []
        for file in glob.glob("chapter*_summary.txt"):
            match = re.search(r"chapter(\d+)_summary\.txt", file)
            if match:
                existing_chapters.append(int(match.group(1)))

        if existing_chapters:
            next_chapter_num = max(existing_chapters) + 1
        else:
            next_chapter_num = 1

    # Load the ongoing story context
    full_summary_text = static_lore + "\n\n"
    loaded_summaries = 0
    for chapter_num, summary_file in summary_files:
        if chapter_num < next_chapter_num:
            with open(summary_file, "r", encoding="utf-8") as f:
                full_summary_text += f"\n--- Chapter {chapter_num} Summary ---\n"
                full_summary_text += f.read() + "\n"
            loaded_summaries += 1
    
    if loaded_summaries == 0:
        print("No previous chapter summaries loaded. Starting fresh.")

    new_chapter_file = f"chapter{next_chapter_num}_story.txt"
    new_summary_file = f"chapter{next_chapter_num}_summary.txt"
    new_instructions_file = f"chapter{next_chapter_num}_instructions.txt"

    # Load and process instructions
    if regenerating_chapter:
        # Regenerating an existing chapter
        if regenerate_match:
            # Instructions file was specified directly (chapterN_instructions.txt)
            with open(args.instructions, "r", encoding="utf-8") as f:
                instructions_text = f.read()
        else:
            # Using --chapter flag, load from expected instructions file
            if Path(new_instructions_file).is_file():
                with open(new_instructions_file, "r", encoding="utf-8") as f:
                    instructions_text = f.read()
            else:
                print(f"Error: Could not find {new_instructions_file}")
                return
    else:
        # Creating a new chapter
        with open(args.instructions, "r", encoding="utf-8") as f:
            instructions_text = f.read()

        if next_chapter_num > 1:
            prev_instructions_file = f"chapter{next_chapter_num - 1}_instructions.txt"
            if Path(prev_instructions_file).is_file():
                with open(prev_instructions_file, "r", encoding="utf-8") as f:
                    prev_instructions_text = f.read()
                if instructions_text.strip() == prev_instructions_text.strip():
                    print(f"Chapter {next_chapter_num - 1} already generated for those instructions.")
                    return

        # Save instructions for this new chapter
        with open(new_instructions_file, "w", encoding="utf-8") as f:
            f.write(instructions_text)

    # Extract key events from instructions.txt
    key_events = []
    in_events = False
    for line in instructions_text.splitlines():
        stripped = line.strip()
        if stripped == "START OF KEY EVENTS:":
            in_events = True
            continue
        if stripped == "END OF KEY EVENTS:":
            break
        if in_events and stripped:
            key_events.append(stripped)

    # Print instructions section with header
    print("\n" + "="*50)
    print(f"Instructions ({new_instructions_file}) used to create {new_chapter_file}...")
    print("="*50)

    for event in key_events:
        print(event)
    print("\n")

    # Pre-processor: continuity check
    if not args.no_preprocess:
        key_events, instructions_text = run_preprocess(
            key_events, instructions_text, static_lore, full_summary_text, llm, args.instructions, args.context_size
        )

    # Generate the new story chapter
    print("\n" + "="*50)
    print(f"Applying instructions to create {new_chapter_file}...")
    print("="*50)

    # Group events into chunks
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    Path(new_chapter_file).unlink(missing_ok=True)

    # Prompt template
    chunk_prompt_template = (
        "/no_think\n"
        "INSTRUCTION\n"
        "You are an expert at writing engaging children's fantasy stories. \n"
        "Write ONLY the next section continuing from where the previous text ended. \n"
        "You MUST cover EVERY numbered key event listed below, in order, without skipping any.\n"
        "Each numbered event is mandatory. Do not skip, merge, or omit any event.\n"
        "Each event MUST happen sequentially in the order listed. Do not reorder.\n"
        "DO NOT repeat or rewrite any part of the previous story.\n"
        "STOP writing the moment you have covered the last numbered key event. Do NOT write anything after it.\n"
        "Do NOT wrap up, conclude, or add any content beyond the last numbered event.\n"
        "Use simple language, short to medium sentences.\n"
        "Write in present tense throughout. Use 'Vasu strikes' not 'Vasu struck', 'she moves' not 'she moved', 'he does not hesitate' not 'he did not hesitate'. Never slip into past tense.\n"
        "Before writing, silently assess the narrative significance of each event in this chunk. Combat and action sequences are always major, deserving rich expansion (200–300 words each). Dialogue and negotiation scenes should be tighter — focus on what's said and decided, not internal monologue or atmosphere. Other events deserve 75–125 words. Do not output this assessment.\n"
        "Expand each event meaningfully with sensory details, dialogue, and character thoughts. Avoid padding - if a scene is simple dialogue or transit, keep it concise.\n"
        "Do not introduce new characters or events unless requested.\n"
        "Do not add titles, headers, or numbered sections. Output pure flowing prose only.\n"
        "Open each new section with something unexpected — a sharp action, a sound, a line of dialogue, or a single vivid detail that drops the reader straight into the scene. Avoid formulaic openings like 'Morning light filtered...', 'The fire crackled...', or rolling through each character's state one by one.\n"
        "NEVER reproduce or paraphrase any key event text as a header, label, or sentence opener.\n"
        "Vary your sentence openings. Never start more than two sentences in a row with the same word, especially pronouns like 'He', 'She', or 'They'. Break up pronoun runs by starting with the character's name, an action ('Reaching into his pack...'), a detail ('Eyes wide, he...'), or a prepositional phrase ('From across the room...').\n"
        "ONLY OUTPUT THE NEW CONTINUATION, directly related to key events below.\n"
        "BEGINNING OF BACKGROUND\n{previous_story}\n"
        "END OF BACKGROUND\n\n"
        "BEGINNING OF KEY EVENTS (MUST ALL BE COVERED, IN ORDER, NONE SKIPPED)\n{key_events}\n"
        "END OF KEY EVENTS\n\n"
        "*** HARD STOP: As soon as the last numbered event above is written, stop immediately. Write nothing after it. ***\n\n"
        "Write the continuation now, covering every single numbered event above, starting immediately after where the previous text ended. Stop the moment the last event is done:\n"
    )

    chunk_prompt = PromptTemplate(
        input_variables=["previous_story", "key_events"],
        template=chunk_prompt_template
    )

    chunk_summary_prompt_template = (
        "/no_think\n"
        "Compact and summarize this story passage into ~400 words MAX.\n"
        "Write in tight, tense prose. NO dialogue. NO emotions. NO descriptions.\n"
        "Just factual events: what happened, where they are, when, their condition, and the last moment.\n"
        "Output ONLY the summary paragraph, no lists or introductions.\n"
        "\nPASSAGE:\n{chunk_text}\n\nSUMMARY:"
    )
    chunk_summary_prompt = PromptTemplate(
        input_variables=["chunk_text"],
        template=chunk_summary_prompt_template
    )

    whole_new_chapter = ""
    summary_plus_new_story = full_summary_text

    event_chunks = list(chunk_list(key_events, args.key_event_chunk_size))

    background_tokens = count_tokens(full_summary_text)
    chunk_metrics = []
    actual_model_name = "Unknown"

    for idx, chunk in enumerate(event_chunks):
        chunk_start = time.time()
        key_events_str = "\n".join(f"{i+1}. {event}" for i, event in enumerate(chunk))
        prompt = chunk_prompt.format(
            previous_story=summary_plus_new_story,
            key_events=key_events_str
        )
        try:
            new_story_section = llm_invoke(llm, prompt, f"Chunk {idx+1}/{len(event_chunks)}", args.context_size)
            whole_new_chapter += new_story_section.content.strip() + "\n\n"
        except Exception as e:
            print(f"Error generating chunk {idx+1}: {e}")
            break

        chunk_end = time.time()
        duration = chunk_end - chunk_start

        input_tokens = 0
        output_tokens = 0
        if (metadata := getattr(new_story_section, 'response_metadata', {})) and 'token_usage' in metadata:
            input_tokens = metadata['token_usage'].get('prompt_tokens', 0)
            output_tokens = metadata['token_usage'].get('completion_tokens', 0)
        else:
            output_tokens = count_tokens(new_story_section.content)

        if metadata := getattr(new_story_section, 'response_metadata', {}):
            actual_model_name = metadata.get('model_name', actual_model_name)

        # Summarise the chunk and use that as rolling context instead of full text
        summary_tokens = 0
        summary_input_tokens = 0
        summary_duration = 0
        try:
            summary_start_time = time.time()
            chunk_summary_msg = llm_invoke(
                llm,
                chunk_summary_prompt.format(chunk_text=new_story_section.content.strip()),
                f"Chunk {idx+1}/{len(event_chunks)} summary",
                args.context_size,
            )
            summary_duration = time.time() - summary_start_time
            chunk_summary_text = chunk_summary_msg.content.strip()
            summary_plus_new_story += f"\n{chunk_summary_text}"
            if (meta := getattr(chunk_summary_msg, 'response_metadata', {})) and 'token_usage' in meta:
                summary_tokens = meta['token_usage'].get('completion_tokens', 0)
                summary_input_tokens = meta['token_usage'].get('prompt_tokens', 0)
            else:
                summary_tokens = count_tokens(chunk_summary_text)
        except Exception as e:
            print(f"Warning: chunk {idx+1} summary failed ({e}), falling back to full text.")
            summary_plus_new_story += "\nxx\n" + new_story_section.content.strip()

        chunk_metrics.append({
            "duration": duration,
            "summary_duration": summary_duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "summary_tokens": summary_tokens,
            "summary_input_tokens": summary_input_tokens,
        })

    print(f"New chapter written to {new_chapter_file}")
    with open(new_chapter_file, "w", encoding="utf-8") as f:
        f.write(whole_new_chapter.strip() + "\n\n")

    # Print to console (optional)
    print("\n" + "="*50)
    print(whole_new_chapter)
    print("="*50)

    # Generate summary
    # ==============================================================================
    print("\n" + "="*50)
    print(f"Generating summary of {new_chapter_file}...")
    print("="*50)

    chapter_summary_prompt_template = (
        "/no_think\n"
        "You are a game master writing a continuity record so future chapters stay coherent.\n"
        "Summarize the chapter below in 200-400 words. Hard limit: 400 words.\n"
        "Write in plain, terse sentences. No dialogue. No descriptions. No emotions.\n"
        "\n"
        "INCLUDE ONLY facts with lasting consequences — facts a future writer must know to avoid contradictions:\n"
        "- Items acquired or lost (weapons, treasure, equipment, consumables)\n"
        "- Locations visited, with any features that affect future events\n"
        "- Named characters met, aided, or parted from\n"
        "- New abilities, spells, or knowledge the party now possesses\n"
        "- Party condition at chapter end (injuries, exhausted spells/resources)\n"
        "- Unresolved threats or open story hooks\n"
        "\n"
        "OMIT anything that has no lasting effect on future chapters:\n"
        "- Routine actions (checking equipment, packing, eating, testing weapons)\n"
        "- Combat blow-by-blow narration\n"
        "- Dialogue and speech\n"
        "- Sensory descriptions and atmosphere\n"
        "- Emotional reactions\n"
        "- Any action whose omission would not cause a future contradiction\n"
        "\n"
        "CHAPTER TEXT:\n"
        "{chapter_text}\n"
        "\n"
        "CONTINUITY RECORD (200-400 words):")

    chapter_summary_prompt = PromptTemplate(
        input_variables=["chapter_text"],
        template=chapter_summary_prompt_template
    )

    if not whole_new_chapter.strip():
        print("Error: no chapter content was generated — skipping summary.")
        return

    summary_start = time.time()
    try:
        new_summary_message = llm_invoke(
            llm,
            chapter_summary_prompt.format(chapter_text=whole_new_chapter),
            "Chapter summary",
            args.context_size,
        )
    except Exception as e:
        print(f"Error generating chapter summary: {e}")
        return

    if new_summary_message:
        new_summary_text = new_summary_message.content.strip()
        word_count = len(whole_new_chapter.split())
        summary_word_count = len(new_summary_text.split())

        with open(new_summary_file, "w", encoding="utf-8") as f:
            f.write(new_summary_text + "\n")

        print(f"\nChapter {next_chapter_num} consists of {word_count} words.")
        print(f"Chapter {next_chapter_num} summary consists of {summary_word_count} words.")

        summary_end = time.time()

        summary_tokens = 0
        summary_input_tokens = 0
        if metadata := getattr(new_summary_message, 'response_metadata', {}):
            summary_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
            summary_input_tokens = metadata.get('token_usage', {}).get('prompt_tokens', 0)
        summary_metrics = {"duration": summary_end - summary_start, "tokens": summary_tokens, "input_tokens": summary_input_tokens}
    else:
        print("Error generating summary: No response from LLM")
        summary_metrics = None

    # Calculate and display metrics
    end_time = time.time()

    def format_time(elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"\nLoading story background and chapter summaries: {background_tokens} tokens (context_size={background_tokens})")

    total_pp_tokens = 0
    total_tg_tokens = 0
    total_gen_duration = 0

    for i, metrics in enumerate(chunk_metrics):
        duration = metrics["duration"]
        summary_duration = metrics.get("summary_duration", 0)
        input_tokens = metrics["input_tokens"]
        output_tokens = metrics["output_tokens"]
        summary_tokens = metrics.get("summary_tokens", 0)
        summary_input_tokens = metrics.get("summary_input_tokens", 0)
        total_pp_tokens += input_tokens + summary_input_tokens
        total_tg_tokens += output_tokens + summary_tokens
        total_gen_duration += duration + summary_duration
        tps = output_tokens / duration if duration > 0 else 0
        sum_tps = summary_tokens / summary_duration if summary_duration > 0 else 0
        print(f"Generating chunk {i+1} of {len(chunk_metrics)}: {format_time(duration)} | pp={input_tokens} tg={output_tokens} tokens ({tps:.2f} t/s)")
        print(f"  Rolling summary {i+1}: {format_time(summary_duration)} | pp={summary_input_tokens} tg={summary_tokens} tokens ({sum_tps:.2f} t/s)")

    if summary_metrics:
        duration = summary_metrics["duration"]
        tokens = summary_metrics["tokens"]
        input_tokens = summary_metrics.get("input_tokens", 0)
        total_pp_tokens += input_tokens
        total_tg_tokens += tokens
        total_gen_duration += duration
        tps = tokens / duration if duration > 0 else 0
        print(f"Summarizing story: {format_time(duration)} | pp={input_tokens} tg={tokens} tokens ({tps:.2f} t/s)")

    if total_gen_duration > 0:
        avg_tps = total_tg_tokens / total_gen_duration
        print(f"\nGeneration Summary:\n  Model used:    {actual_model_name}\n  PP tokens:     {total_pp_tokens}\n  TG tokens:     {total_tg_tokens}\n  Total time:    {format_time(total_gen_duration)}\n  Avg TG perf:   {avg_tps:.2f} t/s")

    print(f"\nTotal execution time: {format_time(end_time - start_time)}")

if __name__ == "__main__":
    main()
