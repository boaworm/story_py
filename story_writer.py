# story_writer.py

# Import necessary libraries
import argparse
import datetime
import os
import sys
import types
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
from pathlib import Path
import re

# Named constant for magic number
MAX_KEY_EVENTS_PER_CHUNK = 5

_think_prefix = "/no_think\n"
_invocation_log = []




def llm_invoke(llm, prompt, label):
    full_prompt = _think_prefix + prompt
    chars_in = len(full_prompt)
    t_start = datetime.datetime.now()
    t_first_token = None

    content_parts = []
    response_metadata = {}
    additional_kwargs = {}
    reasoning_parts = []

    for chunk in llm.stream(full_prompt):
        chunk_content = chunk.content or ''
        if t_first_token is None and chunk_content:
            t_first_token = datetime.datetime.now()
        content_parts.append(chunk_content)
        if getattr(chunk, 'response_metadata', None):
            response_metadata = chunk.response_metadata
        for k, v in (getattr(chunk, 'additional_kwargs', None) or {}).items():
            if k == 'reasoning_content':
                reasoning_parts.append(v)
            else:
                additional_kwargs[k] = v

    t_end = datetime.datetime.now()
    if reasoning_parts:
        additional_kwargs['reasoning_content'] = ''.join(reasoning_parts)

    response = types.SimpleNamespace(
        content=''.join(content_parts),
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
    )

    chars_out = len(response.content)
    token_usage = response_metadata.get('token_usage', {})
    tokens_in = token_usage.get('prompt_tokens', 0) or (chars_in // 4)
    tokens_out = token_usage.get('completion_tokens', 0) or (chars_out // 4)

    _invocation_log.append({
        "label": label,
        "start": t_start, "first_token": t_first_token, "end": t_end,
        "chars_in": chars_in, "chars_out": chars_out,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
    })
    print(f"  {label}")
    #print(f"  [DEBUG raw response] {response!r}")
    if not getattr(response, 'content', '').strip():
        extra = getattr(response, 'additional_kwargs', {}) or {}
        reasoning = extra.get('reasoning_content', '')
        other_keys = [k for k in extra if k != 'reasoning_content']
        if reasoning:
            print(f"  WARNING: {label} — empty content but reasoning_content present "
                  f"({len(reasoning.split())} words). Model spent tokens on thinking "
                  "with nothing left for the answer. Try --disable-thinking.")
        else:
            print(f"  WARNING: {label} — empty content, no reasoning_content. "
                  f"additional_kwargs keys: {list(extra.keys()) or 'none'}, "
                  f"response_metadata: {getattr(response, 'response_metadata', {})}")
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


def _print_benchmark_table(total_start, total_end):
    def fmt_clock(dt):
        return dt.strftime("%H:%M:%S")

    def fmt_elapsed(delta):
        s = int(delta.total_seconds())
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def tps(tokens, delta):
        secs = delta.total_seconds()
        return f"{tokens / secs:6.1f}" if secs > 0 and tokens > 0 else "   N/A"

    col_label = max(len("Invocation"), max((len(e["label"]) for e in _invocation_log), default=0))
    header = (
        f"{'Invocation':<{col_label}}  {'Start':>8}  {'Stop':>8}  {'Elapsed':>8}"
        f"  {'Chars In':>10}  {'Chars Out':>10}"
        f"  {'Tok In':>8}  {'TokIn/s':>8}"
        f"  {'Tok Out':>8}  {'TokOut/s':>9}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("LLM Invocation Benchmark")
    print("=" * len(header))
    print(header)
    print(sep)

    total_chars_in = total_chars_out = total_tok_in = total_tok_out = 0
    total_prefill = datetime.timedelta()
    total_decode = datetime.timedelta()

    for e in _invocation_log:
        total_chars_in += e["chars_in"]
        total_chars_out += e["chars_out"]
        total_tok_in += e["tokens_in"]
        total_tok_out += e["tokens_out"]

        ft = e["first_token"]
        prefill = (ft - e["start"]) if ft else (e["end"] - e["start"])
        decode  = (e["end"] - ft)   if ft else datetime.timedelta(0)
        total_prefill += prefill
        total_decode  += decode

        print(
            f"{e['label']:<{col_label}}  {fmt_clock(e['start']):>8}  {fmt_clock(e['end']):>8}"
            f"  {fmt_elapsed(e['end'] - e['start']):>8}"
            f"  {e['chars_in']:>10,}  {e['chars_out']:>10,}"
            f"  {e['tokens_in']:>8,}  {tps(e['tokens_in'], prefill):>8}"
            f"  {e['tokens_out']:>8,}  {tps(e['tokens_out'], decode):>9}"
        )

    print(sep)
    total_elapsed = total_end - total_start
    print(
        f"{'TOTAL':<{col_label}}  {fmt_clock(total_start):>8}  {fmt_clock(total_end):>8}"
        f"  {fmt_elapsed(total_elapsed):>8}"
        f"  {total_chars_in:>10,}  {total_chars_out:>10,}"
        f"  {total_tok_in:>8,}  {tps(total_tok_in, total_prefill):>8}"
        f"  {total_tok_out:>8,}  {tps(total_tok_out, total_decode):>9}"
    )
    print("=" * len(header))


def main():
    """
    Main function to parse arguments, load files, and execute the summarization and instruction chains.
    This version uses a 'refine' chain for a more detailed, longer output.
    """
    # Record the start time of the script
    start_time = time.time()
    run_start_dt = datetime.datetime.now()
    
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
        "--model",
        type=str,
        default=None,
        help="Model name/ID to pass to the LLM API (default: 'default').",
    )

    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        action="store_true",
        default=None,
        dest="enable_thinking",
        help="Send enable_thinking=true in the request body (Qwen3/vLLM thinking models).",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Send enable_thinking=false in the request body to suppress thinking mode.",
    )

    args = parser.parse_args()

    global _think_prefix
    _think_prefix = "" if args.enable_thinking is False else "/no_think\n"

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
        "model": args.model if args.model is not None else "default",
        "max_tokens": 32768,
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
    if args.enable_thinking is not None:
        # vLLM requires this nested under chat_template_kwargs for Qwen3 thinking models
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
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
        "Write natural, fluent prose. Never drop articles (a, an, the) or prepositions — every noun phrase must be complete. Write 'into the sun', not 'into sun'; 'by the fire', not 'by fire'. Missing articles make prose sound like a telegram or a game log, not a story.\n"
        "The key events are GM notes written in game language. Translate them into narrative — never copy game-mechanic phrases like 'checks for traps', 'scans for weakness', or 'uses healing spell' verbatim into prose. Instead show what the character actually does, sees, and feels.\n"
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

    background_tokens = len(full_summary_text) // 4
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
            new_story_section = llm_invoke(llm, prompt, f"Chunk {idx+1}/{len(event_chunks)}")
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
        if not input_tokens:
            input_tokens = len(prompt) // 4
        if not output_tokens:
            output_tokens = len(new_story_section.content) // 4

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
            )
            summary_duration = time.time() - summary_start_time
            chunk_summary_text = chunk_summary_msg.content.strip()
            summary_plus_new_story += f"\n{chunk_summary_text}"
            if (meta := getattr(chunk_summary_msg, 'response_metadata', {})) and 'token_usage' in meta:
                summary_tokens = meta['token_usage'].get('completion_tokens', 0)
                summary_input_tokens = meta['token_usage'].get('prompt_tokens', 0)
            if not summary_input_tokens:
                summary_input_tokens = len(chunk_summary_prompt.format(chunk_text=new_story_section.content.strip())) // 4
            if not summary_tokens:
                summary_tokens = len(chunk_summary_text) // 4
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
        if not summary_input_tokens:
            summary_input_tokens = len(chapter_summary_prompt.format(chapter_text=whole_new_chapter)) // 4
        if not summary_tokens:
            summary_tokens = len(new_summary_text) // 4
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

    run_end_dt = datetime.datetime.now()
    _print_benchmark_table(run_start_dt, run_end_dt)
    print(f"\nTotal execution time: {format_time(end_time - start_time)}")

if __name__ == "__main__":
    main()
