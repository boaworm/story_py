# story_writer.py

# Import necessary libraries
import argparse
import os
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
        default="story_background.txt",
        help="Path to the static lore/background text file (default: story_background.txt).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Path to the file containing instructions. Defaults to chapterN_instructions.txt when --regenerate is used.",
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
        "--min_tokens",
        type=int,
        default=None,
        help="Minimum tokens to generate per chunk before the model is allowed to stop.",
    )

    args = parser.parse_args()

    # Resolve instructions file
    if args.instructions is None:
        if args.regenerate is not None:
            args.instructions = f"chapter{args.regenerate}_instructions.txt"
        else:
            print("Error: --instructions is required unless --regenerate is used.")
            return

    # Verify that the files exist
    if not (story_path := Path(args.story)).is_file():
        print(f"Error: The static lore file '{{story_path}}' was not found.")
        return
    if not (instructions_path := Path(args.instructions)).is_file():
        print(f"Error: The instructions file '{{instructions_path}}' was not found.")
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
        "Before writing, silently assess the narrative significance of each event in this chunk. Major events — any combat, action, dramatic moment, revelation, or important character decision — deserve rich expansion (200–300 words each). All other events deserve at least 150 words — enough to feel immersive and real. Combat and action sequences are always major, even individual steps within a battle. Do not output this assessment.\n"
        "Expand each event meaningfully with sensory details, dialogue, and character thoughts. Do not pad, but do not rush either.\n"
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
            new_story_section = llm.invoke(prompt)
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
            chunk_summary_msg = llm.invoke(
                chunk_summary_prompt.format(chunk_text=new_story_section.content.strip())
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

    summary_start = time.time()
    if (new_summary_message := llm.invoke(
        chapter_summary_prompt.format(chapter_text=whole_new_chapter)
    )):
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
