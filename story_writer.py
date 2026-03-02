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
        required=True,
        help="Path to the static lore/background text file.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Path to the file containing instructions for the final output.",
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

    args = parser.parse_args()

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
    try:
        llm = ChatOpenAI(
            openai_api_base=args.api_url,
            openai_api_key="lm-studio",
            model="default",
            temperature=1,
            max_tokens=-1,
        )
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
                    print(f"No new instructions. Using previous for chapter {next_chapter_num - 1}")
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
        "Cover ONLY the key events listed below - nothing more, nothing less.\n"
        "Each event MUST happen sequentially as listed. Do not reorder.\n"
        "DO NOT repeat or rewrite any part of the previous story.\n"
        "DO NOT go beyond the last key event listed.\n"
        "DO NOT create a conclusion or wrap up the story unless the key events indicate the story ends.\n"
        "Use simple language, short to medium sentences.\n"
        "Do not introduce new characters or events unless requested.\n"
        "Do not add titles or headers.\n"
        "ONLY OUTPUT THE NEW CONTINUATION, directly related to key events below.\n"
        "BEGINNING OF BACKGROUND\n{previous_story}\n"
        "END OF BACKGROUND\n\n"
        "BEGINNING OF KEY EVENTS (MUST BE IN ORDER)\n{key_events}\n"
        "END OF KEY EVENTS\n\n"
        "Write the continuation now, starting immediately after where the previous text ended:\n"
    )

    chunk_prompt = PromptTemplate(
        input_variables=["previous_story", "key_events"],
        template=chunk_prompt_template
    )

    whole_new_chapter = ""
    summary_plus_new_story = full_summary_text

    event_chunks = list(chunk_list(key_events, args.key_event_chunk_size))

    reading_time_end = time.time()
    chunk_metrics = []
    actual_model_name = "Unknown"

    for idx, chunk in enumerate(event_chunks):
        chunk_start = time.time()
        key_events_str = "\n".join(f"- {event}" for event in chunk)
        prompt = chunk_prompt.format(
            previous_story=summary_plus_new_story,
            key_events=key_events_str
        )
        try:
            new_story_section = llm.invoke(prompt)
            whole_new_chapter += new_story_section.content.strip() + "\n\n"
            summary_plus_new_story += "\nxx\n" + new_story_section.content.strip()
        except Exception as e:
            print(f"Error generating chunk {idx+1}: {e}")
            break
        
        chunk_end = time.time()
        duration = chunk_end - chunk_start
        
        tokens = 0
        if (metadata := getattr(new_story_section, 'response_metadata', {})) and 'token_usage' in metadata:
            tokens = metadata['token_usage'].get('completion_tokens', 0)
        else:
            tokens = count_tokens(new_story_section.content)
        
        if metadata := getattr(new_story_section, 'response_metadata', {}):
            actual_model_name = metadata.get('model_name', actual_model_name)
        
        chunk_metrics.append({"duration": duration, "tokens": tokens})

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
        "You are an expert at creating detailed factual summaries.\n"
        "Summarize the events of the following new chapter.\n"
        "Retain all factual details, names, locations, items, places visited, experiences gained and actions.\n"
        "Remove flowery language, dialogue, and pacing.\n"
        "Make it a dense factual record of what happened.\n"
        "Output ONLY the summary. Do not include any introductory text or titles.\n"
        "\n"
        "CHAPTER TEXT:\n"
        "{chapter_text}\n"
        "\n"
        "DENSE FACTUAL SUMMARY:")

    chapter_summary_prompt = PromptTemplate(
        input_variables=["chapter_text"],
        template=chapter_summary_prompt_template
    )

    summary_start = time.time()
    if (new_summary_message := llm.invoke(
        chapter_summary_prompt.format(chapter_text=whole_new_chapter)
    )):
        new_summary_text = new_summary_message.content.strip()
        
        with open(new_summary_file, "w", encoding="utf-8") as f:
            f.write(new_summary_text + "\n")
        
        summary_end = time.time()
        
        summary_tokens = 0
        if metadata := getattr(new_summary_message, 'response_metadata', {}):
            summary_tokens = metadata.get('token_usage', {}).get('completion_tokens', 0)
        
        summary_metrics = {"duration": summary_end - summary_start, "tokens": summary_tokens}
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

    print(f"\nReading background and instructions: {format_time(reading_time_end - start_time)}")
    
    total_gen_tokens = 0
    total_gen_duration = 0
    
    for i, metrics in enumerate(chunk_metrics):
        duration = metrics["duration"]
        tokens = metrics["tokens"]
        total_gen_tokens += tokens
        total_gen_duration += duration
        tps = tokens / duration if duration > 0 else 0
        print(f"Generating chunk {i+1} of {len(chunk_metrics)}: {format_time(duration)} | {tokens} tokens ({tps:.2f} t/s)")
    
    if summary_metrics:
        duration = summary_metrics["duration"]
        tokens = summary_metrics["tokens"]
        total_gen_tokens += tokens
        total_gen_duration += duration
        tps = tokens / duration if duration > 0 else 0
        print(f"Summarizing story: {format_time(duration)} | {tokens} tokens ({tps:.2f} t/s)")
    
    if total_gen_duration > 0:
        avg_tps = total_gen_tokens / total_gen_duration
        print(f"\nGeneration Summary:\n  Model used:    {actual_model_name}\n  Total tokens:  {total_gen_tokens}\n  Total time:    {format_time(total_gen_duration)}\n  Avg performance: {avg_tps:.2f} t/s")

    print(f"\nTotal execution time: {format_time(end_time - start_time)}")

if __name__ == "__main__":
    main()
