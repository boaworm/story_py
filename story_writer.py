# Import necessary libraries
import argparse
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
from pathlib import Path
import re 

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
        default=5,
        help="Optional parameter to the number of key events to process in each iteration. Lower number = longer story.",
    )
    
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Optional. Specify a chapter number to rebuild. If provided, it reads chapterN_instructions.txt and builds story using previous chapter summaries.",
    )
    
    args = parser.parse_args()

    # Verify that the files exist
    if not Path(args.story).is_file():
        print(f"Error: The static lore file '{args.story}' was not found.")
        return
    if not Path(args.instructions).is_file():
        print(f"Error: The instructions file '{args.instructions}' was not found.")
        return

    # LLM Setup
    # ==============================================================================
    stop_tokens = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]
    try:
        llm = ChatOpenAI(
            openai_api_base=args.api_url,
            openai_api_key="lm-studio",  # LM Studio does not require a real key
            model="default",  # LM Studio uses whichever model is currently loaded
            temperature=1,
            max_tokens=-1,  # Adjust as needed for LM Studio
        )
        print(f"Connected to LLM at {args.api_url}")
    except Exception as e:
        print(f"Failed to connect to the LLM service at {args.api_url}.")
        print(f"Error: {e}")
        return



    # Narrative Retelling Process (Load Background and Previous Chapter Summaries)
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

    # Determine the chapter number to build
    if args.chapter is not None:
        next_chapter_num = args.chapter
        print(f"Rebuilding chapter {next_chapter_num} as requested by --chapter flag.")
    else:
        next_chapter_num = 1
        existing_chapters = []
        for file in glob.glob("chapter*_summary.txt"):
            match = re.search(r"chapter(\d+)_summary\.txt", file)
            if match:
                existing_chapters.append(int(match.group(1)))
        
        if existing_chapters:
            next_chapter_num = max(existing_chapters) + 1

    # Now load the ongoing story, only including chapters BEFORE the target chapter
    full_summary_text = static_lore + "\n\n"
    loaded_summaries = 0
    if summary_files:
        print(f"Loading ongoing story up to chapter {next_chapter_num - 1}:")
        for chapter_num, summary_file in summary_files:
            if chapter_num < next_chapter_num:
                print(f"- Loading {summary_file}")
                with open(summary_file, "r", encoding="utf-8") as f:
                    full_summary_text += f"\n--- Chapter {chapter_num} Summary ---\n"
                    full_summary_text += f.read() + "\n"
                loaded_summaries += 1
                
    if loaded_summaries == 0:
        print("No previous chapter summaries loaded. Starting fresh from static lore.")

    new_chapter_file = f"chapter{next_chapter_num}_story.txt"
    new_summary_file = f"chapter{next_chapter_num}_summary.txt"
    new_instructions_file = f"chapter{next_chapter_num}_instructions.txt"

    # Generate the new story chapter
    # ==============================================================================
    print("\n" + "="*50)
    print(f"Applying provided instructions to create {new_chapter_file}...")
    print("="*50)

    # Load the instructions
    if args.chapter is not None:
        # If rebuilding an existing chapter, we assume its instructions already exist.
        if Path(new_instructions_file).is_file():
            print(f"Reading target instructions from {new_instructions_file}")
            with open(new_instructions_file, "r", encoding="utf-8") as f:
                instructions_text = f.read()
        else:
            print(f"Error: Could not find {new_instructions_file} to rebuild chapter {next_chapter_num}.")
            return
    else:
        # If building a new chapter, read from args.instructions and copy it to the new_instructions_file
        with open(args.instructions, "r", encoding="utf-8") as f:
            instructions_text = f.read()
            
        if next_chapter_num > 1:
            prev_instructions_file = f"chapter{next_chapter_num - 1}_instructions.txt"
            if Path(prev_instructions_file).is_file():
                with open(prev_instructions_file, "r", encoding="utf-8") as f:
                    prev_instructions_text = f.read()
                if instructions_text.strip() == prev_instructions_text.strip():
                    print(f"No new instructions have been given. Current instructions were used to generate chapter {next_chapter_num - 1}.")
                    return

        # Copy the instructions to the chapter-specific instructions file
        with open(new_instructions_file, "w", encoding="utf-8") as f:
            f.write(instructions_text)
            print(f"Saved instructions to {new_instructions_file}")

    # Extract key events from instructions.txt
    key_events = []
    in_events = False
    for line in instructions_text.splitlines():
        if line.strip() == "START OF KEY EVENTS:":
            in_events = True
            continue
        if line.strip() == "END OF KEY EVENTS:":
            break
        if in_events:
            line = line.strip()
            if line:
                key_events.append(line)

    # Group key events into chunks of 5
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


    # Truncate new chapter file if it happens to exist
    Path(new_chapter_file).unlink(missing_ok=True)

    # Prompt template for each chunk
    chunk_prompt_template = (
        "INSTRUCTION\n"
        "You are an expert at writing engaging children's fantasy stories. \n"
        "Write a detailed, coherent chapter that covers the following key events IN THE EXACT ORDER LISTED. \n"
        "Each event MUST happen in sequence, exactly as listed. Do not mix up or reorder events.\n"
        "Using the previous story as context, continue the narrative maintaining strict chronological order.\n"
        "Use simple language, short to medium sentences.\n"
        "Each event should flow naturally into the next, following the exact sequence provided.\n"
        "Avoid too complicated new words.\n"
        "Do not introduce new characters or events unless explicitly requested.\n"
        "If I describe a character, do not assume they are meeting them.\n"
        "If i describe a place, do not assume they will travel there.\n"
        "If i describe a creature or other being, do not assume they will meet, see or encounter them.\n"
        "Do not take the story line beyond what is described in the key events.\n"
        "Do not add titles or headers.\n"
        "ONLY OUTPUT THE NEW STORY, directly related to the key events. Do not repeat anything from the previous story.\n"
        "BEGINNING OF BACKGROUND\n{previous_story}\n"
        "END OF BACKGROUND\n\n"
        "BEGINNING OF KEY EVENTS (MUST BE WRITTEN IN THIS EXACT ORDER)\n{key_events}\n"
        "END OF KEY EVENTS\n\n"
    )

    chunk_prompt = PromptTemplate(
        input_variables=["previous_story", "key_events"],
        template=chunk_prompt_template
    )

    whole_new_chapter = ""
    summary_plus_new_story = full_summary_text

    event_chunks = list(chunk_list(key_events, args.key_event_chunk_size))

    # Mark the end of the reading phase before we start generating
    reading_time_end = time.time()
    chunk_metrics = []
    actual_model_name = "Unknown"

    for idx, chunk in enumerate(event_chunks):
        chunk_start = time.time()
        print(f"Generating story section for key events {idx*args.key_event_chunk_size+1}-{idx*args.key_event_chunk_size+len(chunk)} [chunk {idx+1} of {len(event_chunks)}]...")
        key_events_str = "\n".join(f"- {event}" for event in chunk)
        prompt = chunk_prompt.format(previous_story=summary_plus_new_story, key_events=key_events_str)
        try:
            new_story_section = llm.invoke(prompt)
            whole_new_chapter += new_story_section.content.strip() + "\n\n"
            summary_plus_new_story += "\nxx\n" + new_story_section.content.strip()
        except Exception as e:
            print(f"Error generating story section for chunk {idx+1}: {e}")
            break
            
        chunk_end = time.time()
        duration = chunk_end - chunk_start
        
        # Capture token usage from LLM response metadata
        tokens = 0
        try:
            if hasattr(new_story_section, 'response_metadata') and 'token_usage' in new_story_section.response_metadata:
                tokens = new_story_section.response_metadata['token_usage'].get('completion_tokens', 0)
            else:
                # Fallback to word count approximation if metadata is missing
                tokens = count_tokens(new_story_section.content)
        except Exception:
            tokens = count_tokens(new_story_section.content)
            
        # Capture actual model name if available
        if hasattr(new_story_section, 'response_metadata'):
            actual_model_name = new_story_section.response_metadata.get('model_name', actual_model_name)
            
        chunk_metrics.append({"duration": duration, "tokens": tokens})
    
    print(f"New chapter written to {new_chapter_file}.")
    with open(new_chapter_file, "w", encoding="utf-8") as f:
        f.write(whole_new_chapter.strip() + "\n\n")

    ## Print the whole new chapter to console    
    print("\n" + "="*50)
    print(whole_new_chapter)
    print("="*50)

    # 7. Generate a summary of the newly written chapter
    # ==============================================================================
    print("\n" + "="*50)
    print(f"Generating factual summary of {new_chapter_file}...")
    print("="*50)

    chapter_summary_prompt_template = """
    You are an expert at creating detailed factual summaries from source material.
    Summarize the events of the following new chapter. 
    Retain all factual details, names, locations, items, places visited, experiences gained and actions. 
    Remove flowery language, dialogue, and pacing. 
    Make it a dense factual record of what happened in this exact chapter.
    Output ONLY the summary. Do not include any introductory text, titles, or conversational filler like "Here is the summary".
    
    CHAPTER TEXT:
    "{chapter_text}"
    
    DENSE FACTUAL SUMMARY:"""
    
    chapter_summary_prompt = PromptTemplate(
        input_variables=["chapter_text"],
        template=chapter_summary_prompt_template
    )

    summary_start = time.time()
    summary_end = summary_start
    try:
        new_summary_message = llm.invoke(
            chapter_summary_prompt.format(chapter_text=whole_new_chapter)
        )
        new_summary_text = new_summary_message.content.strip()
        
        with open(new_summary_file, "w", encoding="utf-8") as f:
            f.write(new_summary_text + "\n")
            
        print(f"Summary of new chapter written to {new_summary_file}.")
        summary_end = time.time()
        
        # Capture summary token usage
        summary_tokens = 0
        try:
            if hasattr(new_summary_message, 'response_metadata') and 'token_usage' in new_summary_message.response_metadata:
                summary_tokens = new_summary_message.response_metadata['token_usage'].get('completion_tokens', 0)
            else:
                summary_tokens = count_tokens(new_summary_text)
        except Exception:
            summary_tokens = count_tokens(new_summary_text)
            
        # Capture actual model name if available
        if hasattr(new_summary_message, 'response_metadata'):
            actual_model_name = new_summary_message.response_metadata.get('model_name', actual_model_name)
            
        summary_metrics = {"duration": summary_end - summary_start, "tokens": summary_tokens}
    except Exception as e:
        print(f"Error generating summary for the new chapter: {e}")
        summary_metrics = None

    
    # 8. Calculate and print the total execution time
    # ==============================================================================
    end_time = time.time()
    
    def format_time(elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"\nReading background and new instructions: {format_time(reading_time_end - start_time)}")
    
    total_gen_tokens = 0
    total_gen_duration = 0
    
    for i, metrics in enumerate(chunk_metrics):
        duration = metrics["duration"]
        tokens = metrics["tokens"]
        total_gen_tokens += tokens
        total_gen_duration += duration
        tps = tokens / duration if duration > 0 else 0
        print(f"Generating story chunk {i+1} of {len(chunk_metrics)}: {format_time(duration)} | {tokens} tokens ({tps:.2f} t/s)")
        
    if summary_metrics:
        duration = summary_metrics["duration"]
        tokens = summary_metrics["tokens"]
        total_gen_tokens += tokens
        total_gen_duration += duration
        tps = tokens / duration if duration > 0 else 0
        print(f"Summarizing full story: {format_time(duration)} | {tokens} tokens ({tps:.2f} t/s)")
        
    if total_gen_duration > 0:
        avg_tps = total_gen_tokens / total_gen_duration
        print(f"\nGeneration Summary:")
        print(f"  Model actually used:    {actual_model_name}")
        print(f"  Total tokens generated: {total_gen_tokens}")
        print(f"  Total generation time:  {format_time(total_gen_duration)}")
        print(f"  Average performance:    {avg_tps:.2f} t/s")

    print(f"\nTotal script execution time: {format_time(end_time - start_time)}")

if __name__ == "__main__":
    main()
