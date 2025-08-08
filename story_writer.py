# Import necessary libraries
import argparse
import os
from pathlib import Path
import time
import re

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
        description="Rewrite a large text file using LangChain and Ollama with a 'refine' chain."
    )
    parser.add_argument(
        "--story",
        type=str,
        required=True,
        help="Path to the large text file to be rewritten.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Path to the file containing instructions for the final output.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:27b",
        help="The Ollama model name to use (e.g., 'gemma3:27b').",
    )

    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="The URL for the Ollama service.",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=75000,
        help="The maximum size of each text chunk (in characters). Should be less than the model's context window.",
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=10000,
        help="The number of characters to overlap between chunks to maintain context.",
    )

    parser.add_argument(
        "--save_summary",
        type=str,
        default="summary.txt",
        help="File path to save the generated detailed narrative to. This file will be appended to as the narrative is built.",
    )

    parser.add_argument(
        "--new_chapter",
        type=str,
        default="new_chapter.txt",
        help="Optional file path to save the final output after instructions are applied.",
    )

    parser.add_argument(
        "--summary_length",
        type=int,
        default=75000,
        help="Optional parameter to define the length of the summary.",
    )
    
    parser.add_argument(
        "--key_event_chunk_size",
        type=int,
        default=5,
        help="Optional parameter to the number of key events to process in each iteration. Lower number = longer story.",
    )
    
    
    args = parser.parse_args()

    # Verify that the files exist
    if not Path(args.story).is_file():
        print(f"Error: The story file '{args.story}' was not found.")
        return
    if not Path(args.instructions).is_file():
        print(f"Error: The instructions file '{args.instructions}' was not found.")
        return

    # LLM Setup
    # ==============================================================================
    stop_tokens = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]
    try:
        llm = OllamaLLM(
            model=args.model,
            base_url=args.ollama_url,
            temperature=1,
            max_tokens=-1,
            num_predict=-1, # Generate max number of tokens
            stop=stop_tokens, # Prevent the input prompt from being returned
            num_ctx=128000, # 65000, # 32768        
        )
    except Exception as e:
        print(f"Failed to connect to Ollama. Please ensure the service is running at {args.ollama_url} and the model '{args.model}' is downloaded.")
        print(f"Error: {e}")
        return



    # Narrative Retelling Process (Refine Chain for Full Narrative)
    # ==============================================================================
    length_constraint_str = f"The final narrative should be a detailed, comprehensive retelling of around {args.summary_length} tokens."

    # Load the large story file
    with open(args.story, "r", encoding="utf-8") as f:
        story_text = f.read()

    # Define a prompt for the initial chunk
    initial_prompt_template = """
    You are an expert at creating detailed factual summaries from source material. 
    Create a comprehensive recap of the following text, focusing on all key plot points, characters, and settings. 
    Your goal is to be expansive, not concise.
    Remove all extra words, padding, repeated descriptions etc. 
    Retain factual statements about characters, places, events and encounters.
    The summary should not be written as a readable story, but as a detailed narrative that captures all essential elements.
    {length_constraint}
    
    Text: "{text}"
    
    DETAILED NARRATIVE:"""
    initial_prompt = PromptTemplate(
        input_variables=["text", "length_constraint"],
        template=initial_prompt_template
    )

    # Define a prompt for refining the narrative with subsequent chunks
    refine_prompt_template = """
    You are an expert at creating detailed summary. Your task is to continue the following summary by seamlessly integrating a new chunk of text.
    Your goal is to build upon the existing summary. 
    Do not condense; instead, build upon the existing story with details from the new text.
    Avoid adding details that are already present in the existing narrative.
    
    Current story summary:
    "{current_story}"
    
    New story to Integrate:
    "{new_story}"
    
    Updated Detailed Narrative:"""
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"], 
        template=refine_prompt_template
    )
    
    # Process the first chunk to create the initial narrative and write it to file
    print("Processing initial chunk to create the story background summary...")
    save_summary_path = args.save_summary
    if Path(save_summary_path).exists():
        save_summary_path = get_incremented_filename(save_summary_path)

    print(f"Generated summary will be saved to: {save_summary_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    docs = text_splitter.create_documents([story_text])

    print(f"Original text split into {len(docs)} chunks.")

    try:
        initial_narrative = llm.invoke(
            initial_prompt.format(
                text=docs[0].page_content, 
                length_constraint=length_constraint_str
            )
        )
        with open(save_summary_path, "w", encoding="utf-8") as f:
            f.write(initial_narrative)
        current_narrative = initial_narrative
    except Exception as e:
        print("'\nError processing the first chunk.")
        print(f"Reason: {e}")
        return

    # Process subsequent chunks using the refine prompt, appending to the file
    # As each chunk does not overlap (by much), we append to the summary file. 
    # This way, we can create a larger summary without overflowing the context. 
    total_chunks = len(docs)
    for i in range(1, total_chunks):
        print(f"Refining background story with chunk {i + 1} of {total_chunks}...")
        try:
            current_narrative = llm.invoke(
                refine_prompt.format(
                    current_story=current_narrative,
                    new_story=docs[i].page_content,
                )
            )
            with open(save_summary_path, "a", encoding="utf-8") as f: # Appending to the file
                f.write(current_narrative)
            
        except Exception as e:
            print(f"Error refining narrative with chunk {i + 1}.")
            print(f"Reason: {e}")
            break # Stop processing if an error occurs



    # Generate the new story chapter, expanding on story-background
    # ==============================================================================
    # Load the large story summary
    print("\n" + "="*50)
    print("Applying provided instructions to create a new chapter to the story...")
    print("="*50)

    full_summary_text = ""
    with open(args.save_summary, "r", encoding="utf-8") as f:
        full_summary_text = f.read()

    # Load the instructions from the instructions file
    with open(args.instructions, "r", encoding="utf-8") as f:
        instructions_text = f.read()

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


    # Truncate new chapter file
    Path(args.new_chapter).unlink(missing_ok=True)

    # Prompt template for each chunk
    chunk_prompt_template = (
        "INSTRUCTION\n"
        "You are an expert at writing engaging children's fantasy stories. \n"
        "Write a detailed, coherent chapter that covers the following key events. \n"
        "using the previous story as context. Do not repeat content. Use simple language.\n"
        "short to medium sentences, and references to Tolkien's works where appropriate. \n"
        "Avoid too complicated new words. \n"
        "Do not introduce new characters or events unless explicitly requested. \n"
        "If I describe a character, do not assume they are meeting them. \n"
        "If i describe a place, do not assume they will travel there.\n" 
        "If i describe a creature or other being, do not assume they will meet, see or encounter them. \n"
        "Do not take the story line beyond what is described in the key events. \n"
        "Do not add titles or headers.\n"
        "ONLY OUTPUT THE NEW STORY, directly related to the key events. Do not repeat anything from the previous story.\n"
        "BEGINNING OF BACKGROUND\n{previous_story}\n"
        "END OF BACKGROUND\n\n"
        "BEGINNING OF KEY EVENTS\n{key_events}\n"
        "END OF KEY EVENTS\n\n"
    )

    chunk_prompt = PromptTemplate(
        input_variables=["previous_story", "key_events"],
        template=chunk_prompt_template
    )

    whole_new_chapter = ""
    summary_plus_new_story = full_summary_text

    # Truncate new chapter file
    # Path(args.new_chapter).unlink(missing_ok=True)
    event_chunks = list(chunk_list(key_events, key_event_chunk_size))

    for idx, chunk in enumerate(event_chunks):
        print(f"Generating story section for key events {idx*key_event_chunk_size+1}-{idx*key_event_chunk_size+len(chunk)} [chunk {idx+1} of {len(event_chunks)}]...")
        key_events_str = "\n".join(f"- {event}" for event in chunk)
        prompt = chunk_prompt.format(previous_story=summary_plus_new_story, key_events=key_events_str)
        try:
            new_story_section = llm.invoke(prompt)
            whole_new_chapter += new_story_section.strip() + "\n\n"
            summary_plus_new_story += "\nxx\n" + new_story_section.strip()
        except Exception as e:
            print(f"Error generating story section for chunk {idx+1}: {e}")
            break
    
    print(f"New chapter written to {args.new_chapter}.")
    with open(args.new_chapter, "w", encoding="utf-8") as f:
        f.write(whole_new_chapter.strip() + "\n\n")
        f.close()

    ## Print the whole new chapter to console    
    print("\n" + "="*50)
    print(whole_new_chapter)
    print("="*50)

    
    # 8. Calculate and print the total execution time
    # ==============================================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal script execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    main()
