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
        default=50000,
        help="The maximum size of each text chunk (in characters). Should be less than the model's context window.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=5000,
        help="The number of characters to overlap between chunks to maintain context.",
    )
    parser.add_argument(
        "--summary_length",
        type=int,
        default=70000,
        help="The desired length of the final output in tokens. This parameter is now a hard target for the 'refine' prompt.",
    )
    parser.add_argument(
        "--save_summary",
        type=str,
        required=True, # This is now a required argument for the new streaming logic
        help="File path to save the generated detailed narrative to. This file will be appended to as the narrative is built.",
    )
    parser.add_argument(
        "--new_story",
        type=str,
        default=None,
        help="Optional file path to save the final output after instructions are applied.",
    )
    
    args = parser.parse_args()

    # Verify that the files exist
    if not Path(args.story).is_file():
        print(f"Error: The story file '{args.story}' was not found.")
        return
    if not Path(args.instructions).is_file():
        print(f"Error: The instructions file '{args.instructions}' was not found.")
        return

    print("Loading documents and instructions...")

    # 2. Document and Instruction Loading
    # ==============================================================================
    # Load the large story file
    with open(args.story, "r", encoding="utf-8") as f:
        story_text = f.read()

    # Load the instructions from the instructions file
    with open(args.instructions, "r", encoding="utf-8") as f:
        instructions_text = f.read()

    # 3. Text Splitting (Sliding Window Logic)
    # ==============================================================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    docs = text_splitter.create_documents([story_text])

    print(f"Original text split into {len(docs)} chunks.")

    # 4. LLM Setup
    # ==============================================================================
    try:
        llm = OllamaLLM(
            model=args.model,
            base_url=args.ollama_url
        )
    except Exception as e:
        print(f"Failed to connect to Ollama. Please ensure the service is running at {args.ollama_url} and the model '{args.model}' is downloaded.")
        print(f"Error: {e}")
        return

    # 5. Narrative Retelling Process (Refine Chain for Full Narrative)
    # ==============================================================================
    length_constraint_str = f"The final narrative should be a detailed, comprehensive retelling of around {args.summary_length} tokens."

    # Define a prompt for the initial chunk
    initial_prompt_template = """
    You are an expert at creating detailed narratives from source material. Create a comprehensive retelling of the following text, focusing on all key plot points, characters, and settings. Your goal is to be expansive, not concise.
    {length_constraint}
    
    Text: "{text}"
    
    DETAILED NARRATIVE:"""
    initial_prompt = PromptTemplate(
        input_variables=["text", "length_constraint"],
        template=initial_prompt_template
    )

    # Define a prompt for refining the narrative with subsequent chunks
    refine_prompt_template = """
    You are an expert at creating detailed narratives. Your task is to continue the following narrative by seamlessly integrating a new chunk of text.
    Your goal is to build upon the existing story. Do not condense; instead, build upon the existing story with details from the new text.
    
    Existing Narrative:
    "{existing_answer}"
    
    New Text Chunk to Integrate:
    "{text}"
    
    Updated Detailed Narrative:"""
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"], # Removed summary_length from refine prompt
        template=refine_prompt_template
    )
    
    # Process the first chunk to create the initial narrative and write it to file
    print("Processing initial chunk to create the narrative...")
    try:
        initial_narrative = llm.invoke(
            initial_prompt.format(
                text=docs[0].page_content, 
                length_constraint=length_constraint_str
            )
        )
        with open(args.save_summary, "w", encoding="utf-8") as f:
            f.write(initial_narrative)
        current_narrative = initial_narrative
    except Exception as e:
        print("Error processing the first chunk.")
        print(f"Reason: {e}")
        return

    # Process subsequent chunks using the refine prompt, appending to the file
    total_chunks = len(docs)
    for i in range(1, total_chunks):
        print(f"Refining narrative with chunk {i + 1} of {total_chunks}...")
        try:
            current_narrative = llm.invoke(
                refine_prompt.format(
                    existing_answer=current_narrative,
                    text=docs[i].page_content,
                )
            )
            with open(args.save_summary, "a", encoding="utf-8") as f: # Appending to the file
                f.write(current_narrative)

            # Reset current_narrative to just the last output to keep it within the context window
            current_narrative = current_narrative
            
        except Exception as e:
            print(f"Error refining narrative with chunk {i + 1}.")
            print(f"Reason: {e}")
            break # Stop processing if an error occurs

    # 6. Apply Instructions to the Final Narrative (Separate Step)
    # ==============================================================================
    print("\n" + "="*50)
    print("Applying final instructions to the detailed narrative...")
    print("="*50)

    # Reload the entire narrative from the file, as it may be too large to hold in memory
    try:
        with open(args.save_summary, "r", encoding="utf-8") as f:
            final_narrative = f.read()
    except Exception as e:
        print(f"Error: Could not read the saved narrative file '{args.save_summary}'.")
        print(f"Reason: {e}")
        return

    # This prompt is kept to ensure no additional commentary is added
    instructions_prompt_template = """
    You have a detailed narrative and a set of instructions.
    Your task is to apply these instructions to the narrative and provide only the result.
    Do not add any additional sections, commentary, or new content not directly requested by the instructions.

    Narrative:
    "{narrative}"

    Instructions:
    "{instructions}"

    TASK RESULT:"""
    
    instructions_prompt = PromptTemplate.from_template(instructions_prompt_template)
    
    final_output = llm.invoke(
        instructions_prompt.format(
            narrative=final_narrative, 
            instructions=instructions_text
        )
    )

    print(final_output)

    # 7. Optional: Write the final output to a file if --new_story is provided
    # ==============================================================================
    if args.new_story:
        try:
            with open(args.new_story, "w", encoding="utf-8") as f:
                for line in final_output.splitlines():
                    f.write(line + "\n")
            print(f"\nFinal output successfully written to '{args.new_story}'.")
        except Exception as e:
            print(f"\nError: Could not write final output to '{args.new_story}'.")
            print(f"Reason: {e}")

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
