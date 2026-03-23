# story_py
Write a continuously evolving children's story using LLMs, Python, and LangChain.

## Introduction

This Python script is designed to build an evolving children's story chapter by chapter. It uses LangChain to interface with a local LLM (like LM Studio) to process a static lore background, maintain context through previous chapter summaries, and generate new story sections based on specific "Key Events".

## How It Works

The script follows an automated chapter progression:
1. **Static Lore**: Defined in a background file (e.g., `story_background.txt`).
2. **Context Progression**: It automatically loads all existing `chapter*_summary.txt` files to maintain continuity.
3. **Instructions**: You provide a file (e.g., `instructions.txt`) containing the plot points for the next chapter.
4. **Generation**: The script chunks the key events and generates the story in sequence, saving it to `chapterN_story.txt`.
5. **Summarization**: After generating a chapter, it creates a dense factual summary and saves it to `chapterN_summary.txt` for use in future chapters.

## File Types & Structure

### 1. `story_background.txt` (The Anchor)
This is the most critical file for maintaining story coherence. It contains the static lore, world-building, and core character descriptions.
- **Why it matters**: While the script uses chapter summaries for context, those summaries can lose fine detail over time. The background file acts as the permanent "truth" for the LLM.
- **Maintenance**: **You should manually update this file periodically.** When a major event happens (e.g., a character gains a new ability, a city is destroyed, or a new main character joins), add a concise note here. This ensures the LLM never "forgets" these foundational changes in future chapters.

### 2. `instructions.txt` (The Driver)
This is where you tell the script what happens next. It must contain the "Key Events" section.
- **Chunking**: The `--key_event_chunk_size` parameter (default: 5) determines how many events are processed in a single LLM prompt. Lower numbers result in more detailed, longer chapters, while higher numbers produce more concise summaries.
- **Format**: Ensure events are listed between `START OF KEY EVENTS:` and `END OF KEY EVENTS:`.

### 3. `chapterN_story.txt` (The Output)
These are the generated story chapters. Once generated, they are not read again by the script directly; instead, the script relies on their corresponding summaries.

### 4. `chapterN_summary.txt` (The Context)
After each chapter is written, the script generates a dense factual summary.
- **Role**: All previous summaries are loaded at the start of a new chapter to provide the LLM with the "story so far".
- **Density**: These summaries are intentionally stripped of flowery language to maximize the amount of story context that can fit into the LLM's window.

### 5. `chapterN_instructions.txt` (The Archive)
The script automatically saves a copy of the instructions used for each chapter. This is useful for rebuilding chapters or tracking your original plot points.


### Prerequisites
- **Python**: Ensure you have Python installed.
- **Dependencies**: Install required packages via pip:
  ```bash
  pip install -r requirements.txt
  ```
- **Local LLM**: This script is optimized for **LM Studio**. By default, it looks for an OpenAI-compatible API at `http://localhost:1234/v1`.

### Hardware Note
The script is optimized for high-memory environments (like a Mac Studio M1 Max with 64GB RAM) to handle large context windows (up to 128k tokens). Depending on your hardware, ensure LM Studio's context length set appropriately.

## Usage

### Instructions Format
The instructions file must contain a specific section for the script to parse:

With --key_event_chunk_size 4, the script will consume four rows of key events at a time. Depending on the model you use, and the amount of output it produces, you will want to adjust this value. Gemma3-27b and chunk size 4 and 12 lines of key events produces a good childrens story taking 10-15 minutes to read.

```text
START OF KEY EVENTS:
Hoby finds a mysterious map.
The wind picks up, signaling a storm.
A dragon appears on the horizon.
They raise the sails and escapes the dragon.
END OF KEY EVENTS:
```

### Running the Script
Run the script using the following command structure:

```bash
python story_writer.py --story story_background.txt --instructions instructions.txt
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--story` | (Required) | Path to the static lore/background text file. |
| `--instructions`| (Required) | Path to the file containing instructions for the final output. |
| `--api_url` | `http://localhost:1234/v1`| The URL for the local LLM service. |
| `--key_event_chunk_size` | `5` | Number of key events to process in each iteration. Lower = longer story. |
| `--chapter` | `None` | Specify a chapter number to rebuild using previous summaries. |

## Workflow Example

1. Update `story_background.txt` with your world's base lore.
2. Create `instructions.txt` with the events for Chapter 1.
3. Run `python story_writer.py --story story_background.txt --instructions instructions.txt`.
   - This creates `chapter1_story.txt`, `chapter1_summary.txt`, and `chapter1_instructions.txt`.
4. Create a new `instructions.txt` for Chapter 2.
5. Run the same command again.
   - The script detects Chapter 1 is done and automatically creates Chapter 2, using the Chapter 1 summary as context.

## Performance
Performance depends on your GPU/NPU. On an M1 Max, generating a chapter with ~30KB of background context typically takes around 15 minutes. Using larger models like `gemma3:27b` is recommended for higher-quality storytelling.

# Sample run
story_writer.py --story story_background.txt --instructions instructions.txt --save_summary summary.txt --new_chapter new_chapter.txt --key_event_chunk_size 4 
Generated summary will be saved to: summary16.txt
Refining background story with chunk 1 of 3...
Refining background story with chunk 2 of 3...
Refining background story with chunk 3 of 3...

==================================================
Applying provided instructions to create a new chapter to the story...
==================================================
Generating story section for key events 1-4 [chunk 1 of 3]...
Generating story section for key events 5-8 [chunk 2 of 3]...
Generating story section for key events 9-11 [chunk 3 of 3]...
New chapter written to new_chapter.txt.

==================================================
The warm southern breeze filled the sails of *Sunstone* as Hoby steered her southward. The hold was brimming with sweet mangoes, juicy pineapples, and cool water drawn from mountain springs – provisions enough for a long voyage. But a shadow lingered in his mind – the dark cave on the island, and the unsettling feeling that a dangerous creature lurked within.

...

They were sailing towards Eärcaraxe, the lair of the dragon. And whatever awaited them there, they would face it together.


==================================================

Total script execution time: 00:45:36

