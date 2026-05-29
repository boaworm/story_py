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
- **Maintenance**: **You should manually update this file periodically.** When a major event happens (e.g., a character gains a new ability, a city is discovered, or a new main character joins), add a concise note here. This ensures the LLM never "forgets" these foundational changes in future chapters.

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
**ATTENTION**: Make sure you write your key events in chunks of 4 senetences. If you have a chunk size of 4 and you provide 9 key event lines, the last line alone will make up a whole section, and likely be a lot more "out of scope" than intended. 

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

Or use the convenience wrapper script:

```bash
./runQwen3.5.sh              # Generate new chapter
./runQwen3.5.sh 33           # Regenerate chapter 33
./runQwen3.5.sh --fix 51     # Review and fix chapter 51
```

### Recommended Parameters

> **Inference backend matters.** The parameter values below are tuned for models served via **vLLM**. vLLM implements OpenAI's sampling spec strictly, while llama.cpp interprets several of the same fields very differently (most notably `frequency_penalty` and the default `repeat_penalty`). A config that produces good prose on one backend can produce run-on, period-less text on the other. See **Backend Notes** below before copying these settings to a llama.cpp setup.

**For gemma3-27b at q8**
```bash
python story_writer.py \
   --api_url http://localhost:1234 \
   --story story_background.txt \
   --instructions instructions.txt \
   --key_event_chunk_size 4 \
   --temperature 0.1 \
   --top_k 100 \
   --repeat_penalty 1.1
```

**For qwen3.5-122b at q5 (vLLM)**
```bash
python story_writer.py \
   --api_url http://localhost:1234 \
   --story story_background.txt \
   --instructions instructions.txt \
   --key_event_chunk_size 4 \
   --temperature 1.00 \
   --presence_penalty 0.3 \
   --frequency_penalty 0.05 \
   --repeat_penalty 1.1 \
   --top_p 0.9 \
   --top_k 100 \
   --min_p 0.09 \
   --disable-thinking
```

**Notes:**
- Write your story outline in multiples of `--key_event_chunk_size` (default 4) to avoid a lone event forming its own under-developed section.
- `gemma3-27b` produces the best local results. `gemma4-31b` was tested but produces noticeably more compact, mechanical prose — avoid it for creative writing.
- `qwen3.5-122b` at 122B produces output quality comparable to gemma3-27b, sometimes better, at the cost of requiring a remote/high-VRAM machine.

### Backend Notes: vLLM vs llama.cpp

The configuration above (and the values baked into `runQwen3.5.sh`) was iterated against a **vLLM** server. If you point the same script at a llama.cpp `llama-server`, expect the model to behave differently even with byte-identical request bodies. The main divergences we hit during tuning:

- **`frequency_penalty`** — vLLM applies the strict OpenAI semantics (per-step logit reduction of `freq_penalty × occurrence_count`). High values quickly suppress the most common tokens, including `.`, until the model literally cannot end a sentence and the output collapses into a run-on. llama.cpp's OpenAI-compat layer historically applies this much more weakly (or near no-op), so a value like `0.85` is harmless there and lethal on vLLM. We settled at `0.05` on vLLM and rely on `repeat_penalty` for actual repetition control.
- **`repeat_penalty` default** — llama.cpp uses `1.1` as a baked-in default even when the request doesn't include the field. vLLM defaults to `1.0` (no penalty). If you port a llama.cpp config to vLLM and don't pass `--repeat_penalty` explicitly, you'll lose repetition control entirely.
- **`chat_template_kwargs.enable_thinking`** — vLLM forwards this to the tokenizer's chat template (Qwen3-style thinking on/off). llama.cpp ignores it. The `--disable-thinking` flag sets both this and the `/no_think` prompt prefix, so it works on both, but only the prefix takes effect on llama.cpp.
- **Stop tokens** — vLLM honours `stop` strings exactly; llama.cpp also stops on the model's EOS token regardless. If you ever see runaway output on vLLM, double-check that the stop list matches the model family (Qwen uses `<|im_end|>` / `<|endoftext|>`, not the Llama-3 tokens hard-coded by default).
- **Debugging** — start vLLM with `--enable-log-requests`; it then logs the resolved `SamplingParams` per request, which is the only reliable way to confirm what actually reached the sampler. llama.cpp's server logs are less explicit.

If you're running llama.cpp and the output looks fine, leave the existing penalties alone — the historical config (`frequency_penalty 0.85`, no explicit `repeat_penalty`) was originally tuned there. Just don't switch backends without also revisiting these numbers.

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--story` | (Required) | Path to the static lore/background text file. |
| `--instructions`| (Required) | Path to the file containing instructions for the final output. |
| `--api_url` | `http://localhost:1234/v1`| The URL for the local LLM service. |
| `--key_event_chunk_size` | `5` | Number of key events to process in each iteration. Lower = longer story. |
| `--chapter` | `None` | Specify a chapter number to rebuild using previous summaries. |
| `--fix` | `None` | Review and propose corrections for an existing chapter. |

### Fix Mode

After generating a chapter (or at any time), you can review and fix inconsistencies:

```bash
./runQwen3.5.sh --fix 51
# or
python story_writer.py --fix 51 --working-dir story_py_private/dnd1
```

This will:
1. Load `chapter51_story.txt`
2. Display the **original story**
3. Show **proposed changes** in colored diff format (red for deletions, green for additions)
4. Display the **full corrected story**
5. Prompt: `Accept changes and replace original (y/N)?`

Press `y` to overwrite the chapter with corrections, or any other key to keep the original.

Common issues detected:
- Character name inconsistencies
- Repeated/duplicated content
- Contradictions within the chapter
- Grammar and typo issues

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

