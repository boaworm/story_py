# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Behavior Rule

**When in doubt, ask. Never guess or hallucinate.** If something is unclear — a file path, a chapter number, which model to use, what an instruction means — stop and ask the user rather than making assumptions. A wrong guess wastes a long LLM generation run.

## What This Project Does

`story_writer.py` generates children's fantasy stories chapter-by-chapter using a local LLM (via LM Studio or a remote OpenAI-compatible API). It feeds a static lore background + all previous chapter summaries as context, then generates new story text from a list of "key events," producing both a chapter file and a dense continuity summary for future chapters.

## Running the Script

The convenience shell scripts are the primary entry points. The active story lives in `story_py_private/dnd1/` (a git submodule):

```bash
./runQwen3.5.sh              # Generate next chapter
./runQwen3.5.sh 33           # Regenerate chapter 33
./runQwen3.5.sh --fix 51     # Review and fix chapter 51
```

Direct invocation:
```bash
python story_writer.py --story story_background.txt --instructions instructions.txt
python story_writer.py --fix 51 --working-dir story_py_private/dnd1
python story_writer.py --regenerate 33 --working-dir story_py_private/dnd1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

The virtual environment is in `story_env/`.

## Key CLI Arguments

| Argument | Default | Notes |
|---|---|---|
| `--story` | `story_background.txt` | Static lore file (required) |
| `--instructions` | `instructions.txt` | Key events file (required) |
| `--working-dir` | CWD | Directory containing story files |
| `--key_event_chunk_size` | `5` | Events per LLM call; lower = longer story |
| `--fix N` | — | Review/patch chapter N interactively |
| `--regenerate N` | — | Delete and rewrite chapter N |
| `--chapter N` | — | Rebuild chapter N without deleting it first |
| `--disable-thinking` | — | Suppress Qwen3 chain-of-thought via `/no_think` |
| `--enable-thinking` | — | Enable chain-of-thought (mutually exclusive with above) |
| `--context-size` | 131072 | Token context window |
| `--max_tokens` | 32768 | Max tokens per LLM response |

## Architecture

All logic is in `story_writer.py` (single file). Key flow:

1. **Argument parsing** — resolves `--working-dir`, then `os.chdir()` so all subsequent file I/O is relative.
2. **Chapter numbering** — auto-detects the next chapter by scanning existing `chapter*_summary.txt` files; `--chapter`/`--regenerate` override.
3. **Context assembly** — `static_lore + all chapter summaries up to N-1` → fed into every prompt.
4. **Chunked generation** — key events split into chunks of `--key_event_chunk_size`. Each chunk is generated, then immediately summarized (rolling context), so later chunks don't blow the context window.
5. **Post-generation fix pass** — after writing the chapter, `review_and_fix_chapter()` runs automatically at `temperature=0.1` (a separate `fix_llm` instance) and offers an interactive diff.
6. **Chapter summary** — a final LLM call produces a 200–400-word continuity record saved to `chapterN_summary.txt`.
7. **Benchmarking** — `_print_benchmark_table()` prints per-invocation timing, token counts, and throughput at the end.

## File Conventions (per working directory)

| File | Role |
|---|---|
| `story_background.txt` | Static lore/world-building; manually maintained |
| `instructions.txt` | Key events for the **next** chapter |
| `chapterN_story.txt` | Generated story output |
| `chapterN_summary.txt` | Dense continuity summary (auto-generated) |
| `chapterN_instructions.txt` | Archive of instructions used for chapter N |

## Instructions File Format

```
START OF KEY EVENTS:
Event one sentence.
Event two sentence.
Event three sentence.
Event four sentence.
END OF KEY EVENTS:
```

Write events in exact multiples of `--key_event_chunk_size` to avoid a lone final event becoming an under-developed section.

## Model Notes

- **gemma3-27b q8**: Best local quality; use `--temperature 0.1 --top_k 100 --repeat_penalty 1.1`.
- **qwen3.5-122b q5**: Comparable quality, needs remote/high-VRAM; use `--disable-thinking` and the penalty flags in `runQwen3.5.sh`.
- Avoid **gemma4-31b**: produces mechanical, compact prose.
- The script auto-detects the loaded model from `/models` endpoint if `--model` is not passed.

## Active Story

`story_py_private/` is a git submodule containing the private story files (background, chapters, summaries). The `runQwen3.5.sh` script points `--working-dir` there automatically.
