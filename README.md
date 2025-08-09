# story_py
Write a continuously evolving childrens story using LLMs, Python, LangGraph

# Introduction

This is a python script that can be used to build an evolving childrens story.

You fill in the background in the story_background.txt file
You fill in your new events in the instructions.txt file
You then run the ./run.sh script 

Python will use LangGraph and the configured/local LLM to parse and learn the background story, after which it will print out the new story line.

You can then append the new story line/chapter to the background.txt file, update instructions.txt with new events, and keep going.

# Dependencies
This assumes you are running ollama locally, and have an LLM loaded. By default, it looks for gemma3:27b
Use the argument to change to a different model if you want.

Note that I have optimized this for a Mac Studio M1 Max with 64GB ram, so i am using the full 128k context window. This uses around 28GB of RAM. 

Depending on your hardware, you may want to change that. Locate the llm section and update as per your needs:

```
try:
        llm = OllamaLLM(
            model=args.model,
            base_url=args.ollama_url,
            temperature=1,
            max_tokens=-1,
            num_predict=-1, # Generate max number of tokens
            num_ctx=128000, # 65000, # 32768        
        )
```

# Sample data
To make it easier to understand, i have put some sample data in the repo. Obviously you want to update this according to your own needs.

# Performance
Will obviously depend on your GPU

With an M1 Max, it takes me 15 minutes to generate a new chapter at this point, with around 30kb of background.

You can obviously do this quite a lot faster using ChatGTP or Gemini or aistudio.google.com, but it is fun running it locally. And gemma3:27b is quite good at this (as opposed to gemma3:8b which writes miserable stories).

# Chunking

## Input chunking
--chunk_size allows you to et the number of tokens to handle when breaking up the input into smaller pieces. 
If your context window is smaller than your full story line, you set this to something of an appropriate size so your LLM Context can handle it.
By default it is 75000, as gemma3:27b can handle 128k tokens.

## Output chunking
Given the outout of each query to the LLM is limited in the number of characters it can produce, i have added a chunking concept.
By default, it is set to 5, meaning that it will run one prompt for each 5 key events you add in the instructions.txt file.

You can update the chunking size via a config parameter.

So if you set chunking to something higher, you will get less output. If you set chunking to something lower, you will get more output (longer new chapter.txt)

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


