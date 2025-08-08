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


# Sample run


