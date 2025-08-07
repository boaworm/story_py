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

# Sample data
To make it easier to understand, i have put some sample data in the repo. Obviously you want to update this according to your own needs.


