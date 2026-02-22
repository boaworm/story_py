#source story_env/bin/activate
python --version
python story_writer.py --story story_background.txt --model mistral-small-3.2 --instructions instructions.txt --save_summary summary.txt --new_chapter new_chapter.txt --key_event_chunk_size 4
