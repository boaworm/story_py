#source story_env/bin/activate
# Appropriate for gemma3-27b q8
python story_writer.py \
	--story story_background.txt \
	--instructions instructions.txt \
	--key_event_chunk_size 4 \
	--temperature 0.1 \
	--top_k 100 \
	--repeat_penalty 1.1
