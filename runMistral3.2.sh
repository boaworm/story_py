#source story_env/bin/activate
# Appropriate for mistral3.2 24b
python story_writer.py \
	--story story_background.txt \
	--instructions instructions.txt \
	--key_event_chunk_size 4 \
	--temperature 0.7 \
	--top_p 0.9 \
	--top_k 40 \
	--repeat_penalty 1.1
