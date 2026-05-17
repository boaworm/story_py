# Suiltable for qwen3.5 122b a10b
# Usage: ./runQwen3.5.sh        — generate next chapter
#        ./runQwen3.5.sh 33     — regenerate chapter 33
if [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
else
    chapter_arg="--instructions instructions.txt"
fi

python story_writer.py \
	--working-dir story_py_private/dnd1 \
	--api_url http://192.168.1.143:8000/v1/ \
	--model /models/qwen3.5-122b-int4-autoround \
	$chapter_arg \
	--key_event_chunk_size 4 \
	--presence_penalty 0.3 \
	--frequency_penalty 0.85 \
	--top_p 0.9 \
	--temperature 1.20 \
	--top_k 100 \
	--min_p 0.09 \
	--disable-thinking
