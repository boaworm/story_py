# Suitable for mistral-large-2411-awq on Spark (192.168.1.143)
# Usage: ./runSpark-mistral-large-2411.sh        — generate next chapter
#        ./runSpark-mistral-large-2411.sh 33     — regenerate chapter 33
if [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
else
    chapter_arg="--instructions instructions.txt"
fi

python story_writer.py \
	--working-dir story_py_private/dnd1 \
	--api_url http://192.168.1.143:8000/v1/ \
	$chapter_arg \
	--key_event_chunk_size 3 \
	--temperature 0.4 \
	--top_p 0.9 \
	--top_k 100 \
	--min_p 0.09 \
	--presence_penalty 0.3 \
	--frequency_penalty 0.85 \
	--repeat_penalty 1.07 \
	--min_tokens 200 \
	--max_tokens 4000 \
	--disable-thinking
