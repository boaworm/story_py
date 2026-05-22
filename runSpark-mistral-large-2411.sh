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
	--model current-spark-model \
	$chapter_arg \
	--key_event_chunk_size 20 \
	--temperature 0.72 \
	--top_p 0.88 \
	--top_k 50 \
	--min_p 0.05 \
	--repeat_penalty 1.07 \
	--min_tokens 100 \
	--max_tokens 4000 \
	--disable-thinking
