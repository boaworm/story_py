# Suitable for gemma4-31b-it 16bit on Spark (192.168.1.143)
# Usage: ./runSparkGemma4.sh        — generate next chapter
#        ./runSparkGemma4.sh 33     — regenerate chapter 33
if [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
else
    chapter_arg="--instructions instructions.txt"
fi

python story_writer.py \
	--working-dir story_py_private/dnd1 \
	--api_url http://192.168.1.143:8000/v1/ \
	--model /models/gemma4-31b-it \
	$chapter_arg \
	--key_event_chunk_size 3 \
	--temperature 0.9 \
	--top_k 60 \
	--repeat_penalty 1.05 \
	--min_tokens 600 \
	--disable-thinking
