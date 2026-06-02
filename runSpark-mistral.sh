# Suitable for Mistral models on Spark (192.168.1.143)
# Note: no --enable/disable-thinking — Mistral tokenizers don't support chat_template_kwargs
# Usage: ./runSpark-mistral-large-2411.sh        — generate next chapter
#        ./runSpark-mistral-large-2411.sh 33     — regenerate chapter 33
if [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
else
    chapter_arg="--instructions instructions.txt"
fi

# min_p is not supported with EAGLE speculative decoding — detect at runtime
model_name=$(curl -s http://192.168.1.143:8000/v1/models \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
    || echo "unknown")
if echo "$model_name" | grep -qi "eagle"; then
    min_p_arg=""
    echo "Note: EAGLE speculative decoding detected — min_p disabled"
else
    min_p_arg="--min_p 0.05"
fi

PYTHONUNBUFFERED=1 python story_writer.py \
	--working-dir story_py_private/dnd1 \
	--template-model mistral \
	--api_url http://192.168.1.143:8000/v1/ \
	$chapter_arg \
	$min_p_arg \
	--key_event_chunk_size 2 \
	--temperature 1.2 \
	--top_p 0.87 \
	--top_k 100 \
	--presence_penalty 0.2 \
	--frequency_penalty 0.05 \
	--repeat_penalty 1.1 \
	--min_tokens 400 \
	--max_tokens 4000
