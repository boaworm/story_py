# Suitable for qwen3.5 122b a10b
# Usage: ./runSparkQwen3.5.sh        — generate next chapter
#        ./runSparkQwen3.5.sh 33     — regenerate chapter 33
if [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
else
    chapter_arg="--instructions instructions.txt"
fi

API_URL="http://192.168.1.143:8000/v1/"

owned_by=$(curl -s "${API_URL}models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0].get('owned_by',''))" 2>/dev/null)

if [[ "$owned_by" == "llamacpp" ]]; then
    echo "Detected engine: llama.cpp"
    python story_writer.py \
        --working-dir story_py_private/dnd1 \
        --api_url "$API_URL" \
        $chapter_arg \
        --key_event_chunk_size 3 \
        --frequency_penalty 0.85 \
        --presence_penalty 0.3 \
        --repeat_penalty 1.15 \
        --top_p 0.9 \
        --temperature 1.20 \
        --top_k 100 \
        --min_p 0.09 \
        --disable-thinking
else
    echo "Detected engine: vLLM"
    python story_writer.py \
        --working-dir story_py_private/dnd1 \
        --api_url "$API_URL" \
        $chapter_arg \
        --key_event_chunk_size 3 \
        --repeat_penalty 1.1 \
        --top_p 0.9 \
        --temperature 1.20 \
        --top_k 100 \
        --min_p 0.09 \
        --disable-thinking
fi
