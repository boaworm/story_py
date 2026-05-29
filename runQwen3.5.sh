# Suiltable for qwen3.5 122b a10b
# Usage: ./runQwen3.5.sh        — generate next chapter
#        ./runQwen3.5.sh 33     — regenerate chapter 33
#        ./runQwen3.5.sh --fix 51  — analyze chapter 51 for inconsistencies
if [[ "$1" == "--fix" && "$2" =~ ^[0-9]+$ ]]; then
    fix_arg="--fix $2"
    chapter_arg="--instructions instructions.txt"
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    chapter_arg="--regenerate $1"
    fix_arg=""
else
    chapter_arg="--instructions instructions.txt"
    fix_arg=""
fi

python story_writer.py \
	--working-dir story_py_private/dnd1 \
	--api_url https://www.thorburn.se/llama/v1/ \
	$chapter_arg \
	$fix_arg \
	--key_event_chunk_size 4 \
	--presence_penalty 0.3 \
	--frequency_penalty 0.05 \
	--repeat_penalty 1.1 \
	--top_p 0.9 \
	--temperature 1.20 \
	--top_k 100 \
	--min_p 0.09 \
	--disable-thinking
