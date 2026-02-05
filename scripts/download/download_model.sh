# MODEL_NAME=llava-hf/llava-1.5-7b-hf
# MODEL_NAME=AI-ModelScope/instructblip-vicuna-7b
# MODEL_NAME=skyline2006/llama-7b
MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
python3 scripts/download/download_model.py --base_model $MODEL_NAME