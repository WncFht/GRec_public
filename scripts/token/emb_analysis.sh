python -m src.token.tsne \
    --model_path ckpt/Instruments/Llava-onevision-emb-item2index,seqrec-1-qwen7B/checkpoint-14984 \
    --interactive \
    --sample_original 2000 \
    --method tsne
    # --lora_checkpoint ckpt/Instruments/Qwen2-VL-7B-Instruct-seqrec-mmitemenrich-lora-1-qwen7B/checkpoint-89892 \