#!/usr/bin/env bash
# 只需改下面 4 个变量即可
DATA_PATH="./data"       # 数据根目录，如 /data/rec
DATASET="Instruments"         # 数据集名称，如 Toys
INDEX_SUFFIX=".index_qwen7B.json"    # 索引文件后缀，如 .index.json
FIG_DIR="./figure"         # 图片保存目录，如 ./figs

python src/token/data_analysis.py "$DATA_PATH" "$DATASET" "$INDEX_SUFFIX" "$FIG_DIR"