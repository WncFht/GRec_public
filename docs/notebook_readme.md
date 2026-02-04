# Notebook 说明

Notebook 主要用于数据检查、离线分析与小规模实验，集中在仓库的 `notebook/` 目录。

建议使用你自己的虚拟环境安装：

```bash
pip install jupyter notebook
```

当前常用 notebook：

1. `notebook/data.ipynb`：查看数据集结构/字段、简单 sanity check
2. `notebook/instruments_i2i_experiments.ipynb`：对比不同 embedding 在协同信息维度的表现（I2I 分析）
3. `notebook/output.ipynb`：对比不同 `base_model` / 微调后模型在 beam search 下的输出与分数分布
4. `notebook/dataset/`：与 DataLoader/num_workers 相关的性能测试与调参记录

