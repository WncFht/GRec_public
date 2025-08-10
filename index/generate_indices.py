import argparse
import collections
import json
import os

import numpy as np
import torch
from datasets import EmbDataset
from models.rqvae import RQVAE
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_collision(all_indices_str):
    """
    检查所有索引字符串是否存在碰撞（即重复）。

    参数:
        all_indices_str: 包含所有索引字符串的 NumPy 数组。

    返回:
        bool: 如果没有碰撞返回 True，否则返回 False。
    """
    tot_item = len(all_indices_str)  # 总项目数
    tot_indice = len(set(all_indices_str.tolist()))  # 唯一索引的数量
    return tot_item == tot_indice  # 如果总项目数等于唯一索引数，则没有碰撞


def get_indices_count(all_indices_str):
    """
    计算每个索引字符串出现的次数。

    参数:
        all_indices_str: 包含所有索引字符串的 NumPy 数组。

    返回:
        collections.defaultdict: 字典，键为索引字符串，值为其出现次数。
    """
    indices_count = collections.defaultdict(int)  # 使用 defaultdict 方便计数
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    """
    获取所有发生碰撞的项目的分组。

    参数:
        all_indices_str: 包含所有索引字符串的 NumPy 数组。

    返回:
        list: 列表中的每个元素是一个列表，包含发生碰撞的项目索引。
    """
    index2id = {}  # 字典，用于存储索引到项目ID的映射
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)  # 将项目ID添加到对应索引的列表中

    # 只保留有冲突的item（即出现次数大于1的索引）
    collision_item_groups = [
        index2id[index] for index in index2id if len(index2id[index]) > 1
    ]
    return collision_item_groups


def main(args):
    device = torch.device(args.device)

    # 加载模型检查点
    ckpt = torch.load(
        args.ckpt_path, map_location=torch.device("cpu"), weights_only=False
    )  # 加载检查点到 CPU
    model_args = ckpt["args"]  # 从检查点中获取训练参数
    state_dict = ckpt["state_dict"]  # 从检查点中获取模型状态字典

    data = EmbDataset(model_args.data_path)  # 加载嵌入数据集

    # 初始化 RQVAE 模型
    model = RQVAE(
        in_dim=data.dim,  # 输入维度
        num_emb_list=model_args.num_emb_list,  # 每层量化器的嵌入数量列表
        e_dim=model_args.e_dim,  # 嵌入维度
        layers=model_args.layers,  # RQVAE 层数
        dropout_prob=model_args.dropout_prob,  # Dropout 概率
        bn=model_args.bn,  # 是否使用 Batch Normalization
        loss_type=model_args.loss_type,  # 损失类型
        quant_loss_weight=model_args.quant_loss_weight,  # 量化损失权重
        kmeans_init=model_args.kmeans_init,  # 是否使用 KMeans 初始化
        kmeans_iters=model_args.kmeans_iters,  # KMeans 迭代次数
        sk_epsilons=model_args.sk_epsilons,  # Sinkhorn-Knopp 算法的 epsilon 值列表
        sk_iters=model_args.sk_iters,  # Sinkhorn-Knopp 算法的迭代次数
    )

    model.load_state_dict(state_dict)  # 加载模型状态字典
    model = model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置模型为评估模式
    print(model)  # 打印模型结构

    # 创建 DataLoader
    data_loader = DataLoader(
        data,
        num_workers=model_args.num_workers,  # 工作进程数
        batch_size=args.batch_size,  # 批处理大小
        shuffle=False,  # 不打乱数据
        pin_memory=True,  # 启用 pin_memory，加快数据传输到 GPU
    )

    all_indices = []  # 存储所有索引的列表
    all_indices_str = []  # 存储所有索引字符串的列表
    # 定义用于构建索引字符串的前缀
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # 遍历数据加载器，生成初始索引
    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(
            d, use_sk=False
        )  # 获取模型索引，不使用 Sinkhorn-Knopp
        # 将索引展平并转换为 numpy 数组
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(
                    prefix[i].format(int(ind))
                )  # 根据前缀和索引值构建代码

            all_indices.append(code)
            all_indices_str.append(str(code))
        # break

    all_indices = np.array(all_indices)  # 将索引列表转换为 NumPy 数组
    all_indices_str = np.array(
        all_indices_str
    )  # 将索引字符串列表转换为 NumPy 数组

    # 设置除最后一层外的 RQ 量化器的 Sinkhorn-Knopp epsilon 为 0
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    # model.rq.vq_layers[-1].sk_epsilon = 0.005  # 示例：最后一层 epsilon 值
    # 如果最后一层的 Sinkhorn-Knopp epsilon 为 0，则设置为 0.003
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0  # 迭代计数器
    # There are often duplicate items in the dataset, and we no longer differentiate them
    # 循环处理碰撞，直到没有碰撞或达到最大迭代次数
    while True:
        if tt >= 20 or check_collision(
            all_indices_str
        ):  # 如果达到最大迭代次数或没有碰撞，则退出循环
            break

        collision_item_groups = get_collision_item(
            all_indices_str
        )  # 获取碰撞的项目组
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)  # 获取发生碰撞的数据

            indices = model.get_indices(
                d, use_sk=True
            )  # 使用 Sinkhorn-Knopp 算法重新获取索引
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(
                collision_items, indices, strict=False
            ):  # 遍历碰撞项目和新的索引
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))  # 构建新的代码

                all_indices[item] = code  # 更新索引
                all_indices_str[item] = str(code)  # 更新索引字符串
        tt += 1

    print("All indices number: ", len(all_indices))  # 打印总索引数量
    print(
        "Max number of conflicts: ",
        max(get_indices_count(all_indices_str).values()),  # 打印最大冲突数量
    )

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print(
        "Collision Rate", (tot_item - tot_indice) / tot_item
    )  # 打印最终碰撞率

    all_indices_dict = {}  # 字典，用于存储最终的索引映射
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)  # 将索引转换为列表并存储到字典中

    os.makedirs(args.output_dir, exist_ok=True)  # 确保输出目录存在
    output_file_path = os.path.join(
        args.output_dir, args.output_file
    )  # 完整的输出文件路径
    with open(output_file_path, "w") as fp:
        json.dump(all_indices_dict, fp)  # 将索引字典保存为 JSON 文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate indices for Multimodal Recommendation Model"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Output directory"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output JSON file name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0 or cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for data loading"
    )

    args = parser.parse_args()
    main(args)
