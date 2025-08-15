import argparse
import copy
import json
import os
import random
from dataclasses import asdict

import numpy as np
from torch.utils.data import Dataset

from src.parser import parse_dataset_args

from .prompt import all_prompt, sft_prompt
from .type import EnrichedData, TrainingSample


def _split_item_ids(all_item_ids: list[str], seed: int) -> dict[str, list[str]]:
    """
    将所有物品ID按8:1:1的比例确定性地划分为训练集、验证集和测试集。

    Args:
    ----
        all_item_ids (list[str]): 所有物品ID的列表。
        seed (int): 用于复现的随机种子。

    Returns:
    -------
        dict[str, list[str]]: 一个字典，包含"train", "valid", "test"三个键，
                             对应的值是各自的物品ID列表。

    """
    rng = np.random.default_rng(seed)
    shuffled_ids = list(all_item_ids)
    rng.shuffle(shuffled_ids)

    n_items = len(shuffled_ids)
    n_train = int(0.8 * n_items)
    n_valid = int(0.1 * n_items)

    train_ids = shuffled_ids[:n_train]
    valid_ids = shuffled_ids[n_train : n_train + n_valid]
    test_ids = shuffled_ids[n_train + n_valid :]

    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


# 定义BaseDataset类，继承自PyTorch的Dataset
class BaseDataset(Dataset):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        dataset_args = args.dataset_args

        self.dataset = dataset_args.dataset
        self.data_path = os.path.join(dataset_args.data_path, self.dataset)

        # 设置历史记录的最大长度、历史记录分隔符和索引文件
        self.max_his_len = dataset_args.max_his_len
        self.his_sep = dataset_args.his_sep
        self.index_file = dataset_args.index_file
        # 是否在历史记录中添加前缀（例如：1. itemA, 2. itemB）
        self.add_prefix = dataset_args.add_prefix

        # 初始化与新token和允许token相关的变量
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        # 从指定的索引文件中加载物品索引
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        # 如果new_tokens已经计算过，则直接返回
        if self.new_tokens is not None:
            return self.new_tokens

        # 初始化一个集合来存储新的tokens
        self.new_tokens = set()
        # 遍历所有物品的索引，提取每个token并添加到集合中
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        # 将集合转换为排序后的列表
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):
        # 如果all_items已经计算过，则直接返回
        if self.all_items is not None:
            return self.all_items

        # 初始化一个集合来存储所有物品（合并后的token字符串）
        self.all_items = set()
        # 遍历所有物品的索引，将每个索引列表连接成字符串并添加到集合中
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    # 获取前缀允许的token的函数，用于控制模型生成
    def get_prefix_allowed_tokens_fn(self, tokenizer):
        """
        使用Trie结构，严格约束生成序列只能为候选集合的前缀。

        Args:
        ----
            tokenizer: 用于将token转换为token ID的tokenizer对象
        Returns:
            prefix_allowed_tokens_fn: 前缀允许的token的函数

        """
        # 延迟导入，避免循环依赖
        from .generation_trie import Trie, prefix_allowed_tokens_fn

        # 构造候选序列，每个物品的token id序列
        # 注意：这里假设self.indices的value为token字符串列表，需要encode为id
        all_items = set()
        for index in self.indices.values():
            # index是token字符串列表，需拼接后encode
            # 例如: ["A", "B"] -> "AB" -> tokenizer.encode("AB")
            token_str = "".join(index)
            all_items.add(token_str)
        # print(all_items)
        candidate_trie = Trie(
            [[0] + tokenizer.encode(candidate) for candidate in all_items]
        )
        # print(candidate_trie.trie_dict)
        return prefix_allowed_tokens_fn(candidate_trie)

    def _process_data(self):
        # 抽象方法，子类必须实现
        raise NotImplementedError


# 序列推荐数据集类，继承自BaseDataset
class SeqRecDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",  # 数据集模式：训练、验证、测试
        prompt_sample_num=1,  # 每个数据点采样prompt的数量
        prompt_id=0,  # 使用的prompt ID
        sample_num=-1,  # 采样数据点的数量，-1表示不采样
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.args = args
        # 加载序列推荐任务的prompt
        self.prompts = all_prompt["seqrec"]

        # 加载原始数据和物品映射
        self._load_data()  # 加载self.inters (用户交互序列) 和 self.indices (物品token索引)
        self._remap_items()  # 将交互序列中的物品ID映射为token形式

        # 根据模式处理数据
        if self.mode == "train":
            self.inter_data = self._process_train_data()
            if self.sample_num > 0:
                self.inter_data = self.inter_data[:sample_num]
        elif self.mode == "valid":
            self.sample_valid = args.dataset_args.sample_valid
            self.valid_prompt_id = args.dataset_args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()  # 构建验证集文本数据
            if self.sample_num > 0:
                self.valid_text_data = self.valid_text_data[:sample_num]
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _load_data(self):
        # 加载用户交互数据
        with open(
            os.path.join(self.data_path, self.dataset + ".inter.json")
        ) as f:
            self.inters = json.load(f)

        total_inters = len(self.inters)
        print("original total inters:", total_inters)
        ratio = self.args.dataset_args.ratio_dataset
        target_size = int(ratio * total_inters)
        sorted_items = sorted(self.inters.items(), key=lambda x: int(x[0]))
        self.inters = dict(sorted_items[:target_size])
        print("new total inters:", len(self.inters))

        # 加载物品索引数据
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)

    def _remap_items(self):
        # 将用户交互序列中的物品ID映射为对应的token字符串
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self):
        # 处理训练数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][
                :-2
            ]  # 移除最后两个物品（验证和测试）
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]  # 当前目标物品
                history = items[:i]  # 历史交互物品
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]  # 截断历史记录
                if self.add_prefix:
                    history = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(history)
                    ]  # 添加前缀
                one_data["inters"] = self.his_sep.join(
                    history
                )  # 用分隔符连接历史记录
                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        # 处理验证数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]  # 验证集的目标物品是倒数第二个
            history = items[:-2]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        # 处理测试数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]  # 测试集的目标物品是最后一个
            history = items[:-1]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def set_prompt(self, prompt_id):
        # 设置当前使用的prompt ID
        self.prompt_id = prompt_id

    def __len__(self):
        # 返回数据集的长度
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "valid":
            return len(self.valid_text_data)
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _construct_valid_text(self):
        # 构建验证集文本数据
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, label_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        TrainingSample(
                            input_text=input_text,
                            label_text=label_text,
                            is_multimodal=False,
                        )
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, label_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    TrainingSample(
                        input_text=input_text,
                        label_text=label_text,
                        is_multimodal=False,
                    )
                )

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        # 构建输入文本（包含instruction，response为空）
        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        # 根据索引获取数据
        if self.mode == "valid":
            return self.valid_text_data[index]

        # 计算实际数据索引
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        # 训练模式下随机选择prompt，测试模式下使用指定prompt
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        prompt = self.prompts[prompt_id]
        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


# 融合序列推荐数据集类，继承自BaseDataset
class FusionSeqRecDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        # 加载数据
        self._load_data()
        # self._remap_items() # 在FusionSeqRecDataset中，物品ID直接用于查找特征，不需要remap

        # 根据模式处理数据
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.sample_valid = args.dataset_args.sample_valid
            self.valid_prompt_id = args.dataset_args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _load_data(self):
        # 加载用户交互数据
        with open(
            os.path.join(self.data_path, self.dataset + ".inter.json")
        ) as f:
            self.inters = json.load(f)
        # 加载物品索引数据
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)
        # 加载物品特征数据
        with open(
            os.path.join(self.data_path, self.dataset + ".item.json")
        ) as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):
        # 处理训练数据，包含物品的文本特征
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = "".join(
                    self.indices[str(items[i])]
                )  # 目标物品的token形式
                one_data["title"] = (
                    self.item_feat[str(items[i])]["title"]
                    .strip()
                    .strip(".!?,;:`")
                )  # 目标物品的标题
                one_data["description"] = self.item_feat[str(items[i])][
                    "description"
                ]  # 目标物品的描述
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]
                inters = [
                    "".join(self.indices[str(j)]) for j in history
                ]  # 历史物品的token形式
                inter_titles = [
                    '"'
                    + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`")
                    + '"'
                    for j in history
                ]  # 历史物品的标题

                if self.add_prefix:
                    inters = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(inters)
                    ]
                    inter_titles = [
                        str(k + 1) + ". " + item_title
                        for k, item_title in enumerate(inter_titles)
                    ]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        # 处理验证数据，逻辑与训练数据类似，但目标物品为倒数第二个
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = (
                self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            )
            one_data["description"] = self.item_feat[str(items[-2])][
                "description"
            ]

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = [
                '"'
                + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`")
                + '"'
                for j in history
            ]

            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]
                inter_titles = [
                    str(k + 1) + ". " + item_title
                    for k, item_title in enumerate(inter_titles)
                ]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        # 处理测试数据，逻辑与训练数据类似，但目标物品为最后一个
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = (
                self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            )
            one_data["description"] = self.item_feat[str(items[-1])][
                "description"
            ]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = [
                '"'
                + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`")
                + '"'
                for j in history
            ]

            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]
                inter_titles = [
                    str(k + 1) + ". " + item_title
                    for k, item_title in enumerate(inter_titles)
                ]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "valid":
            return len(self.valid_text_data)
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _construct_valid_text(self):
        # 构建验证集文本数据
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, label_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        TrainingSample(
                            input_text=input_text,
                            label_text=label_text,
                            is_multimodal=False,
                        )
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, label_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    TrainingSample(
                        input_text=input_text,
                        label_text=label_text,
                        is_multimodal=False,
                    )
                )

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        # 构建输入和输出文本，这里output包含了instruction和response
        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本即为response
        label_text = response

        # 测试模式下，只返回instruction作为输入，response作为目标
        if self.mode == "test":
            return input_text, response

        return input_text, label_text

    def __getitem__(self, index):
        if self.mode == "valid":
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        prompt = self.prompts[prompt_id]

        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


# 物品特征数据集类，继承自BaseDataset
class ItemFeatDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        task="item2index",
        prompt_sample_num=1,
        sample_num=-1,
    ):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # 加载数据并处理
        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        # 加载物品索引文件
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)
        # 加载物品特征文件
        with open(
            os.path.join(self.data_path, self.dataset + ".item.json")
        ) as f:
            self.item_feat = json.load(f)

    def _process_data(self):
        # 处理物品特征数据
        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])  # 物品ID对应的token字符串
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")  # 清理标题
            feat_data.append(feat)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(
                all_idx, self.sample_num, replace=False
            )

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data

    def __len__(self):
        # 返回数据集长度
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        # 在SFT中，标签通常是完整的 "instruction + response"
        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        # 根据索引获取数据
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        # 随机选择prompt
        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


# 物品搜索数据集类，继承自BaseDataset
class ItemSearchDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        # 加载数据并处理
        self._load_data()
        self.search_data = self._process_data()

    def _load_data(self):
        # 加载物品索引文件
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)
        # 加载用户数据文件
        with open(
            os.path.join(self.data_path, self.dataset + ".user.json")
        ) as f:
            self.user_info = json.load(f)

    def _process_data(self):
        # 处理搜索数据
        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        # 根据模式选择模糊意图数据
        if self.mode == "train":
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == "test":
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        for uid in user_explicit_preference:
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep  # 用户明确偏好
            one_data["user_related_intention"] = user_vi[0]  # 用户相关意图
            one_data["item_related_intention"] = user_vi[1]  # 物品相关意图

            iid = user_vague_intention[uid]["item"]  # 目标物品ID
            inters = user_vague_intention[uid]["inters"]  # 交互历史

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len :]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]

            one_data["inters"] = self.his_sep.join(inters)

            search_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(
                all_idx, self.sample_num, replace=False
            )

            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.search_data) * self.prompt_sample_num
        if self.mode == "test":
            return len(self.search_data)
        return len(self.search_data)

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")

        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
            item_id=d["item"],  # 传递item id用于评估
        )


# 偏好获取数据集类，继承自BaseDataset
class PreferenceObtainDataset(BaseDataset):
    def __init__(
        self, args: argparse.Namespace, prompt_sample_num=1, sample_num=-1
    ):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # 加载数据
        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()

    def _load_data(self):
        # 加载用户数据文件
        with open(
            os.path.join(self.data_path, self.dataset + ".user.json")
        ) as f:
            self.user_info = json.load(f)
        # 加载用户交互数据文件
        with open(
            os.path.join(self.data_path, self.dataset + ".inter.json")
        ) as f:
            self.inters = json.load(f)
        # 加载物品索引数据
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file)
        ) as f:
            self.indices = json.load(f)

    def _remap_items(self):
        # 将用户交互序列中的物品ID映射为对应的token字符串
        self.remapped_inters = {}
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):
        # 处理偏好获取数据
        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference:
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len :]
            if self.add_prefix:
                inters = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(inters)
                ]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(
                all_idx, self.sample_num, replace=False
            )

            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        # 返回数据集长度
        return len(self.preference_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        # 根据索引获取数据
        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        # 随机选择prompt
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(
            random.choice(d["explicit_preferences"])
        )

        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


# 多模态数据集类，继承自BaseDataset
class MultimodalDataset(BaseDataset):
    """多模态数据集：支持mmitem2index和mmindex2item任务"""

    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",
        task="mmitem2index",
        prompt_sample_num=1,
        sample_num=-1,
    ):
        super().__init__(args)

        self.mode = mode
        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.image_path = os.path.join(
            self.data_path, args.dataset_args.image_path
        )
        self.item_meta_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item.json"
        )
        self.item2id_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item2id"
        )

        # 根据任务类型选择对应的prompts
        if self.task == "mmitem2index":
            self.prompts = all_prompt["mmitem2index"]
        elif self.task == "mmindex2item":
            self.prompts = all_prompt["mmindex2item"]
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        # 加载数据
        self._load_data()  # 加载物品的index file
        self._load_item_metadata()  # 加载物品的元数据
        self._load_item2id()
        self._process_data()

    def _load_item_metadata(self):
        """加载物品元数据（标题、描述等）"""
        with open(self.item_meta_path, encoding="utf-8") as f:
            self.item_metadata = json.load(f)

    def _load_item2id(self):
        self.item2id = {}
        self.id2item = {}

        with open(self.item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                self.item2id[item_id] = num_id
                self.id2item[num_id] = item_id

    def _process_data(self):
        """处理多模态数据"""
        self.multimodal_data = []

        for item_id, token_list in self.indices.items():
            have_image = True
            if item_id not in self.item_metadata:
                continue

            metadata = self.item_metadata[item_id]
            image_file = f"{self.id2item[item_id]}.jpg"  # 假设图片文件命名格式
            image_path = os.path.join(self.image_path, image_file)

            # if not os.path.exists(image_path):
            #     have_image = False

            one_data = {
                "item_id": item_id,
                "item": "".join(token_list),
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "brand": metadata.get("brand", ""),
                "categories": metadata.get("categories", ""),
                "enhanced_title": metadata.get("enhanced_title", ""),
                "tags": ", ".join(metadata.get("tags", []))
                if isinstance(metadata.get("tags"), list)
                else metadata.get("tags", ""),
                "highlights": ", ".join(metadata.get("highlights", []))
                if isinstance(metadata.get("highlights"), list)
                else metadata.get("highlights", ""),
                "characteristics": ", ".join(
                    metadata.get("characteristics", [])
                )
                if isinstance(metadata.get("characteristics"), list)
                else metadata.get("characteristics", ""),
                "image_path": image_path,
                # 'have_image': have_image
            }
            self.multimodal_data.append(one_data)
        print("len(self.multimodal_data):", len(self.multimodal_data))

        # 数据采样
        if self.sample_num > 0 and len(self.multimodal_data) > self.sample_num:
            sampled_indices = np.random.choice(
                len(self.multimodal_data), self.sample_num, replace=False
            )
            self.multimodal_data = [
                self.multimodal_data[i] for i in sampled_indices
            ]

    def _get_text_data(self, data: dict, prompt: dict) -> tuple[str, str, str]:
        """构造文本数据，并返回图片路径"""
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        # 使用原有的sft_prompt格式
        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text, data["image_path"]

    def __len__(self):
        # 返回数据集长度
        return len(self.multimodal_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        data = self.multimodal_data[idx]

        # 随机选择prompt
        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        input_text, label_text, image_path = self._get_text_data(data, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=True,
            image_path=image_path,
            item_id=data["item_id"],
        )


# 文本丰富任务数据集类，继承自BaseDataset
class TextEnrichDataset(BaseDataset):
    """文本丰富任务数据集"""

    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",
        prompt_sample_num=1,
        sample_num=-1,
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.image_path = os.path.join(
            self.data_path, args.dataset_args.image_path
        )
        self.item_meta_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item_enriched_v2.json"
        )
        self.item2id_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item2id"
        )

        # TextEnrich任务的prompts
        self.prompts = all_prompt["text_enrich"]

        # 加载数据
        self._load_data()
        self._load_item_metadata()
        self._load_item2id()
        self._process_data()

    def _load_item_metadata(self):
        """加载物品元数据"""
        with open(self.item_meta_path, encoding="utf-8") as f:
            self.item_metadata = json.load(f)
        print("len(self.item_metadata):", len(self.item_metadata))

    def _load_item2id(self):
        self.item2id = {}
        self.id2item = {}

        with open(self.item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                self.item2id[item_id] = num_id
                self.id2item[num_id] = item_id
        print("len(self.item2id):", len(self.item2id))
        print("len(self.id2item):", len(self.id2item))

    def _process_data(self):
        """处理文本丰富数据"""
        self.textenrich_data = []

        # 根据 8:1:1 规则获取当前模式下的物品ID列表
        all_item_ids = list(self.indices.keys())
        split_map = _split_item_ids(all_item_ids, self.args.global_args.seed)
        item_ids_for_mode = split_map[self.mode]

        for item_id in item_ids_for_mode:
            if item_id not in self.item_metadata:
                print(f"item_id: {item_id} 没有在item_metadata中")
                continue

            token_list = self.indices[item_id]
            metadata = self.item_metadata[item_id]

            # 检查是否有丰富后的文本信息
            if not all(
                key in metadata
                for key in [
                    "enhanced_title",
                    "tags",
                    "highlights",
                    "characteristics",
                ]
            ):
                # print the lost keys
                for key in [
                    "enhanced_title",
                    "tags",
                    "highlights",
                    "characteristics",
                ]:
                    if key not in metadata:
                        print(f"item_id: {item_id} 的metadata中缺少 {key}")
                continue

            image_file = f"{self.id2item[item_id]}.jpg"
            image_path = os.path.join(self.image_path, image_file)

            # if not os.path.exists(image_path):
            #     continue

            # 构建训练数据，使用 EnrichedData dataclass
            enriched_data = EnrichedData(
                item_id=item_id,
                item="".join(token_list),
                title=metadata.get("title", ""),
                description=metadata.get("description", ""),
                brand=metadata.get("brand", ""),
                categories=metadata.get("categories", ""),
                enhanced_title=metadata.get("enhanced_title", ""),
                tags=", ".join(metadata.get("tags", []))
                if isinstance(metadata.get("tags"), list)
                else metadata.get("tags", ""),
                highlights=", ".join(metadata.get("highlights", []))
                if isinstance(metadata.get("highlights"), list)
                else metadata.get("highlights", ""),
                characteristics=", ".join(metadata.get("characteristics", []))
                if isinstance(metadata.get("characteristics"), list)
                else metadata.get("characteristics", ""),
                image_path=image_path,
            )
            self.textenrich_data.append(enriched_data)
        print("len(self.textenrich_data):", len(self.textenrich_data))

        # 数据采样, 调试的时候调用
        if self.sample_num > 0 and len(self.textenrich_data) > self.sample_num:
            sampled_indices = np.random.choice(
                len(self.textenrich_data), self.sample_num, replace=False
            )
            self.textenrich_data = [
                self.textenrich_data[i] for i in sampled_indices
            ]

    def _get_text_data(
        self, data: EnrichedData, prompt: dict
    ) -> tuple[str, str, str]:
        """构造文本数据"""
        # data现在是一个EnrichedData对象，但prompt.format期望一个字典
        # dataclasses.asdict() 可以很好地处理这个问题
        instruction = prompt["instruction"].format(**asdict(data))
        response = prompt["response"].format(**asdict(data))

        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text, data.image_path

    def __len__(self):
        # 返回数据集长度
        return len(self.textenrich_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        data = self.textenrich_data[idx]

        # 随机选择prompt
        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        input_text, label_text, image_path = self._get_text_data(data, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=True,
            image_path=image_path,
            item_id=data.item_id,
        )


# 文本丰富任务数据集类，继承自BaseDataset
class TextEnrichWihtoutItemIDDataset(BaseDataset):
    """文本丰富任务数据集"""

    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",
        prompt_sample_num=1,
        sample_num=-1,
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.image_path = os.path.join(
            self.data_path, args.dataset_args.image_path
        )
        self.item_meta_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item_enriched_v2.json"
        )
        self.item2id_path = os.path.join(
            self.data_path, f"{args.dataset_args.dataset}.item2id"
        )

        # TextEnrich任务的prompts
        self.prompts = all_prompt["text_enrich_without_id"]

        # 加载数据
        self._load_data()
        self._load_item_metadata()
        self._load_item2id()
        self._process_data()

    def _load_item_metadata(self):
        """加载物品元数据"""
        with open(self.item_meta_path, encoding="utf-8") as f:
            self.item_metadata = json.load(f)
        print("len(self.item_metadata):", len(self.item_metadata))

    def _load_item2id(self):
        self.item2id = {}
        self.id2item = {}

        with open(self.item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                self.item2id[item_id] = num_id
                self.id2item[num_id] = item_id
        print("len(self.item2id):", len(self.item2id))
        print("len(self.id2item):", len(self.id2item))

    def _process_data(self):
        """处理文本丰富数据"""
        self.textenrich_data = []

        # 根据 8:1:1 规则获取当前模式下的物品ID列表
        all_item_ids = list(self.indices.keys())
        # 数据采样, 调试的时候调用
        if self.sample_num > 0 and len(all_item_ids) > self.sample_num:
            # sampled_indices = np.random.choice(
            #     len(all_item_ids), self.sample_num, replace=False
            # )
            sampled_indices = range(self.sample_num)
            all_item_ids = [all_item_ids[i] for i in sampled_indices]
        split_map = _split_item_ids(all_item_ids, self.args.global_args.seed)
        item_ids_for_mode = split_map[self.mode]

        for item_id in item_ids_for_mode:
            if item_id not in self.item_metadata:
                print(f"item_id: {item_id} 没有在item_metadata中")
                continue

            token_list = self.indices[item_id]
            metadata = self.item_metadata[item_id]

            # 检查是否有丰富后的文本信息
            if not all(
                key in metadata
                for key in [
                    "enhanced_title",
                    "tags",
                    "highlights",
                    "characteristics",
                ]
            ):
                # print the lost keys
                for key in [
                    "enhanced_title",
                    "tags",
                    "highlights",
                    "characteristics",
                ]:
                    if key not in metadata:
                        print(f"item_id: {item_id} 的metadata中缺少 {key}")
                continue

            image_file = f"{self.id2item[item_id]}.jpg"
            image_path = os.path.join(self.image_path, image_file)

            # if not os.path.exists(image_path):
            #     continue

            # 构建训练数据，使用 EnrichedData dataclass
            enriched_data = EnrichedData(
                item_id=item_id,
                item="".join(token_list),
                title=metadata.get("title", ""),
                description=metadata.get("description", ""),
                brand=metadata.get("brand", ""),
                categories=metadata.get("categories", ""),
                enhanced_title=metadata.get("enhanced_title", ""),
                tags=", ".join(metadata.get("tags", []))
                if isinstance(metadata.get("tags"), list)
                else metadata.get("tags", ""),
                highlights=", ".join(metadata.get("highlights", []))
                if isinstance(metadata.get("highlights"), list)
                else metadata.get("highlights", ""),
                characteristics=", ".join(metadata.get("characteristics", []))
                if isinstance(metadata.get("characteristics"), list)
                else metadata.get("characteristics", ""),
                image_path=image_path,
            )
            self.textenrich_data.append(enriched_data)
        print("len(self.textenrich_data):", len(self.textenrich_data))

    def _get_text_data(
        self, data: EnrichedData, prompt: dict
    ) -> tuple[str, str, str]:
        """构造文本数据"""
        # data现在是一个EnrichedData对象，但prompt.format期望一个字典
        # dataclasses.asdict() 可以很好地处理这个问题
        instruction = prompt["instruction"].format(**asdict(data))
        response = prompt["response"].format(**asdict(data))

        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text, data.image_path

    def __len__(self):
        # 返回数据集长度
        return len(self.textenrich_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        data = self.textenrich_data[idx]

        # 随机选择prompt
        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        input_text, label_text, image_path = self._get_text_data(data, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=True,
            image_path=image_path,
            item_id=data.item_id,
        )


# 序列推荐数据集类，继承自BaseDataset
class SeqRectWithoutItemIDDataset_1(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",  # 数据集模式：训练、验证、测试
        prompt_sample_num=1,  # 每个数据点采样prompt的数量
        prompt_id=0,  # 使用的prompt ID
        sample_num=-1,  # 采样数据点的数量，-1表示不采样
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.args = args
        # 加载序列推荐任务的prompt
        self.prompts = all_prompt["seqrec"]

        # 加载原始数据和物品映射
        self._load_data()  # 加载self.inters (用户交互序列) 和 self.indices (物品token索引)
        self._remap_items()  # 将交互序列中的物品ID映射为token形式

        # 根据模式处理数据
        if self.mode == "train":
            self.inter_data = self._process_train_data()
            if self.sample_num > 0:
                self.inter_data = self.inter_data[:sample_num]
        elif self.mode == "valid":
            self.sample_valid = args.dataset_args.sample_valid
            self.valid_prompt_id = args.dataset_args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()  # 构建验证集文本数据
            if self.sample_num > 0:
                self.valid_text_data = self.valid_text_data[:sample_num]
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _load_data(self):
        # 加载用户交互数据
        with open(
            os.path.join(self.data_path, self.dataset + ".inter.json")
        ) as f:
            self.inters = json.load(f)
        total_inters = len(self.inters)
        print("original total inters:", total_inters)
        ratio = self.args.dataset_args.ratio_dataset
        target_size = int(ratio * total_inters)
        sorted_items = sorted(self.inters.items(), key=lambda x: int(x[0]))
        self.inters = dict(sorted_items[:target_size])
        print("new total inters:", len(self.inters))

    def _remap_items(self):
        # 将用户交互序列中的物品ID映射为对应的token字符串
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = [f"item{i}" for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self):
        # 处理训练数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][
                :-2
            ]  # 移除最后两个物品（验证和测试）
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]  # 当前目标物品
                history = items[:i]  # 历史交互物品
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]  # 截断历史记录
                if self.add_prefix:
                    history = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(history)
                    ]  # 添加前缀
                one_data["inters"] = self.his_sep.join(
                    history
                )  # 用分隔符连接历史记录
                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        # 处理验证数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]  # 验证集的目标物品是倒数第二个
            history = items[:-2]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        # 处理测试数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]  # 测试集的目标物品是最后一个
            history = items[:-1]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def set_prompt(self, prompt_id):
        # 设置当前使用的prompt ID
        self.prompt_id = prompt_id

    def __len__(self):
        # 返回数据集的长度
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "valid":
            return len(self.valid_text_data)
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _construct_valid_text(self):
        # 构建验证集文本数据
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, label_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        TrainingSample(
                            input_text=input_text,
                            label_text=label_text,
                            is_multimodal=False,
                        )
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, label_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    TrainingSample(
                        input_text=input_text,
                        label_text=label_text,
                        is_multimodal=False,
                    )
                )

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text

    def get_all_items(self):
        return list(self.remapped_inters.values())

    def __getitem__(self, index):
        # 根据索引获取数据
        if self.mode == "valid":
            return self.valid_text_data[index]

        # 计算实际数据索引
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        # 训练模式下随机选择prompt，测试模式下使用指定prompt
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        prompt = self.prompts[prompt_id]
        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


class SeqRecWithTitleDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode="train",  # 数据集模式：训练、验证、测试
        prompt_sample_num=1,  # 每个数据点采样prompt的数量
        prompt_id=0,  # 使用的prompt ID
        sample_num=-1,  # 采样数据点的数量，-1表示不采样
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.args = args
        # 加载序列推荐任务的prompt
        self.prompts = all_prompt["seqrec"]

        # 加载原始数据和物品映射
        self._load_data()  # 加载self.inters (用户交互序列) 和 self.indices (物品token索引)
        self._remap_items()  # 将交互序列中的物品ID映射为token形式

        # 根据模式处理数据
        if self.mode == "train":
            self.inter_data = self._process_train_data()
            if self.sample_num > 0:
                self.inter_data = self.inter_data[:sample_num]
        elif self.mode == "valid":
            self.sample_valid = args.dataset_args.sample_valid
            self.valid_prompt_id = args.dataset_args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()  # 构建验证集文本数据
            if self.sample_num > 0:
                self.valid_text_data = self.valid_text_data[:sample_num]
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _load_data(self):
        # 加载用户交互数据
        with open(
            os.path.join(self.data_path, self.dataset + ".inter.json")
        ) as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + ".item.json")
        ) as f:
            self.items = json.load(f)
        total_inters = len(self.inters)
        print("original total inters:", total_inters)
        ratio = self.args.dataset_args.ratio_dataset
        target_size = int(ratio * total_inters)
        sorted_items = sorted(self.inters.items(), key=lambda x: int(x[0]))
        self.inters = dict(sorted_items[:target_size])
        print("new total inters:", len(self.inters))

    def _remap_items(self):
        # 将用户交互序列中的物品ID映射为对应的token字符串
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = [
                f"<title>{self.items[str(i)]['title']}</title>" for i in items
            ]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self):
        # 处理训练数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][
                :-2
            ]  # 移除最后两个物品（验证和测试）
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]  # 当前目标物品
                history = items[:i]  # 历史交互物品
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]  # 截断历史记录
                if self.add_prefix:
                    history = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(history)
                    ]  # 添加前缀
                one_data["inters"] = self.his_sep.join(
                    history
                )  # 用分隔符连接历史记录
                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        # 处理验证数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]  # 验证集的目标物品是倒数第二个
            history = items[:-2]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)
        return inter_data

    def _process_test_data(self):
        # 处理测试数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]  # 测试集的目标物品是最后一个
            history = items[:-1]  # 历史交互物品
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx
                    for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(
                all_inter_idx, self.sample_num, replace=False
            )
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def set_prompt(self, prompt_id):
        # 设置当前使用的prompt ID
        self.prompt_id = prompt_id

    def __len__(self):
        # 返回数据集的长度
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "valid":
            return len(self.valid_text_data)
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _construct_valid_text(self):
        # 构建验证集文本数据
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(
                    all_prompt_ids, self.prompt_sample_num, replace=False
                )
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input_text, label_text = self._get_text_data(d, prompt)
                    self.valid_text_data.append(
                        TrainingSample(
                            input_text=input_text,
                            label_text=label_text,
                            is_multimodal=False,
                        )
                    )
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input_text, label_text = self._get_text_data(d, prompt)
                self.valid_text_data.append(
                    TrainingSample(
                        input_text=input_text,
                        label_text=label_text,
                        is_multimodal=False,
                    )
                )

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        # 标签文本应该是完整的 instruction + response 格式
        label_text = response

        return input_text, label_text

    def get_all_items(self):
        return list(self.remapped_inters.values())

    def __getitem__(self, index):
        # 根据索引获取数据
        if self.mode == "valid":
            return self.valid_text_data[index]

        # 计算实际数据索引
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        # 训练模式下随机选择prompt，测试模式下使用指定prompt
        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == "test":
            prompt_id = self.prompt_id
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        prompt = self.prompts[prompt_id]
        input_text, label_text = self._get_text_data(d, prompt)

        return TrainingSample(
            input_text=input_text,
            label_text=label_text,
            is_multimodal=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    dataset = SeqRecDataset(args, mode="test")
    print(dataset[0])

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.collator import Collator

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    tokenizer = processor.tokenizer
    collator = Collator(args, tokenizer=tokenizer)

    new_tokens = dataset.get_new_tokens()
    tokenizer.add_tokens(new_tokens)

    # collator = MultiModalCollator(args, processor_or_tokenizer=processor)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
    )
    model.eval()
    model.to("cuda")

    inputs = collator([dataset[i] for i in range(4)])
    print(inputs)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    results = model.generate(**inputs)

    print(tokenizer.decode(results[0], skip_special_tokens=True))
