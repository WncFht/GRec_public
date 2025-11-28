import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .prompt import all_prompt, sft_prompt
from .type import TrainingSample


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
    def __init__(
        self, args: argparse.Namespace, dataset, logger=None, local_rank=0
    ):
        super().__init__()

        self.args = args
        self.logger = logger
        self.local_rank = local_rank
        self.log_func = logger.info if logger else print

        self.dataset = dataset
        self.data_path = os.path.join(self.args.data_path, self.dataset)

        # 设置历史记录的最大长度、历史记录分隔符和索引文件
        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        # 是否在历史记录中添加前缀（例如：1. itemA, 2. itemB）
        self.add_prefix = args.add_prefix

        # 初始化与新token和允许token相关的变量
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def set_prompt(self, prompt_id):
        # 设置当前使用的prompt ID
        self.prompt_id = prompt_id

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

    def build_hash_dict(
        self,
        tokenizer,
        prefix_index: int = 3,
        response_prefix: str = "### Response:\n",
    ) -> dict[str, list[int]]:
        """
        基于当前数据集中所有 item 构建用于前缀约束的 hash_dict。

        逻辑与 MiniOneRec 中基于 `### Response:\n{item}\n` 的实现保持一致：
        - 先构造形如 "### Response:\n{item}\n" 的片段并分词；
        - 以 `prefix_index` 控制前缀长度，枚举所有前缀 -> 下一 token 的映射；
        - 返回 {hash_key(str): [allowed_token_ids]}。
        """
        hash_dict: dict[str, set[int]] = defaultdict(set)

        eos_id = getattr(tokenizer, "eos_token_id", None)

        all_items = self.get_all_items()

        for item in all_items:
            text = f"{response_prefix}{item}\n"
            tokenized = tokenizer(text)
            ids = list(tokenized["input_ids"])

            # 与原版逻辑一致，强制在末尾补一个 eos
            if eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)

            if len(ids) <= prefix_index:
                continue

            for i in range(prefix_index, len(ids)):
                if i == prefix_index:
                    key_ids = ids[:i]
                else:
                    key_ids = ids[prefix_index:i]

                hash_number = "-".join(str(t) for t in key_ids)
                hash_dict[hash_number].add(ids[i])

        return {k: sorted(list(v)) for k, v in hash_dict.items()}

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
        dataset: str,
        mode="train",  # 数据集模式：训练、验证、测试
        prompt_sample_num=1,  # 每个数据点采样prompt的数量
        prompt_id=0,  # 使用的prompt ID
        sample_num=-1,  # 采样数据点的数量，-1表示不采样
        logger=None,
        local_rank=0,
    ):
        super().__init__(args, dataset, logger, local_rank)

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
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            if self.sample_num > 0:
                self.inter_data = self.inter_data[:sample_num]
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
        if self.local_rank == 0:
            self.log_func(f"original total inters: {total_inters}")
        ratio = self.args.ratio_dataset
        target_size = int(ratio * total_inters)
        sorted_items = sorted(self.inters.items(), key=lambda x: int(x[0]))
        self.inters = dict(sorted_items[:target_size])
        if self.local_rank == 0:
            self.log_func(f"new total inters: {len(self.inters)}")

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

    def _process_train_data(self):
        # 处理训练数据：构建历史交互和目标物品对
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][
                :-2
            ]  # 移除最后两个物品（验证和测试）
            for i in range(1, len(items)):
                one_data = {}
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
            one_data = {}
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
            one_data = {}
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
        if self.mode in ["train", "valid"]:
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        # 计算实际数据索引
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        # 训练模式下随机选择prompt，测试模式下使用指定prompt
        if self.mode in ["train", "valid"]:
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

    def to_verl_records(self, split: str) -> list[dict[str, Any]]:
        """
        Convert the processed dataset into Verl-style records (like gsm8k).

        Each record will have fields:
          - data_source: a short name for the task, use 'seqrec'
          - prompt: a list with a single user role dict containing the input text
          - ability: 'rec' (recommendation)
          - reward_model: a dict with style 'rule' and ground_truth the target item id (string)
          - extra_info: contains split, index, item, inters (history)

        Args:
            split: 'train' or 'valid'
        Returns:
            List of dict records

        """
        records: list[dict[str, Any]] = []

        if split not in ("train", "valid"):
            msg = "split must be 'train' or 'valid'"
            raise ValueError(msg)

        # Ensure data prepared
        if split in ["train", "valid"]:
            data_list = self.inter_data
        else:
            raise NotImplementedError(f"Unsupported split:{split}")

        for idx, d in enumerate(data_list):
            target = d.get("item")
            prompt_id = random.randint(0, len(self.prompts) - 1)
            prompt = self.prompts[prompt_id]
            input_text, label_text = self._get_text_data(d, prompt)

            rec = {
                "data_source": "seqrec",
                "prompt": [{"role": "user", "content": input_text}],
                "ability": "rec",
                "reward_model": {"style": "rule", "ground_truth": label_text},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "item": target,
                    "inters": d.get("inters"),
                },
            }
            records.append(rec)

        return records


# 融合序列推荐数据集类，继承自BaseDataset
class FusionSeqRecDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        dataset: str,
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
        logger=None,
        local_rank=0,
    ):
        super().__init__(args, dataset, logger, local_rank)

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
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
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
                one_data = {}
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
            one_data = {}
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
            one_data = {}
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
        if self.mode in ["train", "valid"]:
            return len(self.inter_data) * self.prompt_sample_num
        if self.mode == "test":
            return len(self.inter_data)
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        label_text = response

        return input_text, label_text

    def to_verl_records(self, split: str) -> list[dict[str, Any]]:
        """
        Convert FusionSeqRecDataset into Verl-style records.

        Similar shape as SeqRecDataset.to_verl_records but uses task name
        'fusionseqrec' as data_source and includes title/description when available.
        """
        records: list[dict[str, Any]] = []

        if split not in ("train", "valid"):
            msg = "split must be 'train' or 'valid'"
            raise ValueError(msg)

        data_list = self.inter_data

        for idx, d in enumerate(data_list):
            target = d.get("item")
            prompt_id = random.randint(0, len(self.prompts) - 1)
            prompt = self.prompts[prompt_id]

            input_text, label_text = self._get_text_data(d, prompt)

            rec = {
                "data_source": "fusionseqrec",
                "prompt": [{"role": "user", "content": input_text}],
                "ability": "rec",
                "reward_model": {"style": "rule", "ground_truth": label_text},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "item": target,
                    "inters": d.get("inters"),
                },
            }
            # include title/description if present
            if "title" in d:
                rec["extra_info"]["title"] = d.get("title")
            if "description" in d:
                rec["extra_info"]["description"] = d.get("description")

            records.append(rec)

        return records

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode in ["train", "valid"]:
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
        dataset: str,
        task="item2index",
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
        logger=None,
        local_rank=0,
    ):
        super().__init__(args, dataset, logger, local_rank)

        self.task = task.lower()
        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.args = args

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
        # 根据 8:1:1 规则获取当前模式下的物品ID列表
        all_item_ids = list(self.indices.keys())
        split_map = _split_item_ids(all_item_ids, self.args.seed)
        item_ids_for_mode = split_map[self.mode]

        if self.local_rank == 0:
            self.log_func(
                f"ItemFeatDataset {self.task} {self.dataset} {self.mode}: processing {len(item_ids_for_mode)} items"
            )

        # 处理物品特征数据
        feat_data = []
        for iid in item_ids_for_mode:
            if iid not in self.item_feat:
                continue

            feat = self.item_feat[iid].copy()
            index = "".join(self.indices[iid])  # 物品ID对应的token字符串
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")  # 清理标题
            feat_data.append(feat)

        # 如果指定了采样数量，则进行采样
        if self.sample_num > 0 and len(feat_data) > self.sample_num:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(
                all_idx, self.sample_num, replace=False
            )
            feat_data = np.array(feat_data)[sample_idx].tolist()

        if self.local_rank == 0:
            self.log_func(
                f"ItemFeatDataset {self.task} {self.mode}: final dataset size {len(feat_data)}"
            )

        return feat_data

    def to_verl_records(self, split: str) -> list[dict[str, Any]]:
        """
        Convert ItemFeatDataset into Verl-style records.

        Uses `self.task` as data_source (e.g., 'item2index').
        """
        records: list[dict[str, Any]] = []

        if split not in ("train", "valid"):
            msg = "split must be 'train' or 'valid'"
            raise ValueError(msg)

        data_list = self.feat_data

        for idx, d in enumerate(data_list):
            target = d.get("item")
            prompt_id = random.randint(0, len(self.prompts) - 1)
            prompt = self.prompts[prompt_id]

            input_text, label_text = self._get_text_data(d, prompt)

            rec = {
                "data_source": self.task,
                "prompt": [{"role": "user", "content": input_text}],
                "ability": "item",
                "reward_model": {"style": "rule", "ground_truth": label_text},
                "extra_info": {"split": split, "index": idx, "item": target},
            }
            # include available features
            for k in ("title", "description"):
                if k in d:
                    rec["extra_info"][k] = d.get(k)

            records.append(rec)

        return records

    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        # 根据prompt和数据构造输入和输出文本
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input_text = sft_prompt.format(instruction=instruction, response="")
        label_text = response

        return input_text, label_text

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        # 根据模式选择prompt
        if self.mode in ["train", "valid"]:
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


def dataset_to_text_samples(
    dataset_obj, mode: str = "train"
) -> list[dict[str, str]]:
    """
    Convert any BaseDataset (or its subclasses) instance into a list of text samples
    usable for RL training loops. Each sample is a dict with keys `prompt` and
    `completion` (strings). This avoids writing parquet files and provides the
    samples directly in-memory.

    Args:
        dataset_obj: an instantiated dataset (subclass of BaseDataset)
        mode: which split/mode to extract (some datasets use mode when constructed)

    Returns:
        List of dicts: [{"prompt": <str>, "completion": <str>, "extra_info": <dict>}]

    """
    samples: list[dict[str, str]] = []

    # If dataset provides to_verl_records, use it to get consistent fields
    if hasattr(dataset_obj, "to_verl_records"):
        # prefer 'train' or 'valid' where applicable
        split = "train" if mode == "train" else "valid"
        try:
            records = dataset_obj.to_verl_records(split)
        except Exception:
            records = []

        for rec in records:
            # rec['prompt'] is a list with a single role dict in this codebase
            prompt_field = rec.get("prompt")
            if isinstance(prompt_field, list) and len(prompt_field) > 0:
                prompt = prompt_field[0].get("content", "")
            else:
                prompt = prompt_field if isinstance(prompt_field, str) else ""
            completion = rec.get("reward_model", {}).get("ground_truth", "")
            samples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "extra_info": rec.get("extra_info", {}),
                }
            )

        return samples

    # Fallback: iterate dataset and call __getitem__ to get TrainingSample objects
    # Some dataset classes return TrainingSample from __getitem__
    try:
        length = len(dataset_obj)
    except Exception:
        length = None

    if length is None:
        # try to iterate until exhaustion
        idx = 0
        while True:
            try:
                ts = dataset_obj[idx]
            except Exception:
                break
            if isinstance(ts, TrainingSample):
                samples.append(
                    {
                        "prompt": ts.input_text,
                        "completion": ts.label_text,
                        "extra_info": {},
                    }
                )
            elif isinstance(ts, dict) and "prompt" in ts and "completion" in ts:
                samples.append(
                    {
                        "prompt": ts["prompt"],
                        "completion": ts["completion"],
                        "extra_info": ts.get("extra_info", {}),
                    }
                )
            idx += 1
        return samples

    for i in range(length):
        try:
            ts = dataset_obj[i]
        except Exception:
            continue
        if isinstance(ts, TrainingSample):
            samples.append(
                {
                    "prompt": ts.input_text,
                    "completion": ts.label_text,
                    "extra_info": {},
                }
            )
        elif isinstance(ts, dict) and "prompt" in ts and "completion" in ts:
            samples.append(
                {
                    "prompt": ts["prompt"],
                    "completion": ts["completion"],
                    "extra_info": ts.get("extra_info", {}),
                }
            )

    return samples


def samples_to_hf_dataset(samples: list[dict[str, str]]):
    """
    Convert the list of prompt/completion dicts into a HuggingFace `datasets.Dataset`.
    This helper avoids parquet and returns an in-memory dataset ready for trainers.
    """
    try:
        from datasets import Dataset as HFDataset
    except Exception:
        raise RuntimeError(
            "Please install the `datasets` package to convert samples to HF Dataset"
        )

    if not samples:
        return HFDataset.from_dict({"prompt": [], "completion": []})

    prompts = [s.get("prompt", "") for s in samples]
    completions = [s.get("completion", "") for s in samples]
    extras = [s.get("extra_info", {}) for s in samples]

    return HFDataset.from_dict(
        {"prompt": prompts, "completion": completions, "extra_info": extras}
    )
