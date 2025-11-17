import argparse
import types
import typing
from pathlib import Path

import yaml

from .type import (
    Args,
    BenchmarkModelArgs,
    DatasetArgs,
    GlobalArgs,
    TestArgs,
    TextGenerationArgs,
    TrainingArgs,
)


def load_config_from_yaml(config_file: str) -> Args:
    """从 YAML 文件加载配置并实例化为 Args 对象"""
    with open(config_file, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    args = Args()
    for section, section_args_class in [
        ("global_args", GlobalArgs),
        ("dataset_args", DatasetArgs),
        ("train_args", TrainingArgs),
        ("test_args", TestArgs),
        ("text_generation_args", TextGenerationArgs),
    ]:
        if section in config_dict:
            section_dict = config_dict[section]

            # 特殊处理嵌套的 dataclass 列表, 例如 test_args.models
            if section == "test_args" and "models" in section_dict:
                models_config = section_dict.pop("models", [])
                section_instance = section_args_class(**section_dict)
                if models_config:
                    section_instance.models = [
                        BenchmarkModelArgs(**m) for m in models_config
                    ]
                setattr(args, section, section_instance)
            else:
                # 使用原始逻辑进行字典解包
                setattr(args, section, section_args_class(**section_dict))

    return args


def parse_args() -> Args:
    """
    解析命令行参数。
    主要用于指定配置文件路径，并允许通过命令行覆盖任意配置项。
    """
    parser = argparse.ArgumentParser(
        description="基于 YAML 和 Dataclass 的统一配置管理"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yml",
        help="配置文件的路径",
    )

    # 临时解析，只为获取config_file
    temp_args, remaining_argv = parser.parse_known_args()

    # 从 YAML 加载基础配置
    if Path(temp_args.config_file).exists():
        args = load_config_from_yaml(temp_args.config_file)
    else:
        print(f"警告：找不到配置文件 {temp_args.config_file}，将使用默认配置。")
        args = Args()

    # --- 允许命令行覆盖 YAML 配置 ---
    # 动态地为 Args 中的所有字段添加命令行参数
    for section_name, section_dc in args.__dataclass_fields__.items():
        section_obj = getattr(args, section_name)
        for field_name, field_type in section_obj.__dataclass_fields__.items():
            full_arg_name = f"--{section_name}.{field_name}"

            arg_type = field_type.type
            # 兼容处理 str | None 这样的联合类型
            origin = typing.get_origin(arg_type)
            if origin is typing.Union or origin is types.UnionType:
                union_args = typing.get_args(arg_type)
                non_none_args = [t for t in union_args if t is not type(None)]
                if len(non_none_args) == 1:
                    arg_type = non_none_args[0]

            parser.add_argument(
                full_arg_name,
                type=arg_type,
                help=f"覆盖 {section_name} 中的 {field_name}",
            )

    # 重新解析所有参数（包括命令行覆盖项）
    final_args = parser.parse_args(remaining_argv)

    # 将命令行覆盖的参数更新到 Args 对象中
    for section_name, section_dc in args.__dataclass_fields__.items():
        section_obj = getattr(args, section_name)
        for field_name in section_obj.__dataclass_fields__:
            arg_val = getattr(final_args, f"{section_name}.{field_name}", None)
            if arg_val is not None:
                setattr(section_obj, field_name, arg_val)

    return args


if __name__ == "__main__":
    # 这是一个示例，展示如何使用
    args = parse_args()

    # 打印加载的配置
    import json

    def dataclass_to_dict(dc):
        if not hasattr(dc, "__dataclass_fields__"):
            return dc
        result = {}
        for f in dc.__dataclass_fields__:
            value = dataclass_to_dict(getattr(dc, f))
            result[f] = value
        return result

    print(json.dumps(dataclass_to_dict(args), indent=2))
