import math
import os
import random
from typing import Dict, List, Union, Optional, Tuple

import torch
import numpy as np
from omegaconf import DictConfig


def setup_rl_output_dir(conf: DictConfig):
    if conf.get("dry_run") == True:
        conf.output_dir = None
        return

    if conf.get("output_dir") is not None:
        os.makedirs(conf.output_dir, exist_ok=True)
        return

    dirs = [conf.basedir, conf.env]
    if "task" in conf.env_kwargs:
        dirs.append(conf.env_kwargs["task"])
    if "model" in conf.env_kwargs:
        model = (
            conf.env_kwargs["model-alias"]
            if "model-alias" in conf.env_kwargs
            else conf.env_kwargs.model
        )
        dirs.append(model)

    dirs.extend([conf.agent, conf.name])
    output_dir = os.path.join(*dirs)
    os.makedirs(output_dir, exist_ok=True)
    conf.output_dir = output_dir


def parse_step_from_checkpoint(path: str) -> int:
    """从检查点文件名中解析步数
    
    Args:
        path: 检查点文件路径
        
    Returns:
        检查点中的步数
    """
    try:
        filename = os.path.basename(path)
        assert filename.endswith(".ckpt")
        filename = filename[:-5]
        filename = filename.split("_")[1]
        return int(filename)
    except:
        raise Exception(f"无法从文件名 {path} 中获取步数")


def tensor_stats(t: torch.Tensor) -> torch.Tensor:
    """计算张量的统计信息
    
    Args:
        t: 输入张量
        
    Returns:
        包含最小值、最大值和标准差的张量
    """
    return torch.stack((t.min(), t.max(), t.std()))


def tailsum(t: torch.Tensor) -> torch.Tensor:
    """计算累积和（从后向前）
    
    Args:
        t: 输入张量
        
    Returns:
        累积和张量
    """
    return torch.flip(torch.cumsum(torch.flip(t, dims=(0,)), dim=0), dims=(0,))


def collate_summaries(summaries: List[Dict]) -> Dict:
    """合并多个摘要字典
    
    Args:
        summaries: 摘要字典列表
        
    Returns:
        合并后的摘要字典
    """
    if len(summaries) == 1:
        return summaries[0]

    for summary in summaries:
        assert summary.keys() == summaries[0].keys()

    collated = {}
    for key, value in summaries[0].items():
        values = [summary[key] for summary in summaries]
        if isinstance(value, (float, int)):
            # 计算平均值
            collated[key] = sum(values) / len(values)
            collated[key + "-raw"] = values
        elif isinstance(value, List):
            # 保留列表
            collated[key + "-raw"] = values
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            # 处理标量张量
            collated[key] = sum(values) / len(values)
            collated[key + "-raw"] = values
        else:
            # 默认处理为列表
            collated[key + "-raw"] = values

    return collated


def normalized_entropy(probs: torch.FloatTensor) -> torch.FloatTensor:
    """计算概率分布的标准化熵
    
    标准化熵将熵值除以最大可能熵（均匀分布的熵），
    结果范围在0到1之间。0表示确定性分布，1表示均匀分布。
    
    Args:
        probs: 概率分布张量，应该是一个正规化的概率分布（和为1）
        
    Returns:
        标准化熵值
    """
    # 确保输入是有效的概率分布
    eps = 1e-12
    probs = probs.clamp(min=eps, max=1.0)
    
    # 获取类别数
    num_classes = probs.shape[0]
    
    # 计算标准化熵
    raw_entropy = -(probs * torch.log2(probs)).sum()
    max_entropy = math.log2(num_classes)
    
    return raw_entropy / max_entropy
