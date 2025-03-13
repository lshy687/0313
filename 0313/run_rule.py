import os
import torch
from tqdm import tqdm

from utils import (
    HitsMetric,
    adjust_top_k,
    get_args,
    get_filename,
    load_data,
    prepare_input,
    update_history,
    update_metric,
    write_results,
)

if __name__ == "__main__":
    # 获取命令行参数，包括数据集、模型类型、历史长度等配置
    args = get_args()

    # 加载数据集和搜索空间
    # test_data: 测试数据集，包含待预测的实体关系
    # head_search_space: 用于头实体预测的历史搜索空间，格式为 {head: {time: {relation: [tail]}}}
    # tail_search_space: 用于尾实体预测的历史搜索空间，格式为 {tail: {time: {relation: [head]}}}
    test_data, head_search_space, tail_search_space = load_data(args)

    # 根据测试集中最大目标数调整 top-k 值，确保能够正确评估所有可能的预测结果
    adjust_top_k(test_data, args)

    # 初始化评估指标计算器，用于计算 Hits@1, Hits@3, Hits@10 等指标
    metric = HitsMetric()
    # 生成输出文件名，包含模型配置信息
    filename = get_filename(args)
    
    # 确保路径是绝对路径
    filename = os.path.abspath(filename)
    output_dir = os.path.dirname(filename)
    
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/verified: {output_dir}")
        
        # 验证文件是否可写
        with open(filename, "w", encoding="utf-8") as test_write:
            pass
        print(f"Output file is writable: {filename}")
        
        # 开始处理数据
        # torch.no_grad(): 禁用梯度计算，因为只进行推理
        # tqdm: 显示进度条
        with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_data) as pbar:
            # 遍历测试数据集中的每个样本
            for i, (x, direction) in enumerate(pbar):
                # 分布式处理：根据 rank 和 world_size 划分数据
                # world_size: 总进程数，rank: 当前进程编号
                if i % args.world_size != args.rank:
                    continue

                # 根据预测方向选择对应的搜索空间
                # direction == "tail": 预测尾实体，使用头实体搜索空间
                # direction == "head": 预测头实体，使用尾实体搜索空间
                if direction == "tail":
                    search_space = head_search_space
                elif direction == "head":
                    search_space = tail_search_space
                else:
                    raise ValueError(f"Unknown direction: {direction}")

                # 准备输入数据
                # 根据历史数据和预测目标构建输入
                # return_prompt=False 表示返回统计信息而不是提示文本
                predictions, candidates = prepare_input(x, search_space, args, return_prompt=False)

                # 更新历史记录
                # 将当前预测结果添加到搜索空间中，用于后续的预测
                update_history(x, search_space, predictions, candidates, args)

                # 将预测结果写入文件
                # 包含时间戳、实体、关系、目标和预测等信息
                example = write_results(x, predictions, candidates, direction, writer, args)

                # 更新评估指标
                # 计算当前预测的 Hits@N 指标
                update_metric(example, metric, args)
                # 在进度条中显示当前的评估指标
                pbar.set_postfix(metric.dump())
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to write to: {filename}")
        raise
