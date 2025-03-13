import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging

from model_utils import predict
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
# 导入主动学习相关模块
from active_learning import (
    get_current_time_samples,
    integrate_active_samples,
    get_strategy
)

# 设置 transformers 日志级别为仅显示错误信息
tf_logging.set_verbosity_error()


def train_rl_strategy(train_data, model, tokenizer, args):
    """
    训练强化学习策略
    
    Args:
        train_data: 训练数据集
        model: 预测模型
        tokenizer: 分词器
        args: 运行参数
    """
    if not args.rl_train:
        return
    
    # 获取RL策略
    strategy = get_strategy("rl", config_path=args.rl_config)
    
    # 检查RL是否可用
    if not hasattr(strategy, "agent") or strategy.agent is None:
        try:
            # 导入所需模块
            from src.rl.tkg_environment import TKGEnvironment
            from src.rl.agents.dqn_agent import DQNAgent
            
            # 初始化环境
            environment = strategy._initialize_env(train_data, model, tokenizer, args)
            
            if environment is None:
                print("警告: 无法初始化RL环境，训练已跳过")
                return
                
            # 初始化代理
            agent = strategy._initialize_agent(environment, args)
            
            if agent is None:
                print("警告: 无法初始化RL代理，训练已跳过")
                return
                
            # 设置策略的环境和代理
            strategy.environment = environment
            strategy.agent = agent
            
            # 设置环境模式为训练模式
            if hasattr(environment, "set_mode"):
                environment.set_mode("train")
            
            print(f"开始训练RL策略，总步数: {agent.train_steps}")
            
            # 执行训练
            agent.train()
            
            print(f"RL策略训练完成，模型已保存到: {agent.output_dir}")
            
        except Exception as e:
            print(f"训练RL策略时出错: {e}")
    else:
        print("RL策略已初始化，开始训练")
        strategy.agent.train()


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()

    # 加载测试数据和搜索空间
    # test_data: 测试数据集
    # head_search_space: 头实体搜索空间
    # tail_search_space: 尾实体搜索空间
    test_data, head_search_space, tail_search_space = load_data(args)

    # test_data 最终格式：
    # [
    #     ([实体, 关系, [目标实体列表], 时间戳], "tail"),
    #     ([实体, 关系, [目标实体列表], 时间戳], "head"),
    #     ...
    # ]

    # 根据测试集调整 top-k 值
    adjust_top_k(test_data, args)

    # 加载预训练模型的分词器
    # tokenizer_revision: 指定要加载的模型和分词器版本/分支，默认为 "main"
    # tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision)
    tokenizer = AutoTokenizer.from_pretrained("/data/shangyuan/models/DeepSeek-R1-Distill-Qwen-1.5B", revision="main")
    # 设置填充标记为结束标记
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 添加在device = f"cuda:{args.gpu}" if... 这行之前
    if args.gpu == -2:  # 使用-2作为自动选择GPU的标志
        try:
            # 使用nvidia-smi命令获取GPU使用情况
            import subprocess
            import re
            output = subprocess.check_output('nvidia-smi --query-gpu=memory.used,index --format=csv,nounits,noheader', shell=True)
            output = output.decode('utf-8').strip().split('\n')
            gpu_usage = []
            for line in output:
                memory_used, gpu_id = map(int, re.findall(r'\d+', line))
                gpu_usage.append((memory_used, gpu_id))
            
            # 按内存使用量排序并选择最空闲的GPU
            gpu_usage.sort()
            args.gpu = gpu_usage[0][1]
            print(f"自动选择最空闲的GPU: {args.gpu}")
        except Exception as e:
            print(f"自动选择GPU失败: {e}，使用CPU")
            args.gpu = -1

    # 加载预训练语言模型
    # torch_dtype: 根据参数选择使用 FP16 或 FP32
    # device: 指定使用的 GPU 设备
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.float16 if args.fp16 else torch.float32,
    #     device_map=device,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        "/data/shangyuan/models/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map=device,
    )
    # 设置为评估模式
    model.eval()
    print(f"model is loaded on device {model.device.type}")

    # 如果启用主动学习，初始化策略
    active_strategy = None
    if args.active_learning:
        # 对于RL策略，传递配置路径
        if args.active_strategy == "rl":
            active_strategy = get_strategy(args.active_strategy, config_path=args.rl_config)
        else:
            active_strategy = get_strategy(args.active_strategy)
            
        print(f"主动学习已启用: 使用 {active_strategy.name} 策略")
        print(f"样本数量: {args.active_samples}, 集成方式: {args.active_integration}")
        
        # 如果使用的是RL策略并启用了训练，则训练RL模型
        if args.active_strategy == "rl" and args.rl_train:
            print("RL训练模式已启用，将在开始预测前训练RL策略")
            # 使用测试数据的一部分作为训练数据
            train_size = min(args.rl_train_size, len(test_data))
            train_data = test_data[:train_size]
            train_rl_strategy(train_data, model, tokenizer, args)

    # 初始化评估指标计算器
    metric = HitsMetric()
    # 获取输出文件名
    filename = get_filename(args)
    
    # 记录测试集大小
    test_data_size = len(test_data)
    
    # 开始处理数据
    # torch.no_grad(): 禁用梯度计算，用于推理阶段
    # tqdm: 显示进度条
    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_data) as pbar:
        # 获取所有唯一的时间戳
        timestamps = sorted(list(set([x[0][3] for x in test_data])))
        print(f"数据集中的时间戳: {timestamps}")
        
        # 遍历测试数据集
        for i, (x, direction) in enumerate(pbar):
            # 分布式处理：根据 rank 和 world_size 划分数据
            # world_size: 总进程数，rank: 当前进程编号
            if i % args.world_size != args.rank:
                continue
                
            current_time = x[3]
            
            # 根据预测方向选择搜索空间
            # direction == "tail": 预测尾实体，使用头实体搜索空间
            # direction == "head": 预测头实体，使用尾实体搜索空间
            if direction == "tail":
                search_space = head_search_space
            elif direction == "head":
                search_space = tail_search_space
            else:
                raise ValueError
            
            # 准备模型输入数据
            # return_prompt=True: 返回文本提示而不是统计信息
            model_input, candidates = prepare_input(x, search_space, args, return_prompt=True)
            
            # 如果启用主动学习，为当前预测样本获取主动学习采样
            if args.active_learning:
                # 查找当前时间点的所有样本（排除当前正在预测的样本）
                current_samples, current_indices = get_current_time_samples(test_data, current_time, current_sample_index=i)
                
                if current_samples:
                    # 如果使用RL策略，设置当前查询
                    if args.active_strategy == "rl":
                        rl_strategy = active_strategy
                        if hasattr(rl_strategy, "update_query"):
                            rl_strategy.update_query(((x[0], x[1], x[2], x[3]), direction))
                    
                    # 使用选定的策略选择活跃样本
                    selected_samples, _ = active_strategy.select_samples(
                        current_samples, current_indices, model, tokenizer, args
                    )
                    
                    # 将活跃样本整合到输入提示中
                    if selected_samples:
                        if args.verbose:
                            print(f"样本 {i}: 整合 {len(selected_samples)} 个专家标注样本")
                        model_input = integrate_active_samples(model_input, selected_samples, args)
            
            # 使用模型进行预测
            predictions = predict(tokenizer, model, model_input, args)

            # 更新历史记录
            update_history(x, search_space, predictions, candidates, args)

            # 将结果写入文件
            example = write_results(x, predictions, candidates, direction, writer, args)

            # 更新评估指标并显示
            update_metric(example, metric, args)
            
            # 更新进度条显示
            pbar.set_postfix(metric.dump())
        
        # 打印最终的统计信息
        if args.active_learning:
            print(f"测试集样本总数: {test_data_size}")
            print(f"每个预测样本额外整合了 {args.active_samples} 个专家标注样本")
        
        # 显示最终评估结果
        print(f"最终评估结果: {metric.dump()}")
