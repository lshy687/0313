#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：验证强化学习代理与TKG环境的交互
"""

import os
import sys
import logging
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试RL代理与TKG环境交互")
    parser.add_argument("--model_path", type=str, default="/data/shangyuan/models/DeepSeek-R1-Distill-Qwen-1.5B", help="预训练模型路径")
    parser.add_argument("--config_path", type=str, default="rl_configs/tkg-agent.yaml", help="RL配置文件路径")
    parser.add_argument("--test_size", type=int, default=10, help="测试数据大小")
    parser.add_argument("--max_steps", type=int, default=3, help="每个环境最大步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_rl_environment(args):
    """测试RL环境功能"""
    from src.rl.tkg_environment import TKGEnvironment
    from utils import get_args, load_data
    
    # 获取默认参数
    default_args = get_args()
    default_args.model = args.model_path
    default_args.gpu = args.gpu
    default_args.active_samples = args.max_steps
    
    logger.info(f"加载模型: {args.model_path}")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    logger.info(f"模型已加载到设备: {model.device.type}")
    
    # 加载测试数据
    logger.info("加载测试数据")
    test_data, head_space, tail_space = load_data(default_args)
    test_subset = test_data[:args.test_size]
    logger.info(f"测试数据大小: {len(test_subset)}")
    
    # 初始化环境
    logger.info("初始化TKG环境")
    environment = TKGEnvironment(
        test_data=test_subset,
        model=model,
        tokenizer=tokenizer,
        args=default_args,
        max_steps=args.max_steps,
        reward_scale=10.0,
        state_repr=["sample_features", "query_features", "similarity", "history"],
        current_query=None,  # 将在后面设置
    )
    
    # 测试环境基本属性
    logger.info(f"环境状态维度: {environment.state_dim}")
    logger.info(f"初始环境模式: {environment.mode}")
    
    # 设置环境模式
    environment.set_mode("test")
    logger.info(f"更改后环境模式: {environment.mode}")
    
    # 准备示例查询
    example_query = test_subset[0]
    entity, relation, targets, timestamp = example_query[0]
    direction = example_query[1]
    logger.info(f"示例查询: {entity}, {relation}, {timestamp}, 方向: {direction}")
    
    # 更新环境查询
    environment.update_query(example_query)
    
    # 获取当前时间戳的样本（排除当前查询）
    current_samples = []
    current_indices = []
    for i, (sample, direction) in enumerate(test_subset):
        if i != 0 and sample[3] == timestamp:
            current_samples.append((sample, direction))
            current_indices.append(i)
    
    logger.info(f"当前时间戳的样本数量: {len(current_samples)}")
    
    # 更新环境的候选样本
    environment.update_candidates(current_samples, current_indices)
    
    # 重置环境
    state, _ = environment.reset()
    logger.info(f"初始状态: 形状={state.shape}")
    
    # 验证环境操作
    action_space = environment.action_space()
    logger.info(f"可用动作空间: {action_space}")
    logger.info(f"动作数量: {environment.action_count()}")
    
    return environment, test_subset

def test_agent_interaction(environment, args):
    """测试代理与环境交互"""
    from src.rl.agents.dqn_agent import DQNAgent
    import yaml
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    agent_kwargs = config.get("agent_kwargs", {})
    
    # 创建代理
    logger.info("初始化DQN代理")
    agent = DQNAgent(
        env=environment,
        output_dir=agent_kwargs.get("output_dir", "./rl_outputs/test"),
        overwrite_existing=True,  # 测试时覆盖现有输出
        train_steps=agent_kwargs.get("train_steps", 10),
        save_every=agent_kwargs.get("save_every", 5),
        eval_every=agent_kwargs.get("eval_every", 5),
        batch_size=agent_kwargs.get("batch_size", 4),
        replay_memory_size=agent_kwargs.get("replay_memory_size", 100),
        lr=agent_kwargs.get("lr", 0.001),
    )
    
    # 获取初始状态
    state, done = environment.reset()
    
    # 执行多步交互
    total_reward = 0
    total_steps = 0
    done = False
    
    logger.info("\n===== 开始环境交互 =====")
    
    while not done and total_steps < environment.max_steps:
        # 获取可用动作空间
        action_space = environment.action_space()
        if len(action_space) == 0:
            logger.info("没有可用动作，结束交互")
            break
        
        # 让代理选择动作
        action_idx = agent.choose_action(state, action_space)
        logger.info(f"步骤 {total_steps+1}: 代理选择动作 {action_idx}")
        
        # 执行动作
        next_state, reward, done, info = environment.step(action_idx)
        
        logger.info(f"  奖励: {reward:.4f}")
        logger.info(f"  MRR: {info.get('mrr', 0.0):.4f}")
        logger.info(f"  已选择样本数: {info.get('selected_count', 0)}")
        logger.info(f"  是否完成: {done}")
        
        # 更新状态
        state = next_state
        total_reward += reward
        total_steps += 1
        
        # 将经验添加到回放内存（确保所有输入都是张量）
        if not isinstance(action_idx, int) and hasattr(action_idx, 'item'):
            action_idx = action_idx.item()
        agent.replay_memory.push(state, action_idx, next_state, reward, done)
    
    logger.info(f"\n总步数: {total_steps}")
    logger.info(f"总奖励: {total_reward:.4f}")
    logger.info(f"环境摘要: {environment.summary()}")
    
    # 测试优化步骤
    if len(agent.replay_memory) > 0:
        logger.info("\n===== 测试代理优化 =====")
        logger.info(f"回放内存大小: {len(agent.replay_memory)}")
        logger.info("执行网络优化...")
        try:
            loss = agent.optimize_model()
            logger.info(f"优化损失: {loss:.6f}")
        except Exception as e:
            logger.error(f"优化过程中出错: {e}")
    
    return agent

def main():
    """主函数"""
    args = parse_args()
    set_seed(args.seed)
    
    try:
        # 测试环境
        logger.info("======= 开始测试RL环境 =======")
        environment, test_data = test_rl_environment(args)
        
        # 测试代理交互
        logger.info("\n======= 开始测试代理交互 =======")
        agent = test_agent_interaction(environment, args)
        
        # 测试简单训练
        logger.info("\n======= 测试简单训练过程 =======")
        logger.info(f"执行3步训练...")
        for i in range(3):
            agent.train_step()
            logger.info(f"训练步骤 {i+1} 完成")
        
        logger.info("\n测试成功完成!")
    
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 