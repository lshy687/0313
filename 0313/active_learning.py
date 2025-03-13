import random
import math
import numpy as np
import torch
from tqdm import tqdm
import os
import yaml
import logging
from omegaconf import OmegaConf

from model_utils import predict

# 添加RL相关导入
try:
    from src.rl.environment import BaseEnvironment
    from src.rl.tkg_environment import TKGEnvironment
    from src.rl.agents.dqn_agent import DQNAgent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("警告: 强化学习模块导入失败，RLStrategy将不可用")


def get_current_time_samples(test_data, timestamp, current_sample_index=None):
    """获取特定时间戳的所有样本，排除当前正在预测的样本
    
    Args:
        test_data: 测试数据列表
        timestamp: 目标时间戳
        current_sample_index: 当前正在预测的样本索引（可选），如果提供则排除此样本
    
    Returns:
        当前时间戳的样本列表和对应的索引列表
    """
    current_samples = []
    current_indices = []
    
    for i, (sample, direction) in enumerate(test_data):
        if sample[3] == timestamp and i != current_sample_index:
            # 返回原始样本（包含标签信息）
            current_samples.append((sample, direction))
            current_indices.append(i)
    
    return current_samples, current_indices


def integrate_active_samples(prompt, active_samples, args):
    """将主动学习选择的样本整合到提示中，包含真实标签（专家标注）"""
    if not active_samples:
        return prompt
        
    if args.active_integration == "direct":
        # 直接添加带标签的样本到提示
        for sample, direction in active_samples:
            entity, relation, targets, time = sample
            target = targets[0] if targets else "?"  # 使用第一个目标实体作为标签
            if not args.no_time:
                prompt += f"{time}:"
            if args.label:
                # 使用带标签格式
                prompt += f"[{entity},{relation},{targets.index(target) if target != '?' else '?'}. {target}]\n"
            else:
                # 使用实体格式
                prompt += f"[{entity},{relation},{target}]\n"
    elif args.active_integration == "labeled":
        # 添加标记为"专家标注"的样本
        prompt += "\n专家标注的当前事件:\n"
        for sample, direction in active_samples:
            entity, relation, targets, time = sample
            target = targets[0] if targets else "?"  # 使用第一个目标实体作为标签
            if not args.no_time:
                prompt += f"{time}:"
            if args.label:
                # 使用带标签格式
                prompt += f"[{entity},{relation},{targets.index(target) if target != '?' else '?'}. {target}]\n"
            else:
                # 使用实体格式
                prompt += f"[{entity},{relation},{target}]\n"
    
    return prompt


class BaseStrategy:
    """主动学习策略基类"""
    def __init__(self, name):
        self.name = name
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """选择样本的基本方法，子类应重写此方法"""
        raise NotImplementedError("子类必须实现select_samples方法")


class RandomStrategy(BaseStrategy):
    """随机选择策略"""
    def __init__(self):
        super().__init__("random")
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """随机选择指定数量的样本"""
        num_samples = min(args.active_samples, len(current_samples))
        if num_samples == 0:
            return [], []
        
        # 随机选择索引
        selected_idx = random.sample(range(len(current_samples)), num_samples)
        
        # 返回选中的样本和对应的原始索引
        selected_samples = [current_samples[i] for i in selected_idx]
        selected_indices = [current_indices[i] for i in selected_idx]
        
        return selected_samples, selected_indices


class RandomBalancedStrategy(BaseStrategy):
    """按方向平衡的随机选择策略"""
    def __init__(self):
        super().__init__("random_balanced")
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """平衡head和tail方向随机选择样本"""
        # 按方向分组
        head_samples = []
        head_indices = []
        tail_samples = []
        tail_indices = []
        
        for i, (sample, idx) in enumerate(zip(current_samples, current_indices)):
            if sample[1] == "head":
                head_samples.append(sample)
                head_indices.append(idx)
            else:
                tail_samples.append(sample)
                tail_indices.append(idx)
        
        # 计算每个方向应选择的样本数
        total_samples = min(args.active_samples, len(current_samples))
        head_count = min(len(head_samples), total_samples // 2 + total_samples % 2)
        tail_count = min(len(tail_samples), total_samples - head_count)
        
        # 如果一个方向的样本不足，从另一个方向补充
        if head_count < total_samples // 2 + total_samples % 2:
            tail_count = min(len(tail_samples), total_samples - head_count)
        
        # 随机选择每个方向的样本
        selected_head_idx = random.sample(range(len(head_samples)), head_count) if head_count > 0 else []
        selected_tail_idx = random.sample(range(len(tail_samples)), tail_count) if tail_count > 0 else []
        
        # 收集选中的样本和索引
        selected_samples = [head_samples[i] for i in selected_head_idx] + [tail_samples[i] for i in selected_tail_idx]
        selected_indices = [head_indices[i] for i in selected_head_idx] + [tail_indices[i] for i in selected_tail_idx]
        
        # 随机打乱选中的样本顺序
        combined = list(zip(selected_samples, selected_indices))
        random.shuffle(combined)
        selected_samples, selected_indices = zip(*combined) if combined else ([], [])
        
        return list(selected_samples), list(selected_indices)


class MaxEntropyStrategy(BaseStrategy):
    """最大熵策略"""
    def __init__(self):
        super().__init__("max_entropy")
    
    def _calculate_entropy(self, predictions):
        """计算预测结果的熵"""
        probs = [p[1] for p in predictions]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """选择预测熵最大的样本"""
        if not current_samples:
            return [], []
        
        print(f"MaxEntropyStrategy: 计算{len(current_samples)}个样本的熵...")
        
        # 计算每个样本的熵
        sample_entropies = []
        
        for i, (sample, _) in enumerate(tqdm(current_samples)):
            try:
                # 创建空搜索空间（因为我们只需要计算熵，不需要实际预测）
                empty_search_space = {}
                
                # 准备输入
                from utils import prepare_input
                model_input, _ = prepare_input(sample, empty_search_space, args, return_prompt=True)
                
                # 预测并计算熵
                predictions = predict(tokenizer, model, model_input, args)
                entropy_value = self._calculate_entropy(predictions)
                
                sample_entropies.append((i, entropy_value))
            except Exception as e:
                print(f"计算样本{i}的熵时出错: {e}")
                # 出错时赋予最小熵值
                sample_entropies.append((i, -float('inf')))
        
        # 按熵降序排序并选择指定数量的样本
        sample_entropies.sort(key=lambda x: x[1], reverse=True)
        num_samples = min(args.active_samples, len(sample_entropies))
        
        selected_idx = [idx for idx, _ in sample_entropies[:num_samples]]
        selected_samples = [current_samples[i] for i in selected_idx]
        selected_indices = [current_indices[i] for i in selected_idx]
        
        return selected_samples, selected_indices


def parse_step_from_checkpoint(checkpoint_path):
    """
    从检查点路径中解析训练步数
    
    Args:
        checkpoint_path: 检查点文件路径，格式通常为"path/to/model_STEP.pt"
    
    Returns:
        训练步数
    """
    try:
        # 尝试从文件名中提取步数
        filename = os.path.basename(checkpoint_path)
        if "_" in filename:
            # 格式如 "model_1000.pt"
            step_str = filename.split("_")[-1].split(".")[0]
            return int(step_str)
        else:
            # 格式可能是 "model.pt"
            return 0
    except:
        print(f"无法从路径 {checkpoint_path} 解析步数，使用默认步数0")
        return 0


class BestOfKStrategy(BaseStrategy):
    """从K个随机集合中选择最佳子集"""
    def __init__(self):
        super().__init__("best_of_k")
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """从K个随机子集中选择平均熵最高的子集"""
        if not current_samples:
            return [], []
        
        # 参数设置
        K = 5  # 尝试的随机子集数量
        subset_size = min(args.active_samples, len(current_samples))
        
        if subset_size == 0:
            return [], []
        
        print(f"BestOfKStrategy: 评估{K}个随机子集...")
        
        best_subset = None
        best_entropy = -float('inf')
        
        for k in range(K):
            # 随机选择一个子集
            random_idx = random.sample(range(len(current_samples)), subset_size)
            subset_samples = [current_samples[i] for i in random_idx]
            subset_indices = [current_indices[i] for i in random_idx]
            
            # 计算子集的平均熵
            entropy_sum = 0
            sample_count = 0
            
            for sample, _ in subset_samples:
                try:
                    # 创建空搜索空间
                    empty_search_space = {}
                    
                    # 准备输入
                    from utils import prepare_input
                    model_input, _ = prepare_input(sample, empty_search_space, args, return_prompt=True)
                    
                    # 预测并计算熵
                    predictions = predict(tokenizer, model, model_input, args)
                    entropy_value = self._calculate_entropy(predictions)
                    
                    entropy_sum += entropy_value
                    sample_count += 1
                except Exception as e:
                    print(f"计算样本熵时出错: {e}")
            
            # 计算平均熵
            if sample_count > 0:
                avg_entropy = entropy_sum / sample_count
                
                # 更新最佳子集
                if avg_entropy > best_entropy:
                    best_entropy = avg_entropy
                    best_subset = (subset_samples, subset_indices)
        
        if best_subset:
            return best_subset
        else:
            # 如果没有找到最佳子集，回退到随机选择
            return RandomStrategy().select_samples(current_samples, current_indices, model, tokenizer, args)


class RLStrategy(BaseStrategy):
    """基于强化学习的样本选择策略"""
    def __init__(self, config_path=None):
        super().__init__("rl")
        self.config_path = config_path or "rl_configs/tkg-agent.yaml"
        self.agent = None
        self.environment = None
        self.model_path = None
        
        # 设置日志对象
        self.logger = logging.getLogger("RLStrategy")
        
        # 检查RL模块是否可用
        if not RL_AVAILABLE:
            self.logger.warning("强化学习模块不可用，将使用随机策略代替")
            print("警告: 强化学习模块不可用，将使用随机策略代替")
        
    def _load_config(self, args=None):
        """加载RL配置
        
        Args:
            args: 可选的命令行参数，用于获取配置文件路径
            
        Returns:
            配置字典
        """
        # 如果提供了args且包含rl_config参数，使用它
        config_path = self.config_path
        if args is not None and hasattr(args, "rl_config"):
            config_path = args.rl_config
            
        if not os.path.exists(config_path):
            self.logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件 {config_path} 时出错: {e}")
            print(f"错误: 加载配置文件 {config_path} 时出错: {e}")
            return {}
        
    def _initialize_env(self, test_data, model, tokenizer, args):
        """初始化RL环境"""
        if not RL_AVAILABLE:
            return None
            
        config = self._load_config(args)
        env_kwargs = config.get("env_kwargs", {})
        
        # 设置环境参数
        max_steps = env_kwargs.get("max_steps", args.active_samples)
        reward_scale = env_kwargs.get("reward_scale", 1.0)
        state_repr = env_kwargs.get("state_repr", ["sample_features"])
        
        # 创建TKG环境
        environment = TKGEnvironment(
            test_data=test_data,
            model=model,
            tokenizer=tokenizer,
            args=args,
            state_repr=state_repr,
            max_steps=max_steps,
            reward_scale=reward_scale,
        )
        
        return environment
        
    def _initialize_agent(self, environment, args=None):
        """初始化RL代理
        
        Args:
            environment: 强化学习环境
            args: 可选的命令行参数，用于获取模型路径
            
        Returns:
            初始化好的RL代理
        """
        if not RL_AVAILABLE or environment is None:
            return None
            
        config = self._load_config(args)
        agent_kwargs = config.get("agent_kwargs", {})
        
        # 创建代理
        agent = DQNAgent(
            env=environment,
            output_dir=agent_kwargs.get("output_dir", "./rl_outputs"),
            overwrite_existing=agent_kwargs.get("overwrite_existing", False),
            train_steps=agent_kwargs.get("train_steps", 400),
            save_every=agent_kwargs.get("save_every", 100),
            eval_every=agent_kwargs.get("eval_every", 100),
            batch_size=agent_kwargs.get("batch_size", 32),
            replay_memory_size=agent_kwargs.get("replay_memory_size", 1000),
            lr=agent_kwargs.get("lr", 0.001),
        )
        
        # 从命令行参数或配置文件中获取模型路径
        model_path = None
        if args is not None and hasattr(args, "rl_model_path") and args.rl_model_path:
            model_path = args.rl_model_path
        else:
            model_path = agent_kwargs.get("model_path", None)
            
        # 加载预训练的模型（如果存在）
        if model_path and os.path.exists(model_path):
            try:
                agent.load_model_at_step(parse_step_from_checkpoint(model_path))
                print(f"成功加载RL模型: {model_path}")
            except Exception as e:
                print(f"加载RL模型失败: {e}")
        
        return agent
    
    def select_samples(self, current_samples, current_indices, model, tokenizer, args):
        """使用强化学习选择样本
        
        Args:
            current_samples: 当前时间戳的样本列表
            current_indices: 当前时间戳的样本索引列表
            model: 预测模型
            tokenizer: 分词器
            args: 运行参数
            
        Returns:
            选中的样本和对应的索引
        """
        if not RL_AVAILABLE or not current_samples:
            # 如果RL不可用或没有可用样本，使用随机策略
            return RandomStrategy().select_samples(current_samples, current_indices, model, tokenizer, args)
        
        # 获取完整测试数据（用于创建环境）
        # 注意：这里假设传入的current_samples来自同一个测试集
        # 实际使用时可能需要修改代码来适应不同的数据源
        from run_hf import test_data as full_test_data
        
        try:
            # 初始化环境和代理
            if self.environment is None:
                self.environment = self._initialize_env(full_test_data, model, tokenizer, args)
            
            if self.agent is None and self.environment is not None:
                self.agent = self._initialize_agent(self.environment, args)
            
            if self.environment is None or self.agent is None:
                print("警告: 初始化RL环境或代理失败，回退到随机策略")
                return RandomStrategy().select_samples(current_samples, current_indices, model, tokenizer, args)
            
            # 设置环境模式为验证模式
            if hasattr(self.environment, "set_mode"):
                self.environment.set_mode("val")
            
            # 设置当前样本和索引
            self.environment.current_samples = current_samples
            self.environment.current_indices = current_indices
            self.environment.steps_taken = 0
            self.environment.selected_samples = []
            self.environment.selected_indices = []
            
            # 计算初始MRR
            initial_mrr, _, _ = self.environment.evaluate_mrr([])
            self.environment.mrr_history = [initial_mrr]
            
            # 获取初始状态
            state = self.environment.state
            
            # 选择样本直到达到要求的数量或没有更多样本
            selected_samples = []
            selected_indices = []
            num_samples = min(args.active_samples, len(current_samples))
            
            for _ in range(num_samples):
                # 使用代理选择动作
                action_idx = self.agent.choose_action(state, self.environment.action_space())
                
                # 执行动作
                next_state, reward, done, info = self.environment.step(action_idx)
                
                # 更新选择的样本和索引
                selected_samples.append(self.environment.selected_samples[-1])
                selected_indices.append(self.environment.selected_indices[-1])
                
                # 更新状态
                state = next_state
                
                # 如果环境结束，退出循环
                if done:
                    break
            
            # 在线学习：根据当前经验更新代理（如果配置允许）
            self._update_agent_online(args)
            
            return selected_samples, selected_indices
        
        except Exception as e:
            print(f"使用RL策略选择样本时出错: {e}")
            # 出错时回退到随机策略
            return RandomStrategy().select_samples(current_samples, current_indices, model, tokenizer, args)
            
    def _update_agent_online(self, args):
        """
        在线更新代理（如果配置允许）
        
        Args:
            args: 运行参数
        """
        if not RL_AVAILABLE or self.agent is None or self.environment is None:
            return
            
        # 检查配置中是否启用了在线学习
        config = self._load_config(args)
        online_learning = config.get("online_learning", {})
        enabled = online_learning.get("enabled", False)
        
        if not enabled:
            return
            
        # 获取在线学习参数
        update_frequency = online_learning.get("update_frequency", 10)
        optimization_steps = online_learning.get("optimization_steps", 5)
        min_experiences = online_learning.get("min_experiences", 32)
        
        # 检查是否有足够的经验进行更新
        if len(self.agent.replay_memory) < min_experiences:
            return
            
        # 使用类变量跟踪样本计数
        if not hasattr(self, "_online_sample_count"):
            self._online_sample_count = 0
            
        self._online_sample_count += 1
        
        # 如果达到更新频率，执行优化
        if self._online_sample_count % update_frequency == 0:
            print(f"执行在线学习更新 (样本计数: {self._online_sample_count})")
            
            # 设置环境为训练模式
            if hasattr(self.environment, "set_mode"):
                original_mode = self.environment.mode
                self.environment.set_mode("train")
                
            # 执行几步优化
            for _ in range(optimization_steps):
                self.agent.optimize_model()
                
            # 恢复原始模式
            if hasattr(self.environment, "set_mode"):
                self.environment.set_mode(original_mode)
                
            print(f"在线学习更新完成，执行了{optimization_steps}步优化")
            
    def update_query(self, query):
        """
        更新当前查询
        
        Args:
            query: 当前查询样本和方向的元组 ((entity, relation, targets, timestamp), direction)
        """
        if not RL_AVAILABLE or self.environment is None:
            return
            
        # 如果环境支持更新查询，则设置
        if hasattr(self.environment, "update_query"):
            self.environment.update_query(query)
            print(f"已更新环境查询: {query[0][0]}, {query[0][1]}, 方向: {query[1]}")
            
    def update_exploration(self, ratio=None):
        """
        更新探索率
        
        Args:
            ratio: 新的探索率，如果为None则使用配置中的值
        """
        if not RL_AVAILABLE or self.agent is None:
            return
            
        # 如果提供了比例，更新探索率
        if ratio is not None:
            if hasattr(self.agent, "set_exploration"):
                self.agent.set_exploration(ratio)
                print(f"已更新探索率: {ratio}")
            elif hasattr(self.agent, "epsilon"):
                self.agent.epsilon = ratio
                print(f"已更新探索率 (epsilon): {ratio}")
                
        # 否则，从配置中获取探索率
        else:
            # 使用最后一次加载的配置
            config = self._load_config()
            online_learning = config.get("online_learning", {})
            default_ratio = online_learning.get("exploration_ratio", 0.1)
            
            if hasattr(self.agent, "set_exploration"):
                self.agent.set_exploration(default_ratio)
                print(f"已重置探索率为默认值: {default_ratio}")
            elif hasattr(self.agent, "epsilon"):
                self.agent.epsilon = default_ratio
                print(f"已重置探索率 (epsilon)为默认值: {default_ratio}")


def get_strategy(strategy_name, config_path=None):
    """根据策略名称获取对应的策略实例
    
    Args:
        strategy_name: 策略名称
        config_path: RL策略的配置文件路径（可选）
    
    Returns:
        策略实例
    """
    # 基本策略实例
    strategies = {
        "random": RandomStrategy(),
        "max_entropy": MaxEntropyStrategy(),
        "best_of_k": BestOfKStrategy(),
        "random_balanced": RandomBalancedStrategy(),
    }
    
    # 特殊处理RL策略
    if strategy_name == "rl":
        strategies["rl"] = RLStrategy(config_path=config_path)
    
    if strategy_name not in strategies:
        print(f"警告: 未知策略 '{strategy_name}'，使用默认的随机策略")
        return strategies["random"]
    
    return strategies[strategy_name]
