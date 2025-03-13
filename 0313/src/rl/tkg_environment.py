import logging
import math
import random
import hashlib
from typing import Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model_utils import predict
from utils import prepare_input, HitsMetric  # 导入HitsMetric类
from .environment import BaseEnvironment
from .plm_encoder import PLMEncoder  # 导入PLM编码器

logger = logging.getLogger(__name__)


class TKGEnvironment(BaseEnvironment):
    """
    用于时序知识图谱预测的强化学习环境
    
    状态：当前时间戳下的可用样本特征，融合了查询信息
    动作：选择某个样本进行专家标注
    奖励：基于模型性能变化的奖励（使用MRR指标）
    """
    
    def __init__(
        self,
        test_data,
        model,
        tokenizer,
        args,
        state_repr=["sample_features", "query_features", "similarity", "history"],
        max_steps=5,
        reward_scale=1.0,
        current_query=None,  # 当前查询
    ):
        """
        初始化TKG环境
        
        Args:
            test_data: 测试数据集
            model: 预测模型
            tokenizer: 分词器
            args: 运行参数
            state_repr: 状态表示方式
            max_steps: 每个时间戳最大选择样本数
            reward_scale: 奖励缩放因子
            current_query: 当前查询（可选）
        """
        self.test_data = test_data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.state_repr = state_repr
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        
        # 环境模式：训练、验证或测试
        self.mode = "train"
        
        # 初始化PLM编码器
        self.encoder = PLMEncoder(model, tokenizer)
        
        # 提取所有唯一的时间戳
        self.timestamps = sorted(list(set([x[0][3] for x in test_data])))
        
        # 当前环境状态
        self.current_timestamp_idx = 0
        self.current_timestamp = None
        self.current_samples = []
        self.current_indices = []
        self.selected_samples = []
        self.selected_indices = []
        self.mrr_history = []  # 存储MRR历史
        self.steps_taken = 0
        self.current_query = current_query  # 当前查询
        
        # 创建实体和关系的ID映射
        self.entity_ids = {}
        self.relation_ids = {}
        self._create_id_mappings()
        
        # 初始化特征维度
        self.feature_dim = self._calculate_feature_dim()
        
        # 缓存已计算的特征
        self.feature_cache = {}
        self.semantic_cache = {}  # 语义特征缓存
        
        self.reset()
    
    def _create_id_mappings(self):
        """创建实体和关系的ID映射"""
        # 对所有样本中的实体和关系创建唯一ID
        for (sample, _) in self.test_data:
            entity, relation, targets, _ = sample
            
            if entity not in self.entity_ids:
                self.entity_ids[entity] = len(self.entity_ids)
                
            if relation not in self.relation_ids:
                self.relation_ids[relation] = len(self.relation_ids)
                
            for target in targets:
                if target not in self.entity_ids:
                    self.entity_ids[target] = len(self.entity_ids)
    
    def _calculate_feature_dim(self):
        """计算特征维度"""
        # 基础特征维度
        base_dim = 64
        
        # 根据状态表示类型计算总维度
        total_dim = 0
        
        # 实体特征
        if "sample_features" in self.state_repr:
            total_dim += base_dim
            
        # 查询特征
        if "query_features" in self.state_repr:
            # 查询的语义特征维度取决于模型输出
            query_dim = 384  # 默认维度
            if hasattr(self, "encoder") and self.encoder is not None:
                # 尝试获取实际维度
                try:
                    if self.current_query is not None:
                        sample, direction = self.current_query
                        embedding = self._extract_semantic_features(sample, direction)
                        query_dim = embedding.shape[0]
                except:
                    pass
            total_dim += query_dim
            
        # 相似度特征
        if "similarity" in self.state_repr:
            total_dim += 1  # 余弦相似度是单个值
            
        # 历史信息特征
        if "history" in self.state_repr:
            total_dim += 32
            
        # 当前步骤信息
        if "curr_step" in self.state_repr:
            total_dim += 16
            
        return max(total_dim, base_dim)  # 确保至少有基础维度
    
    def _extract_sample_features(self, sample, direction):
        """
        提取样本特征
        
        Args:
            sample: (entity, relation, targets, timestamp)
            direction: "head" 或 "tail"
        
        Returns:
            特征向量
        """
        # 创建缓存键
        cache_key = f"{sample[0]}_{sample[1]}_{direction}_{sample[3]}"
        
        # 如果特征已缓存，直接返回
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        entity, relation, targets, timestamp = sample
        
        # 获取实体和关系的ID
        entity_id = self.entity_ids.get(entity, 0)
        relation_id = self.relation_ids.get(relation, 0)
        
        # 提取时间戳特征（归一化）
        try:
            timestamp_feature = float(timestamp) / max(float(self.timestamps[-1]), 1.0)
        except:
            # 如果时间戳不是数字，使用哈希
            hash_val = int(hashlib.md5(str(timestamp).encode()).hexdigest(), 16)
            timestamp_feature = (hash_val % 1000) / 1000.0
        
        # 方向特征
        direction_feature = 1.0 if direction == "head" else 0.0
        
        # 目标数量特征（归一化）
        targets_count = len(targets) / 10.0  # 假设最多10个目标
        
        # 构建基础特征向量
        feature_vector = [
            entity_id / max(len(self.entity_ids), 1),
            relation_id / max(len(self.relation_ids), 1),
            timestamp_feature,
            direction_feature,
            targets_count,
        ]
        
        # 填充到基础维度
        base_dim = 64
        padding = [0.0] * (base_dim - len(feature_vector))
        feature_vector.extend(padding)
        
        # 转换为张量
        feature = torch.tensor(feature_vector, dtype=torch.float32)
        
        # 缓存计算结果
        self.feature_cache[cache_key] = feature
        
        return feature
    
    def get_available_samples(self):
        """获取当前时间戳下可用的样本"""
        if self.current_timestamp_idx >= len(self.timestamps):
            return [], [], None
        
        timestamp = self.timestamps[self.current_timestamp_idx]
        samples = []
        indices = []
        
        for i, (sample, direction) in enumerate(self.test_data):
            if sample[3] == timestamp and i not in self.selected_indices:
                samples.append((sample, direction))
                indices.append(i)
        
        return samples, indices, timestamp
    
    def evaluate_mrr(self, selected_samples=None):
        """
        评估在选择给定样本作为专家标注后的MRR（平均倒数排名）
        
        Args:
            selected_samples: 选择的样本列表，如果为None则使用当前已选择的样本
            
        Returns:
            (MRR值, 总样本数, 倒数排名总和)
        """
        # 如果没有指定样本，使用当前已选择的样本
        if selected_samples is None:
            selected_samples = self.selected_samples
            
        # 使用HitsMetric来计算MRR
        metric = HitsMetric()
        
        # 如果当前没有查询，直接返回0
        if self.current_query is None:
            logger.debug("没有当前查询，返回默认MRR值0.0")
            return 0.0, 0, 0
            
        # 获取当前查询
        query_sample, query_direction = self.current_query
        entity, relation, targets, timestamp = query_sample
        
        logger.debug(f"评估查询: {entity}, {relation}, {timestamp}, 方向: {query_direction}")
        logger.debug(f"已选择的样本数: {len(selected_samples)}")
        
        try:
            # 准备输入，包括已选择的样本作为专家标注
            prompt = self._prepare_prompt_with_selected(query_sample, query_direction, selected_samples)
            
            # 进行预测
            predictions = predict(self.tokenizer, self.model, prompt, self.args)
            
            if not predictions:
                logger.debug("模型未返回预测结果，返回默认MRR值0.0")
                return 0.0, 0, 0
                
            logger.debug(f"预测结果数量: {len(predictions)}")
            if predictions and len(predictions) > 0:
                logger.debug(f"前几个预测结果: {predictions[:min(5, len(predictions))]}")
            logger.debug(f"正确目标: {targets}")
            
            # 计算排名（寻找正确答案的排名）
            if predictions:
                # 检查预测结果中正确答案的排名
                correct_targets = query_sample[2]  # 正确的目标实体列表
                
                # 找到第一个正确答案的排名
                rank = float('inf')
                for i, (pred_entity, _) in enumerate(predictions, 1):
                    if pred_entity in correct_targets:
                        rank = i
                        logger.debug(f"找到正确答案 '{pred_entity}'，排名: {rank}")
                        break
                
                # 如果找到了正确答案，更新指标
                if rank != float('inf'):
                    metric.total += 1
                    metric.update(rank)
                else:
                    # 未找到正确答案时，视为最大排名（使用预测列表长度）
                    metric.total += 1
                    max_rank = len(predictions) + 1
                    logger.debug(f"未找到正确答案，使用最大排名: {max_rank}")
                    metric.update(max_rank)
            
        except Exception as e:
            logger.error(f"预测样本时出错: {e}")
            # 出错时返回当前MRR
            return self.mrr_history[-1] if self.mrr_history else 0.0, 0, 0
        
        # 防止除以零错误
        if metric.total == 0:
            logger.debug("评估样本数为0，返回默认MRR值0.0")
            return 0.0, 0, 0
            
        # 计算MRR（平均倒数排名）
        mrr = metric.mrr_sum / metric.total
        logger.debug(f"计算得到MRR: {mrr:.4f}, 总样本数: {metric.total}, 倒数排名总和: {metric.mrr_sum:.4f}")
        return mrr, metric.total, metric.mrr_sum
    
    def _prepare_prompt_with_selected(self, sample, direction, selected_samples):
        """
        准备包含已选择样本的提示
        
        Args:
            sample: 当前样本元组 (entity, relation, targets, timestamp)
            direction: 预测方向 "head" 或 "tail"
            selected_samples: 已选择的样本列表
            
        Returns:
            包含已选择样本的提示
        """
        # 创建空搜索空间（因为我们只需要获取提示，不需要实际搜索空间）
        empty_search_space = {}
        model_input, _ = prepare_input((sample, direction), empty_search_space, self.args, return_prompt=True)
        
        # 将选定的样本添加到提示中
        from active_learning import integrate_active_samples
        model_input = integrate_active_samples(model_input, selected_samples, self.args)
        
        return model_input
    
    @property
    def state_dim(self):
        """状态空间维度"""
        # 根据状态表示类型和当前样本数计算动态维度
        dim = 0
        
        # 样本特征
        if "sample_features" in self.state_repr:
            # 每个样本一个特征向量
            sample_dim = 64  # 基础特征维度
            dim += len(self.current_samples) * sample_dim
        
        # 查询特征
        if "query_features" in self.state_repr and self.current_query is not None:
            # 每个样本都有一份查询特征
            query_dim = 384  # 默认维度
            if hasattr(self, "encoder") and self.encoder is not None:
                try:
                    sample, direction = self.current_query
                    embedding = self._extract_semantic_features(sample, direction)
                    query_dim = embedding.shape[0]
                except:
                    pass
            dim += len(self.current_samples) * query_dim
            
        # 相似度特征
        if "similarity" in self.state_repr and self.current_query is not None:
            # 每个样本与查询的相似度
            dim += len(self.current_samples)
        
        # 历史信息
        if "history" in self.state_repr:
            dim += 32  # 历史特征维度
            
        # 当前步骤
        if "curr_step" in self.state_repr:
            dim += 16  # 步骤信息维度
            
        # 如果维度为0，使用基本特征维度
        return max(dim, self.feature_dim)
    
    @property
    def action_dim(self):
        """动作空间维度"""
        return len(self.current_samples)
    
    @property
    def state(self):
        """当前状态表示"""
        state_parts = []
        
        # 如果没有当前样本，返回零向量
        if not self.current_samples:
            return torch.zeros(self.feature_dim)
            
        # 每个样本为一行，创建特征矩阵
        sample_features = []
        
        for i, (sample, direction) in enumerate(self.current_samples):
            sample_state = []
            
            # 1. 添加样本特征
            if "sample_features" in self.state_repr:
                # 使用已有的特征提取
                basic_feature = self._extract_sample_features(sample, direction)
                sample_state.append(basic_feature)
                
            # 2. 添加查询特征
            if "query_features" in self.state_repr and self.current_query is not None:
                query_sample, query_direction = self.current_query
                query_embedding = self._extract_semantic_features(query_sample, query_direction)
                sample_state.append(query_embedding)
                
            # 3. 添加相似度特征
            if "similarity" in self.state_repr and self.current_query is not None:
                query_sample, query_direction = self.current_query
                sim = self._calculate_query_sample_similarity(
                    query_sample, query_direction, sample, direction
                )
                sample_state.append(sim.unsqueeze(0))  # 扩展为1维张量
                
            # 将该样本的所有特征连接起来
            if sample_state:
                sample_features.append(torch.cat(sample_state))
        
        # 将所有样本特征堆叠为一个矩阵
        if sample_features:
            state_parts.append(torch.stack(sample_features))
            
        # 添加历史信息
        if "history" in self.state_repr:
            # 选择样本数量和MRR历史
            history_feature = [
                self.steps_taken / max(self.max_steps, 1),
                len(self.selected_samples) / max(len(self.selected_samples) + len(self.current_samples), 1),
                self.mrr_history[-1] if self.mrr_history else 0.0,
            ]
            # 填充到历史维度
            history_padding = [0.0] * (32 - len(history_feature))
            history_feature.extend(history_padding)
            state_parts.append(torch.tensor(history_feature, dtype=torch.float32))
        
        # 添加当前步骤信息
        if "curr_step" in self.state_repr:
            step_feature = [
                self.steps_taken / max(self.max_steps, 1),
                self.current_timestamp_idx / max(len(self.timestamps), 1),
            ]
            # 填充到步骤维度
            step_padding = [0.0] * (16 - len(step_feature))
            step_feature.extend(step_padding)
            state_parts.append(torch.tensor(step_feature, dtype=torch.float32))
        
        # 如果没有状态部分，使用基本零向量
        if not state_parts:
            return torch.zeros(self.feature_dim)
            
        # 将各部分拉平并连接
        flattened_parts = []
        for part in state_parts:
            if len(part.shape) > 1:
                flattened_parts.append(part.reshape(-1))  # 拉平多维张量
            else:
                flattened_parts.append(part)
                
        return torch.cat(flattened_parts)
    
    def reset(self):
        """重置环境"""
        self.selected_samples = []
        self.selected_indices = []
        self.mrr_history = []
        self.steps_taken = 0
        
        # 获取初始样本（如果没有设置current_samples，使用时间戳）
        if not self.current_samples:
            self.current_samples, self.current_indices, self.current_timestamp = self.get_available_samples()
        
        # 如果没有样本可用，环境已结束
        if not self.current_samples:
            return self.state, True
            
        # 计算初始MRR
        if self.current_query is not None:
            initial_mrr, _, _ = self.evaluate_mrr([])
            self.mrr_history.append(initial_mrr)
        else:
            self.mrr_history.append(0.0)
        
        return self.state, False
    
    def action_count(self):
        """可用动作数量"""
        return len(self.current_samples)
    
    def action_space(self):
        """动作空间"""
        return list(range(len(self.current_samples)))
    
    def step(self, action_idx):
        """
        执行一步动作（选择一个样本）
        
        Args:
            action_idx: 动作索引，对应要选择的样本索引
        
        Returns:
            (下一状态, 奖励, 是否终止, 信息)
        """
        if action_idx < 0 or action_idx >= len(self.current_samples):
            raise ValueError(f"无效的动作索引: {action_idx}, 应该在 [0, {len(self.current_samples)-1}] 范围内")
        
        # 选择样本
        selected_sample = self.current_samples[action_idx]
        selected_idx = self.current_indices[action_idx]
        
        sample, direction = selected_sample
        entity, relation, _, timestamp = sample
        logger.debug(f"选择样本: {entity}, {relation}, {timestamp}, 方向: {direction}, 索引: {action_idx}")
        
        # 添加到已选择列表
        self.selected_samples.append(selected_sample)
        self.selected_indices.append(selected_idx)
        
        # 从当前可用样本中移除
        self.current_samples.pop(action_idx)
        self.current_indices.pop(action_idx)
        
        # 增加步数
        self.steps_taken += 1
        logger.debug(f"步数: {self.steps_taken}/{self.max_steps}")
        
        # 计算新的MRR
        new_mrr, total, reciprocal_sum = self.evaluate_mrr()
        
        # 计算奖励（MRR提升）
        prev_mrr = self.mrr_history[-1] if self.mrr_history else 0.0
        base_reward = (new_mrr - prev_mrr) * self.reward_scale
        reward = base_reward
        logger.debug(f"基础奖励(MRR提升): {base_reward:.4f}, 从{prev_mrr:.4f}到{new_mrr:.4f}")
        
        # 如果奖励为负，我们可以适当减少惩罚强度
        if reward < 0:
            old_reward = reward
            reward = reward * 0.5  # 减轻负奖励的影响
            logger.debug(f"负奖励减轻: {old_reward:.4f} -> {reward:.4f}")
            
        # 如果是最后一个可选样本，给予额外奖励（鼓励选择完所有样本）
        if not self.current_samples and self.steps_taken < self.max_steps:
            complete_bonus = 0.1 * self.reward_scale
            reward += complete_bonus
            logger.debug(f"完成所有样本奖励: +{complete_bonus:.4f}")
            
        # 如果样本与查询相似度高，给予额外奖励
        similarity_reward = 0.0
        if self.current_query is not None:
            query_sample, query_direction = self.current_query
            sample, direction = selected_sample
            similarity = self._calculate_query_sample_similarity(
                query_sample, query_direction, sample, direction
            )
            # 相似度奖励，最大0.5，确保不会超过主要奖励
            similarity_reward = min(similarity.item() * 0.5, 0.5) * self.reward_scale
            reward += similarity_reward
            logger.debug(f"相似度奖励: +{similarity_reward:.4f} (相似度: {similarity.item():.4f})")
            
        self.mrr_history.append(new_mrr)
        logger.debug(f"总奖励: {reward:.4f}")
        
        # 检查是否达到最大步数或没有可用样本
        done = self.steps_taken >= self.max_steps or not self.current_samples
        if done:
            logger.debug(f"环境结束: 达到最大步数={self.steps_taken >= self.max_steps}, 没有样本={not self.current_samples}")
        
        # 如果当前时间戳的样本已处理完，移到下一个时间戳
        if done and self.current_timestamp_idx < len(self.timestamps) - 1:
            self.current_timestamp_idx += 1
            next_samples, next_indices, next_timestamp = self.get_available_samples()
            
            if next_samples:
                self.current_samples = next_samples
                self.current_indices = next_indices
                self.current_timestamp = next_timestamp
                self.selected_samples = []
                self.selected_indices = []
                self.steps_taken = 0
                
                # 计算新时间戳的初始MRR
                if self.current_query is not None:
                    initial_mrr, _, _ = self.evaluate_mrr([])
                    self.mrr_history = [initial_mrr]
                else:
                    self.mrr_history = [0.0]
                
                done = False
        
        # 额外信息
        info = {
            "mrr": new_mrr,
            "total": total,
            "reciprocal_sum": reciprocal_sum,
            "mrr_history": self.mrr_history,
            "selected_count": len(self.selected_samples),
            "remaining_count": len(self.current_samples),
            "timestamp": self.current_timestamp,
            "reward": reward,
        }
        
        return self.state, reward, done, info
    
    def summary(self):
        """返回环境摘要信息"""
        return {
            f"{self.mode}/最终MRR": self.mrr_history[-1] if self.mrr_history else 0.0,
            f"{self.mode}/MRR变化": self.mrr_history,
            f"{self.mode}/选择的样本数": len(self.selected_samples),
            f"{self.mode}/当前时间戳": self.current_timestamp,
            f"{self.mode}/时间戳索引": self.current_timestamp_idx,
            f"{self.mode}/总时间戳数": len(self.timestamps),
        }

    def update_query(self, query):
        """
        更新当前查询
        
        Args:
            query: 新的查询 (sample, direction)
        """
        self.current_query = query
        # 清除与查询相关的缓存
        self._clear_similarity_cache()
        
    def update_candidates(self, samples, indices):
        """
        更新候选样本和索引
        
        Args:
            samples: 候选样本列表
            indices: 候选样本索引列表
        """
        self.current_samples = samples
        self.current_indices = indices
        # 清除与样本相关的缓存
        self._clear_similarity_cache()

    def _clear_similarity_cache(self):
        """清除相似度缓存"""
        # 仅清除相似度缓存，保留语义特征缓存
        keys_to_delete = [k for k in self.feature_cache if k.startswith("sim_")]
        for key in keys_to_delete:
            del self.feature_cache[key] 

    def _extract_sample_text(self, sample, direction):
        """
        从样本中提取文本表示
        
        Args:
            sample: (entity, relation, targets, timestamp) 元组
            direction: "head" 或 "tail" 表示预测方向
            
        Returns:
            样本的文本表示
        """
        entity, relation, _, timestamp = sample
        
        # 根据预测方向决定文本格式
        if direction == "head":
            # 预测头实体
            text = f"{relation}的主体在{timestamp}时刻是什么?"
        else:
            # 预测尾实体
            text = f"{entity}在{timestamp}时刻的{relation}是什么?"
            
        if self.args.no_time:
            # 如果不使用时间
            if direction == "head":
                text = f"{relation}的主体是什么?"
            else:
                text = f"{entity}的{relation}是什么?"
                
        return text
        
    def _extract_semantic_features(self, sample, direction, use_cache=True):
        """
        提取样本的语义特征
        
        Args:
            sample: (entity, relation, targets, timestamp) 元组
            direction: "head" 或 "tail" 表示预测方向
            use_cache: 是否使用缓存
            
        Returns:
            样本的语义特征向量
        """
        # 创建缓存键
        cache_key = f"sem_{sample[0]}_{sample[1]}_{direction}_{sample[3]}"
        
        # 如果特征已缓存，直接返回
        if use_cache and cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
            
        # 提取文本表示
        text = self._extract_sample_text(sample, direction)
        
        # 使用PLM编码器获取语义表示
        embedding = self.encoder.encode(text, use_cache=use_cache)
        
        # 缓存结果
        if use_cache:
            self.semantic_cache[cache_key] = embedding
            
        return embedding
        
    def _calculate_query_sample_similarity(self, query_sample, query_direction, sample, sample_direction, use_cache=True):
        """
        计算查询与样本之间的相似度
        
        Args:
            query_sample: 查询样本 (entity, relation, targets, timestamp)
            query_direction: 查询方向 "head" 或 "tail"
            sample: 候选样本 (entity, relation, targets, timestamp)
            sample_direction: 样本方向 "head" 或 "tail"
            use_cache: 是否使用缓存
            
        Returns:
            查询与样本的余弦相似度
        """
        # 创建缓存键
        cache_key = f"sim_{query_sample[0]}_{query_sample[1]}_{query_direction}_{query_sample[3]}_{sample[0]}_{sample[1]}_{sample_direction}_{sample[3]}"
        
        # 如果相似度已缓存，直接返回
        if use_cache and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        # 提取查询和样本的语义特征
        query_embedding = self._extract_semantic_features(query_sample, query_direction, use_cache=use_cache)
        sample_embedding = self._extract_semantic_features(sample, sample_direction, use_cache=use_cache)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(query_embedding.unsqueeze(0), sample_embedding.unsqueeze(0))[0]
        
        # 缓存计算结果
        if use_cache:
            self.feature_cache[cache_key] = similarity
            
        return similarity
        
    def _calculate_query_samples_similarity(self, query_sample, query_direction, samples, directions, use_cache=True):
        """
        计算查询与多个样本的相似度
        
        Args:
            query_sample: 查询样本
            query_direction: 查询方向
            samples: 样本列表
            directions: 方向列表
            use_cache: 是否使用缓存
            
        Returns:
            相似度列表
        """
        similarities = []
        
        for sample, direction in zip(samples, directions):
            similarity = self._calculate_query_sample_similarity(
                query_sample, query_direction, sample, direction, use_cache=use_cache
            )
            similarities.append(similarity)
            
        return torch.tensor(similarities)

    def set_mode(self, mode: str):
        """
        设置环境模式（训练、验证或测试）
        
        Args:
            mode: 环境模式，可以是 "train"、"val" 或 "test"
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"不支持的环境模式: {mode}，应为 'train'、'val' 或 'test'")
            
        self.mode = mode
        logger.info(f"环境模式已设置为: {mode}")
        
        # 重置环境状态
        self.reset() 