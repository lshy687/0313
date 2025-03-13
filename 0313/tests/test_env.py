#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的TKG环境测试脚本
"""

import os
import sys
import logging
import torch
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    try:
        # 导入BaseEnvironment类
        from src.rl.environment import BaseEnvironment
        
        logger.info("成功导入BaseEnvironment类")
        
        # 创建一个简单的测试环境类
        class SimpleTestEnvironment(BaseEnvironment):
            def __init__(self):
                self.state_value = torch.zeros(10)
                self.steps = 0
                self.mode = "train"
                
            def reset(self):
                self.state_value = torch.zeros(10)
                self.steps = 0
                return self.state_value
                
            @property
            def state(self):
                return self.state_value
                
            @property
            def state_dim(self):
                return 10
                
            @property
            def action_dim(self):
                return 5
                
            def action_count(self):
                return 5
                
            def action_space(self):
                return list(range(5))
                
            def step(self, idx):
                self.state_value[idx] += 1.0
                self.steps += 1
                reward = float(idx) / 5.0
                done = self.steps >= 3
                return self.state_value, reward, done
                
            def summary(self):
                return {
                    "steps": self.steps,
                    "state_sum": self.state_value.sum().item(),
                    "mode": self.mode
                }
        
        # 测试环境
        env = SimpleTestEnvironment()
        logger.info(f"创建测试环境: {env}")
        
        # 测试重置
        state = env.reset()
        logger.info(f"重置环境，初始状态: {state}")
        
        # 测试动作
        for i in range(3):
            action = i % env.action_count()
            next_state, reward, done = env.step(action)
            logger.info(f"执行动作 {action}, 获得奖励 {reward}, 下一状态 {next_state}, 是否结束 {done}")
            
            if done:
                break
                
        # 测试摘要
        summary = env.summary()
        logger.info(f"环境摘要: {summary}")
        
        # 测试模式设置
        env.set_mode("eval")
        logger.info(f"设置模式为eval，当前模式: {env.mode}")
        
        logger.info("测试成功完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        
if __name__ == "__main__":
    main() 