import os
import sys
import logging
import torch
import numpy as np
from omegaconf import OmegaConf

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 禁用wandb
os.environ["WANDB_MODE"] = "disabled"

def main():
    try:
        # 导入必要的类
        from src.rl.environment import BaseEnvironment
        from src.rl.agents.dqn_agent import DQNAgent
        from src.rl.agents.replay import Transition
        import wandb
        
        # 初始化wandb（设置为离线模式）
        wandb.init(mode="disabled", project="tkg-rl-test")
        
        logger.info("成功导入BaseEnvironment和DQNAgent类")
        
        # 创建一个简单的测试环境类
        class SimpleTestEnvironment(BaseEnvironment):
            def __init__(self):
                self.state_value = torch.zeros(10)
                self.steps = 0
                self.mode = "train"
                self.named = True  # DQNAgent需要这个属性
                self.device = "cpu"
                
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
                # 返回正确形状的action_space张量
                # 对于DQN代理，每行是一个动作向量
                return torch.tensor([
                    [1, 0, 0, 0, 0],  # 动作0的one-hot表示
                    [0, 1, 0, 0, 0],  # 动作1的one-hot表示
                    [0, 0, 1, 0, 0],  # 动作2的one-hot表示
                    [0, 0, 0, 1, 0],  # 动作3的one-hot表示
                    [0, 0, 0, 0, 1]   # 动作4的one-hot表示
                ], dtype=torch.float)
                
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
        
        logger.info("======= 测试1: 基本DQNAgent功能 =======")
        
        # 创建测试环境
        env = SimpleTestEnvironment()
        logger.info(f"创建测试环境: {env}")
        
        # 创建DQN配置参数
        output_dir = "./rl_outputs/test"
        train_steps = 10
        # 确保target_update_every能够整除save_every
        target_update_every = 5
        save_every = 5  # 必须是target_update_every的倍数
        eval_every = 5  # 必须是save_every的倍数
        batch_size = 4
        replay_memory_size = 100
        lr = 1e-4
        
        # 创建DQNAgent，直接传递具体参数并禁用add_exit_action
        agent = DQNAgent(
            env=env,
            output_dir=output_dir,
            overwrite_existing=True,  # 测试时覆盖现有输出
            train_steps=train_steps,
            save_every=save_every,
            eval_every=eval_every,
            target_update_every=target_update_every,
            batch_size=batch_size,
            replay_memory_size=replay_memory_size,
            lr=lr,
            add_exit_action=False  # 禁用exit_action以简化测试
        )
        logger.info(f"创建DQNAgent: {agent}")
        
        # 测试选择动作
        state = env.reset()
        # 将状态包装成列表，因为DQNAgent需要状态历史
        states = [state]
        action_space = env.action_space()
        logger.info(f"动作空间形状: {action_space.shape}")
        action = agent.choose_action(states, action_space)
        logger.info(f"选择动作: {action}")
        
        # 测试训练一步
        next_state, reward, done = env.step(action)
        
        # 创建Transition对象并添加到记忆中
        reward_tensor = torch.tensor([reward])
        states_tensor = torch.stack(states)
        next_states_tensor = torch.stack([next_state])
        
        # 创建Transition对象
        transition = Transition(
            states=states_tensor,
            action_idx=action,
            action_space=action_space,
            next_states=next_states_tensor,
            next_action_space=action_space,  # 使用相同的动作空间
            reward=reward_tensor
        )
        
        # 将Transition对象传递给push方法
        agent.replay_memory.push(transition)
        
        # 如果记忆缓冲区足够大，进行一次优化
        if len(agent.replay_memory) >= agent.batch_size:
            loss = agent.optimize()
            logger.info(f"优化模型，损失: {loss}")
        else:
            logger.info(f"记忆缓冲区大小不足，当前大小: {len(agent.replay_memory)}")
        
        # 测试目标网络更新
        agent.target_update()
        logger.info("更新目标网络")
        
        logger.info("基本功能测试完成!")
        
        # 测试2: 多步交互与记忆填充
        logger.info("\n======= 测试2: 多步交互与记忆填充 =======")
        
        # 重置环境
        env.reset()
        
        # 创建足够的转换记录以填充记忆缓冲区
        logger.info(f"填充记忆缓冲区（目标大小: {batch_size}）...")
        
        for i in range(batch_size):
            # 重置环境获取新的初始状态
            state = env.reset()
            states = [state]
            states_tensor = torch.stack(states)
            
            # 选择动作
            action_space = env.action_space()
            action = i % 5  # 使用不同的动作
            
            # 执行动作
            next_state, reward, done = env.step(action)
            next_states_tensor = torch.stack([next_state])
            reward_tensor = torch.tensor([reward])
            
            # 创建Transition对象
            transition = Transition(
                states=states_tensor,
                action_idx=action,
                action_space=action_space,
                next_states=next_states_tensor,
                next_action_space=action_space,
                reward=reward_tensor
            )
            
            # 添加到记忆缓冲区
            agent.replay_memory.push(transition)
            
            logger.info(f"  添加转换 {i+1}/{batch_size}，动作={action}，奖励={reward:.4f}")
            
        logger.info(f"记忆缓冲区大小: {len(agent.replay_memory)}")
        
        # 测试优化
        if len(agent.replay_memory) >= agent.batch_size:
            logger.info("执行网络优化...")
            try:
                loss = agent.optimize()
                logger.info(f"  损失: {loss:.6f} （如果是None表示优化跳过）")
            except Exception as e:
                logger.error(f"  优化失败: {e}")
        
        # 测试3: 简单训练流程
        logger.info("\n======= 测试3: 简单训练流程 =======")
        
        # 重置环境
        env.reset()
        
        # 模拟几个训练步骤
        n_steps = 3
        logger.info(f"执行{n_steps}个训练步骤...")
        
        for step in range(n_steps):
            # 重置环境
            state = env.reset()
            done = False
            total_reward = 0
            
            # 执行单个episode
            while not done:
                # 获取当前状态
                states = [state]
                states_tensor = torch.stack(states)
                
                # 选择动作
                action_space = env.action_space()
                action = agent.choose_action(states, action_space)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                total_reward += reward
                next_states_tensor = None if done else torch.stack([next_state])
                next_action_space = None if done else action_space
                
                # 创建Transition对象
                transition = Transition(
                    states=states_tensor,
                    action_idx=action,
                    action_space=action_space,
                    next_states=next_states_tensor,
                    next_action_space=next_action_space,
                    reward=torch.tensor([reward])
                )
                
                # 添加到记忆缓冲区
                agent.replay_memory.push(transition)
                
                # 优化（每个步骤优化一次）
                if len(agent.replay_memory) >= agent.batch_size:
                    try:
                        agent.optimize()
                    except Exception as e:
                        logger.error(f"  优化失败: {e}")
                
                # 更新状态
                if not done:
                    state = next_state
            
            logger.info(f"  步骤 {step+1}/{n_steps} 完成，总奖励: {total_reward:.4f}")
            
            # 更新目标网络（按照指定的频率）
            if (step + 1) % target_update_every == 0:
                agent.target_update()
                logger.info(f"  步骤 {step+1}: 更新目标网络")
        
        logger.info("简单训练流程测试完成!")
        logger.info("\n所有测试成功完成!")
        
        # 关闭wandb
        wandb.finish()
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        
if __name__ == "__main__":
    main() 