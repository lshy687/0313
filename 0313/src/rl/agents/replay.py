import logging
import random
from collections import deque
from typing import Union

import torch

from src.rl.environment import BaseEnvironment

logger = logging.getLogger(__name__)


class Transition:
    def __init__(
        self, states, action_idx, action_space, next_states, next_action_space, reward
    ):
        self.states = states
        self.action_idx = action_idx
        self.action_space = action_space
        self.next_states = next_states
        self.next_action_space = next_action_space
        self.reward = reward


class NamedTransition(Transition):
    def __init__(self, *args):
        super(NamedTransition, self).__init__(*args)


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, t: Transition):
        if t.action_space.shape[0] == 1 or (
            t.next_action_space is not None and t.next_action_space.shape[0] == 1
        ):
            logger.info("skip push")
            return
        self.memory.append(t)

    def sample(self, k=1):
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)

    def load(
        self,
        path: Union[str, list[str]],
        env: BaseEnvironment,
        add_exit_action: bool,
        exit_action_prob: float = 0.05,
    ):
        """
        从文件加载转换
        """
        if not path:
            return

        assert isinstance(env, BaseEnvironment)
        logger.info(f"loading transitions from path {path} with env {env}")
        transitions = torch.load(path)
        
        # 处理转换记录
        if isinstance(transitions, list):
            for t in transitions:
                try:
                    self.push(t)
                except Exception as e:
                    logger.warning(f"Error pushing transition: {e}")
            
            logger.info(f"Loaded {len(transitions)} transitions")
            return
            
        # 处理具有任务信息的转换词典
        for task, task_trans in transitions.items():
            for t in task_trans:
                try:
                    self.push(t)
                except Exception as e:
                    logger.warning(f"Error pushing transition from task {task}: {e}")
        
        logger.info(f"Loaded transitions from {len(transitions)} tasks")
