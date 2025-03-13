from src.rl.agents.dqn_agent import DQNAgent
from src.rl.agents.random_agent import RandomAgent
from src.rl.environment import BaseEnvironment
from src.rl.tkg_environment import TKGEnvironment
from src.rl.plm_encoder import PLMEncoder

ENVIRONMENTS = {
    "tkg": TKGEnvironment,
}

AGENTS = {"dqn": DQNAgent, "random": RandomAgent}
