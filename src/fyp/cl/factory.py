from .strategies.naive import NaiveFineTuning
from .strategies.er import ExperienceReplay
from .strategies.derpp import DarkExperienceReplayPP
from .strategies.mer import MetaExperienceReplay
from .strategies.si import SynapticIntelligence
from .strategies.lwf import LearningWithoutForgetting
from .strategies.slca import SlowLearnerClassifierAlignment
from .strategies.er_ace import ExperienceReplayACE
from ..config import LEARNING_RATE, DEVICE

def get_cl_strategy(strategy_name, model, device=DEVICE, lr=LEARNING_RATE):
    """Factory function to create CL strategies"""
    strategies = {
        'naive': NaiveFineTuning,
        'er': ExperienceReplay,
        'der++': DarkExperienceReplayPP,
        'mer': MetaExperienceReplay,
        'si': SynapticIntelligence,
        'lwf': LearningWithoutForgetting,
        'slca': SlowLearnerClassifierAlignment,
        'er_ace': ExperienceReplayACE,
    }

    if strategy_name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name.lower()](model, device, lr)
