"""Airbase interception environment."""
from .interception import InterceptionConfig, InterceptionEnv
from .one_vs_one import OneVsOneConfig, OneVsOneEnv


def make_environment(config):
    return OneVsOneEnv(config) if isinstance(config, OneVsOneConfig) else InterceptionEnv(config)

__all__=['InterceptionConfig','InterceptionEnv','OneVsOneConfig','OneVsOneEnv','make_environment']
