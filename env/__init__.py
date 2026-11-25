"""Airbase interception environment."""
from .interception import InterceptionConfig, InterceptionEnv
from .one_vs_one import OneVsOneConfig, OneVsOneEnv
from .radial_scale import RadialScaleConfig, RadialScaleEnv
from .jia_heading import JiaHeadingConfig, JiaHeadingEnv


def make_environment(config):
    return (OneVsOneEnv(config) if isinstance(config, OneVsOneConfig) else
            RadialScaleEnv(config) if isinstance(config, RadialScaleConfig) else
            JiaHeadingEnv(config) if isinstance(config, JiaHeadingConfig) else InterceptionEnv(config))

__all__=['InterceptionConfig','InterceptionEnv','OneVsOneConfig','OneVsOneEnv',
         'RadialScaleConfig','RadialScaleEnv','JiaHeadingConfig','JiaHeadingEnv','make_environment']
