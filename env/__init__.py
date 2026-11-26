"""Airbase interception environment."""
from .interception import InterceptionConfig, InterceptionEnv
from .one_vs_one import OneVsOneConfig, OneVsOneEnv
from .radial_scale import RadialScaleConfig, RadialScaleEnv
from .jia_heading import JiaHeadingConfig, JiaHeadingEnv
from .jia_heading_nearest import JiaHeadingNearestConfig, JiaHeadingNearestEnv


def make_environment(config):
    return (OneVsOneEnv(config) if isinstance(config, OneVsOneConfig) else
            RadialScaleEnv(config) if isinstance(config, RadialScaleConfig) else
            JiaHeadingNearestEnv(config) if isinstance(config, JiaHeadingNearestConfig) else
            JiaHeadingEnv(config) if isinstance(config, JiaHeadingConfig) else InterceptionEnv(config))

__all__=['InterceptionConfig','InterceptionEnv','OneVsOneConfig','OneVsOneEnv',
         'RadialScaleConfig','RadialScaleEnv','JiaHeadingConfig','JiaHeadingEnv',
         'JiaHeadingNearestConfig','JiaHeadingNearestEnv','make_environment']
