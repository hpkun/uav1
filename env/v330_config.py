"""4v4 identity mapped back to the strictly frozen v3.29 configuration."""
from copy import deepcopy
from .v329_config import validate_v329


def validate_v330(config):
    if not isinstance(config,dict) or config.get('environment_version')!='3.30':
        raise ValueError('v3.30 requires version 3.30')
    scenario=config.get('scenario',{})
    if type(scenario.get('team_size')) is not int or scenario['team_size']!=4:
        raise ValueError('v3.30 requires team_size=4')
    if scenario.get('formation_offsets')!=[-300.,-100.,100.,300.]:
        raise ValueError('v3.30 requires symmetric four-aircraft offsets')
    baseline=deepcopy(config);baseline['environment_version']='3.29'
    baseline['scenario']['team_size']=5
    baseline['scenario']['formation_offsets']=[-400.,-200.,0.,200.,400.]
    validate_v329(baseline)
    return config
