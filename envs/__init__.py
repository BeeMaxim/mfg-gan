from envs.bottleneck_cylinder import BottleneckCylinderEnv
from envs.twodiag_cylinder import TwoDiagCylinderEnv
from envs.quadcopter import QuadcopterEnv
from envs.seir_hcd import SEIRHCD_Env

# Environment dictionary
env_dict = {'BottleneckCylinderEnv': BottleneckCylinderEnv,
            'TwoDiagCylinderEnv': TwoDiagCylinderEnv,
            'QuadcopterEnv': QuadcopterEnv,
            'SEIR-HCD': SEIRHCD_Env}