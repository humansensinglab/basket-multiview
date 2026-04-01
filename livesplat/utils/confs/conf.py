from dataclasses import dataclass, field
from utils.confs.densification import DensificationParams
from utils.confs.optimizer import OptimizerConf
from utils.confs.rendering import RenderingParams

@dataclass
class DynamicTraningConf:
    rendering:RenderingParams = field(default_factory=RenderingParams)
    optimization:OptimizerConf = field(default_factory=OptimizerConf)
    densification:DensificationParams = field(default_factory=DensificationParams)