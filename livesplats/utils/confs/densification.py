from dataclasses import dataclass

@dataclass
class DensificationParams:
    percent_dense:float = 0.01
    densification_interval:int = 100
    opacity_reset_interval:int = 3000
    densify_from_iter:int = 400
    densify_until_iter: int = 6000
    densify_grad_threshold:float = 0.0002
    max_num_points: int = 10000
