from dataclasses import dataclass, field
from typing import List

@dataclass 
class OptParamConf:
    name: str
    lr: float

@dataclass
class FrameOptimizerConf:
    iterations: int
    range: List[int]
    parameters_to_optimize: List[OptParamConf]



@dataclass
class LrBaseConf:
    feature_lr:float
    opacity_lr:float
    scaling_lr:float
    rotation_lr:float

@dataclass
class SchedulerConf:
    position_lr_init:float
    position_lr_final:float
    position_lr_delay_mult:float
    
@dataclass
class OptimizerConf:
    # lr_base_confs = {
    #     'static': LrBaseConf(
    #         feature_lr = 0.0025 * 1,
    #         opacity_lr = 0.05 * 1,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    #     'dynamic': LrBaseConf(
    #         feature_lr = 0.0025 * 2.5,
    #         opacity_lr = 0.05 * 0.01,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    # }
    # scheduler_confs = {
    #     'static': SchedulerConf(
    #         position_lr_init = 0.00016 * 1,
    #         position_lr_final = 0.00000016 * 1,
    #         position_lr_delay_mult = 0.01,
    #     ),
    #     'dynamic': SchedulerConf(
    #         position_lr_init = 0.00016 * 12.0,
    #         position_lr_final = 0.00000016 * 8.0,
    #         position_lr_delay_mult = 0.01,
    #     ),
    # }
    
    
    # lr_base_confs = {
    #     'static': LrBaseConf(
    #         feature_lr = 0.0025 * 1,
    #         opacity_lr = 0.05 * 1,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    #     'dynamic': LrBaseConf(
    #         feature_lr = 0.0025 * 1,
    #         opacity_lr = 0.05 * 1,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    # }
    # scheduler_confs = {
    #     'static': SchedulerConf(
    #         position_lr_init = 0.00016 * 1,
    #         position_lr_final = 0.00000016 * 1,
    #         position_lr_delay_mult = 0.01,
    #     ),
    #     'dynamic': SchedulerConf(
    #         position_lr_init = 0.00016 * 1.0,
    #         position_lr_final = 0.00000016 * 1.0,
    #         position_lr_delay_mult = 0.01,
    #     ),
    # }
    
    ## stadium player
    lr_base_confs = {
        'static': LrBaseConf(
            feature_lr = 0.0025 * 1,
            opacity_lr = 0.05 * 1,
            scaling_lr = 0.005 * 1,
            rotation_lr = 0.001 * 1,
        ),
        'dynamic': LrBaseConf(
            feature_lr = 0.0025 * 8,
            opacity_lr = 0.05 * 0.2,
            scaling_lr = 0.005 * 1,
            rotation_lr = 0.001 * 1,
        ),
    }
    scheduler_confs = {
        'static': SchedulerConf(
            position_lr_init = 0.00016 * 1,
            position_lr_final = 0.00000016 * 1,
            position_lr_delay_mult = 0.01,
        ),
        'dynamic': SchedulerConf(
            position_lr_init = 0.00016 * 0.2,
            position_lr_final = 0.00000016 * 0.2,
            position_lr_delay_mult = 0.01,
        ),
    }
    
    ## Black bg
    # lr_base_confs = {
    #     'static': LrBaseConf(
    #         feature_lr = 0.0025 * 1,
    #         opacity_lr = 0.05 * 1,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    #     'dynamic': LrBaseConf(
    #         feature_lr = 0.0025 * 7,
    #         opacity_lr = 0.05 * 0.5,
    #         scaling_lr = 0.005 * 1,
    #         rotation_lr = 0.001 * 1,
    #     ),
    # }
    # scheduler_confs = {
    #     'static': SchedulerConf(
    #         position_lr_init = 0.00016 * 1,
    #         position_lr_final = 0.00000016 * 1,
    #         position_lr_delay_mult = 0.01,
    #     ),
    #     'dynamic': SchedulerConf(
    #         position_lr_init = 0.00016 * 1.0,
    #         position_lr_final = 0.00000016 * 1.0,
    #         position_lr_delay_mult = 0.01,
    #     ),
    # }
    
    # lr_base_confs = {
    #     'static': LrBaseConf(
    #         feature_lr = 0.0025 * 2,
    #         opacity_lr = 0.05 * 2,
    #         scaling_lr = 0.005 * 2,
    #         rotation_lr = 0.001 * 2,
    #     ),
    #     'dynamic': LrBaseConf(
    #         feature_lr = 0.0025 * 1,
    #         opacity_lr = 0.05 * 1,
    #         scaling_lr = 0.005 * 3,
    #         rotation_lr = 0.001 * 3,
    #     ),
    # }
    

    # scheduler_confs = {
    #     'static': SchedulerConf(
    #         position_lr_init = 0.00016 * 0.5,
    #         position_lr_final = 0.00000016 * 0.5,
    #         position_lr_delay_mult = 0.01,
    #     ),
    #     'dynamic': SchedulerConf(
    #         position_lr_init = 0.00016 * 2.0,
    #         position_lr_final = 0.00000016 * 2.0,
    #         position_lr_delay_mult = 0.01,
    #     ),
    # }
    
    opt_confs = {
        'static': FrameOptimizerConf( # train from scratch
            # [start, end)
            range=[0, 1], 
            iterations = 30000,
            parameters_to_optimize=[
                OptParamConf(name='xyz', lr=scheduler_confs['static'].position_lr_init),
                OptParamConf(name='f_dc', lr=lr_base_confs['static'].feature_lr),
                OptParamConf(name='f_rest', lr=lr_base_confs['static'].feature_lr / 20.0),
                OptParamConf(name='opacity', lr=lr_base_confs['static'].opacity_lr),
                OptParamConf(name='scaling', lr=lr_base_confs['static'].scaling_lr),
                OptParamConf(name='rotation', lr=lr_base_confs['static'].rotation_lr),
            ]
        ),
        'dynamic': FrameOptimizerConf( # train from previous frame
            # [start, end)
            range=[1, -1],
            iterations = 500,
            parameters_to_optimize=[
                OptParamConf(name='xyz', lr=scheduler_confs['dynamic'].position_lr_init),
                OptParamConf(name='f_dc', lr=lr_base_confs['dynamic'].feature_lr),
                OptParamConf(name='f_rest', lr=lr_base_confs['dynamic'].feature_lr / 20.0),
                OptParamConf(name='opacity', lr=lr_base_confs['dynamic'].opacity_lr),
                OptParamConf(name='scaling', lr=lr_base_confs['dynamic'].scaling_lr),
                OptParamConf(name='rotation', lr=lr_base_confs['dynamic'].rotation_lr),
            ]
        ),

        'static_opt': FrameOptimizerConf(
                # [start, end)
                range=[1, -1],
                iterations=600,
                parameters_to_optimize=[
                    OptParamConf(name="f_dc", lr=lr_base_confs['dynamic'].feature_lr),
                    OptParamConf(name="f_rest", lr=lr_base_confs['dynamic'].feature_lr / 20.0),
                ],
            )
    }
    
    def get_lr_base_conf(self, mode):
        return self.lr_base_confs[mode]
    
    def get_scheduler_conf(self, mode):
        return self.scheduler_confs[mode]

    def get_opt_conf(self, mode):
        return self.opt_confs[mode]
    
    def get_opt_params(self, frame_idx):
        if(frame_idx==0):
            return self.opt_confs['static']
        else:
            return self.opt_confs['dynamic']
        
    
    
    

