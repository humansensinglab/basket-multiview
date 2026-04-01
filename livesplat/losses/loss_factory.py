from losses.confs import default_loss_conf, dynamic_loss_conf, static_opt_loss_conf
from losses.default_loss import DefaultLoss
from losses.dynamic_loss import DynamicLoss
from losses.static_opt_loss import StaticOptLoss

def loss_factory(type, conf=None):
    
    if type == 'default':
        return DefaultLoss(conf=conf)
    elif type == 'dynamic':
        return DynamicLoss(conf=conf)
    elif type == 'static_opt':
        return StaticOptLoss(conf=conf)
    else:
        raise NotImplementedError(f'Loss {type} not implemented')
        
def get_loss_config(type):
    
    if type == 'default':
        return default_loss_conf
    elif type == 'dynamic':
        return dynamic_loss_conf
    elif type == 'static_opt':
        return static_opt_loss_conf
    else:
        raise NotImplementedError(f'Conf Loss {type} not implemented')