from post_proc.confs import dynamic_post_proc_conf, default_post_proc_conf
from post_proc._utils import BasePostProc
from post_proc.dynamic_post_proc import DynamicPostProc

def post_proc_factory(type, conf=None):
    
    if type == 'default':
        return BasePostProc(conf=conf)
    elif type == 'dynamic':
        return DynamicPostProc(conf=conf)
    else:
        raise NotImplementedError(f'PostProc {type} not implemented')
        
def get_post_proc_config(type):
    
    if type == 'default':
        return default_post_proc_conf
    elif type == 'dynamic':
        return dynamic_post_proc_conf
    else:
        raise NotImplementedError(f'Conf PostProc {type} not implemented')