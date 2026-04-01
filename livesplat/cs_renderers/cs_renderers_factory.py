from cs_renderers.confs import default_cs_renderers_conf, dynamic_cs_renderers_conf, accel_precomp_cs_renderers_conf
from cs_renderers.dynamic_renderer import DynamicRenderer
from cs_renderers.accelerated_renderer import AccceleratedRenderer

def cs_renderer_factory(type, conf=None):
    
    if type == 'accel-precomp':
        return AccceleratedRenderer(conf=conf)
    elif type== 'dynamic':
        return DynamicRenderer(conf=conf)
    else:
        raise NotImplementedError(f'Renderer {type} not implemented')
        
def get_cs_renderer_config(type):
    
    if type== 'dynamic':
        return dynamic_cs_renderers_conf
    elif type == 'accel-precomp':
        return accel_precomp_cs_renderers_conf
    else:
        raise NotImplementedError(f'Conf Renderer {type} not implemented')