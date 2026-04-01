
class BaseRenderer():
    def __init__(self, conf:dict()):
        self.conf = conf
        
    def render(self, sample:dict(), gaussian_model, pipeline_params) ->dict:
        return None