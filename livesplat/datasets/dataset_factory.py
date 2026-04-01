import datasets.basket_mv_dataset as basketmv_dataset

DATASET_CONF = {
    'data_path': None, 
    'timestamp': None, 
    'spatial_scale_factor': 1 / 100.0,
    'img_scale_factor': 0.5,
    'verbose': False,
    'require_all_img_data_available':True,
    'first_k_cams': -1,
    'load_depth': True,
}

class BaseDataset():
    def __init__(self, conf:dict):
        self.conf = conf

def dataset_factory(type_dataset, conf=DATASET_CONF):
    
    if type_dataset == 'basket_mv':
        return basketmv_dataset.BasketMVCamDataset(conf=conf)
    else:
        raise NotImplementedError(f'Dataset {type} not implemented')