
default_loss_conf = {
    'type': 'default',
    'conf': {
        'lambda_dssim': 0.2,
        'stretch_thres': 100,
        'stretch_loss_weight': 0.0
    }
}

dynamic_loss_conf = {
    'type': 'dynamic',
    'conf': {
        'lambda_dssim': 0.2,
        'stretch_thres': 100,
        'stretch_loss_weight': 0.0
    }
}

static_opt_loss_conf = {
    'type': 'static_opt',
    'conf': {
        'lambda_dssim': 0.2,
        'stretch_thres': 100,
        'stretch_loss_weight': 0.0
    }
}