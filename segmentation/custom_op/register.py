import logging
from functools import reduce
import torch.nn as nn
from .conv2d.conv_avg import wrap_conv_layer
from .conv2d.conv_hosvd import wrap_convHOSVD
from tools.utils import attach_hooks_for_conv
from .conv2d.conv_ASI import wrap_convASI


from .conv2d.conv_measure_perplexity_HOSVD import wrap_conv_measure_perplexity_HOSVD

DEFAULT_CFG = {
    "path": "",
    "radius": 8,
    "type": "",
    "SVD_var": 0.8,
}


def add_grad_filter(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_layer(module.conv, cfg['radius'], True)

        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook, special_param=cfg['radius'])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_layer(module.conv1, cfg['radius'], True)
        module.conv2 = wrap_conv_layer(module.conv2, cfg['radius'], True)

        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook, special_param=cfg['radius'])
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook, special_param=cfg['radius'])
    elif cfg['type'] == 'conv':
        module = wrap_conv_layer(module, cfg['radius'], True)

        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook, special_param=cfg['radius'])
    else:
        raise NotImplementedError
    return module

def add_hosvd_filter(module: nn.Module, cfg):
    if 'k_hosvd' in cfg:
        if cfg['type'] == 'cbr':
            module.conv = wrap_convHOSVD(module.conv, cfg['SVD_var'], True, cfg["k_hosvd"])
        elif cfg['type'] == 'resnet_basic_block':
            module.conv1 = wrap_convHOSVD(module.conv1, cfg['SVD_var'], True, cfg["k_hosvd"])
            module.conv2 = wrap_convHOSVD(module.conv2, cfg['SVD_var'], True, cfg["k_hosvd"])
        elif cfg['type'] == 'conv':
            module = wrap_convHOSVD(module, cfg['SVD_var'], True, cfg["k_hosvd"])
        else:
            raise NotImplementedError
    else:
        if cfg['type'] == 'cbr':
            module.conv = wrap_convHOSVD(module.conv, cfg['SVD_var'], True)
        elif cfg['type'] == 'resnet_basic_block':
            module.conv1 = wrap_convHOSVD(module.conv1, cfg['SVD_var'], True)
            module.conv2 = wrap_convHOSVD(module.conv2, cfg['SVD_var'], True)
        elif cfg['type'] == 'conv':
            module = wrap_convHOSVD(module, cfg['SVD_var'], True)
        else:
            raise NotImplementedError

    return module

def add_hook_for_base_conv(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook)
    elif cfg['type'] == 'resnet_basic_block':
        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook)
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook)
    elif cfg['type'] == 'conv':
        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook)
    else:
        raise NotImplementedError
    return module

def add_measure_filter(module: nn.Module, cfg, layer_idx, cfgs):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_measure_perplexity_HOSVD(module.conv, True, cfgs["SVD_var"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        cfgs["layer_name"][layer_idx] = cfg['path'] + ".conv"
        layer_idx += 1

    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_measure_perplexity_HOSVD(module.conv1, True, cfgs["SVD_var"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        cfgs["layer_name"][layer_idx] = cfg['path'] + ".conv1"
        layer_idx += 1
        module.conv2 = wrap_conv_measure_perplexity_HOSVD(module.conv2, True, cfgs["SVD_var"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        cfgs["layer_name"][layer_idx] = cfg['path'] + ".conv2"
        layer_idx += 1

    elif cfg['type'] == 'conv':
        module = wrap_conv_measure_perplexity_HOSVD(module, True, cfgs["SVD_var"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        cfgs["layer_name"][layer_idx] = cfg['path'] + ".conv"
        layer_idx += 1

    else:
        raise NotImplementedError
    return module, layer_idx

def add_ASI(module: nn.Module, cfg, cfgs, hook):
    # xác định layer_idx dựa trên tên layer hiện tại, xem với tên này thì nó ứng với index nào trong perplexity.layername
    if cfg['type'] == 'cbr':
        layer_name = cfg['path'] + ".conv"
        layer_idx = cfgs["layer_names"].index(layer_name)
        module.conv = wrap_convASI(module.conv, True, cfgs["rank"][layer_idx])

        attach_hooks_for_conv(module=module.conv, name=layer_name, hook=hook, special_param=cfgs["rank"][layer_idx])

    elif cfg['type'] == 'resnet_basic_block':
        layer_name = cfg['path'] + ".conv1"
        layer_idx = cfgs["layer_names"].index(layer_name)
        module.conv1 = wrap_convASI(module.conv1, True, cfgs["rank"][layer_idx])

        attach_hooks_for_conv(module=module.conv1, name=layer_name, hook=hook, special_param=cfgs["rank"][layer_idx])

        layer_name = cfg['path'] + ".conv2"
        layer_idx = cfgs["layer_names"].index(layer_name)
        module.conv2 = wrap_convASI(module.conv2, True, cfgs["rank"][layer_idx])

        attach_hooks_for_conv(module=module.conv2, name=layer_name, hook=hook, special_param=cfgs["rank"][layer_idx])

    elif cfg['type'] == 'conv':
        layer_name = cfg['path'] + ".conv"
        layer_idx = cfgs["layer_names"].index(layer_name)
        module = wrap_convASI(module, True, cfgs["rank"][layer_idx])

        attach_hooks_for_conv(module=module, name=layer_name, hook=hook, special_param=cfgs["rank"][layer_idx])

    else:
        raise NotImplementedError
    return module

###############################################################################
def register_filter(module, cfgs, hook=None):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_grad_filter(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_filter(module, cfgs):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        if 'k_hosvd' in cfgs:
            cfg['k_hosvd'] = cfgs['k_hosvd']
        else:
            cfg['k_hosvd'] = None

        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_hosvd_filter(target, cfg)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_SVD_filter(module, cfgs):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        if 'svd_size' in cfgs:
            cfg['svd_size'] = cfgs['svd_size']
        else:
            cfg['svd_size'] = None
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_svd_filter(target, cfg)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)



def attach_hook_for_base_conv(module, cfgs, hook=None):
    filter_install_cfgs = cfgs['filter_install']
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        # for k in cfg.keys():
        #     assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_hook_for_base_conv(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)


def register_measure_perplexity_HOSVD(module, cfgs):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering measure_perplexity HOSVD convolution")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    layer_idx = 0
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer, layer_idx = add_measure_filter(target, cfg, layer_idx, cfgs)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_power4_budget_filter(module, cfgs, hook=None):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering HOSVD 4 with budget filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_ASI(target, cfg, cfgs, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)
#################################################################################
            