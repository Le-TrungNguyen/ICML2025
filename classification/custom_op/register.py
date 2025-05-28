import logging
from functools import reduce

from .conv2d.conv_avg import wrap_conv_avg_layer

from .conv2d.conv_hosvd_var import wrap_convHOSVD_var
from .linear.linear_hosvd_var import wrap_linearHOSVD_var

from .conv2d.conv_ASI import wrap_convASI
from .linear.linear_ASI import wrap_linearASI



from .conv2d.conv_normal import wrap_conv

from .conv2d.conv_measure_perplexity_HOSVD import wrap_conv_measure_perplexity_HOSVD
from .linear.linear_measure_perplexity_HOSVD import wrap_linear_measure_perplexity_HOSVD




def register_grad_filter(module, cfgs):
    logging.info("Registering Gradient Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False
            
        upd_layer = wrap_conv_avg_layer(target, cfgs['radius'], True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_var_filter(module, cfgs):
    logging.info("Registering HOSVD compression")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if cfgs["type"] == "conv":
            upd_layer = wrap_convHOSVD_var(target, True, cfgs["explained_variance_threshold"], cfgs["k_hosvd"])
        elif cfgs["type"] == "linear":
            upd_layer = wrap_linearHOSVD_var(target, True, cfgs["explained_variance_threshold"], cfgs["k_hosvd"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_ASI(module, cfgs):
    logging.info("Registering HOSVD 4 with budget filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if cfgs["type"] == "conv":
            upd_layer = wrap_convASI(target, True, cfgs["truncation_threshold"][layer_idx], no_reuse=cfgs["no_reuse"])
        elif cfgs["type"] == "linear":
            upd_layer = wrap_linearASI(target, True, cfgs["truncation_threshold"][layer_idx], no_reuse=cfgs["no_reuse"])


        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_normal_conv(module, cfgs):
    logging.info("Registering normal convolution")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if cfgs["type"] == "conv":
            upd_layer = wrap_conv(target, True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_measure_perplexity_HOSVD(module, cfgs):
    logging.info("Registering measure_perplexity HOSVD convolution")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')

        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if cfgs["type"] == "conv":
            upd_layer = wrap_conv_measure_perplexity_HOSVD(target, True, cfgs["explain_variance_threshold"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        
        elif cfgs["type"] == "linear":
            upd_layer = wrap_linear_measure_perplexity_HOSVD(target, True, cfgs["explain_variance_threshold"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)


        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)