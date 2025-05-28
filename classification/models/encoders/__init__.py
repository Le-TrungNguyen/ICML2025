import functools
import torch.utils.model_zoo as model_zoo

import torch
from .resnet import resnet_encoders
from .mobilenet import mobilenet_encoders
from .mcunet import mcunet_encoders

from ._preprocessing import preprocess_input
import os
from .prepare_ckpt import save_full_ckpt
from torchvision.models import swin_t, Swin_T_Weights, vit_b_32, ViT_B_32_Weights

encoders = {}
encoders.update(resnet_encoders)
encoders.update(mobilenet_encoders)
encoders.update(mcunet_encoders)


def get_encoder(name, setup, checkpoint=None, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    if name == "swinT":
        if weights == "full_imagenet":
            return swin_t(weights=Swin_T_Weights.DEFAULT)
        elif weights == "raw":
            return swin_t()
    if name == "vit_b_32":
        if checkpoint is not None:
            pruned_dict = torch.load(checkpoint, weights_only=False, map_location='cpu')
            model = pruned_dict['model']
            return model
        elif weights == "full_imagenet":
            return vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        elif weights == "raw":
            return vit_b_32(weights=None)
    
    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(
            name, list(encoders.keys())))
    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(log_grad='log_grad' in kwargs and kwargs['log_grad'])
    encoder = Encoder(**params)

    if setup == "B" and name != "mcunet":
        weights = kwargs["weights_setupB"]
        if weights is not None:
            try:
                settings = encoders[name]["pretrained_settings"][weights]
            except KeyError:
                raise KeyError(
                    "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                        weights,
                        name,
                        list(encoders[name]["pretrained_settings"].keys()),
                    )
                )
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))    
    elif setup == "A" and name != "mcunet":
        weights = kwargs["weights_setupA"]
        saved_location = "./pretrained_ckpts/"
        if weights == "full_imagenet": # Setup A
            if name == "resnet18":
                checkpoint = "pre_trained_resnet18_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("pre_trained_resnet18", saved_location, False)
            elif name == "resnet34":
                checkpoint = "pre_trained_resnet34_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("pre_trained_resnet34", saved_location, False)
            elif name == "mobilenet_v2":
                checkpoint = "pre_trained_mbv2_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("pre_trained_mbv2", saved_location, False)
            model_state_dict = torch.load(os.path.join(saved_location, checkpoint))['state_dict']
            encoder.load_state_dict(model_state_dict)
        elif weights == "raw": # Raw checkpoint
            if name == "resnet18":
                checkpoint = "resnet18_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("resnet18", saved_location, False)
            elif name == "resnet34":
                checkpoint = "resnet34_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("resnet34", saved_location, False)
            elif name == "mobilenet_v2":
                checkpoint = "mbv2_raw.ckpt"
                if not os.path.exists(os.path.join(saved_location, checkpoint)):
                    save_full_ckpt("mbv2", saved_location, False)
            model_state_dict = torch.load(os.path.join(saved_location, checkpoint))['state_dict']
            encoder.load_state_dict(model_state_dict)
        
    if "mcunet" in name:
        assert "pretrained" in kwargs, "[Warning] pretrained condition is not defined for mcunet"
        encoder.set_in_channels(in_channels, pretrained=kwargs["pretrained"])
    else:
        encoder.set_in_channels(in_channels, pretrained=weights is not None)
    
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):

    all_settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in all_settings.keys():
        raise ValueError(
            "Available pretrained options {}".format(all_settings.keys()))
    settings = all_settings[pretrained]

    formatted_settings = {}
    formatted_settings["input_space"] = settings.get("input_space", "RGB")
    formatted_settings["input_range"] = list(
        settings.get("input_range", [0, 1]))
    formatted_settings["mean"] = list(settings.get("mean"))
    formatted_settings["std"] = list(settings.get("std"))

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
