import torch
import os
from torchvision.models import (
    resnet34,
    resnet18,
    resnet50,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
    ResNet34_Weights
)

from pytorch_lightning import LightningModule


# Add 'backbone' attribute to checkpoints
class ClassificationModel(LightningModule):
    def __init__(self, backbone: str):
        super(ClassificationModel, self).__init__()
        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone == "mbv2":
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"No such model {backbone}")

# Save function
def save_full_ckpt(net_name, saved_location, turn_to_backbone=False):
    if os.path.exists(saved_location) == False:
        os.mkdir(saved_location)
    if turn_to_backbone:
        name = net_name + ".ckpt"
        if net_name == "pre_trained_resnet18":
            model = ClassificationModel(backbone='resnet18')
        elif net_name == "pre_trained_resnet34":
            model = ClassificationModel(backbone='resnet34')
        elif net_name == "pre_trained_resnet50":
            model = ClassificationModel(backbone='resnet50')
        elif net_name == "pre_trained_mbv2":
            model = ClassificationModel(backbone='mbv2')
        else:
            raise ValueError(f"No such model {net_name}")
    else:
        name = net_name + "_raw.ckpt"
        if net_name == "pre_trained_resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif net_name == "pre_trained_resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif net_name == "pre_trained_resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif net_name == "pre_trained_mbv2":
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        elif net_name == "resnet18":
            model = resnet18()
        elif net_name == "resnet34":
            model = resnet34()
        elif net_name == "resnet50":
            model = resnet50()
        elif net_name == "mbv2":
            model = mobilenet_v2()
        else:
            raise ValueError(f"No such model {net_name}")

    
    checkpoint_path = os.path.join(saved_location, name)
    
    torch.save({
        'state_dict': model.state_dict(),
    }, checkpoint_path)

    print("Checkpoint is saved at:", checkpoint_path)
