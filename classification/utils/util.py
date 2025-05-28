import torch.nn as nn
import torch
def get_all_linear_with_name(model):
    linear_layers = {}

    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.linear.Linear) and 'mlp' in name:
            linear_layers[name] = mod

    return linear_layers

def get_active_linear_with_name(model):
    total_linear_layer = get_all_linear_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_linear_layer):
        return total_linear_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1
    else:
        active_linear_layers = dict(list(total_linear_layer.items())[-model.num_of_finetune:])
        return active_linear_layers

def get_all_conv_with_name(model):
    conv_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d):
            conv_layers[name] = mod
    return conv_layers

def get_all_conv_with_name_and_previous_non_linearity(model, parent_name='', previous_mod=None, previous_name=None):
    name_conv_layers_with_relu = [] # List of convolutional layer name that is connected with relu
    conv_layers = []

    for name, mod in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if len(list(mod.children())) > 0:
            result, name_with_relu, previous_mod, previous_name = get_all_conv_with_name_and_previous_non_linearity(mod, full_name, previous_mod, previous_name)
            name_conv_layers_with_relu.extend(name_with_relu)
            conv_layers.extend(result)
        else:
            if isinstance(mod, (nn.Conv2d)):
                conv_layers.append(full_name)
                if isinstance(previous_mod, (nn.ReLU, nn.ReLU6)):
                    name_conv_layers_with_relu.append(f"{full_name}_relu")
                else:
                    name_conv_layers_with_relu.append(full_name)
        
            previous_mod = mod
            previous_name = name

    return conv_layers, name_conv_layers_with_relu, previous_mod, previous_name
    
def get_active_conv_with_name(model):
    total_conv_layer = get_all_conv_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_conv_layer):
        return total_conv_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1
    else:
        active_conv_layers = dict(list(total_conv_layer.items())[-model.num_of_finetune:])
        return active_conv_layers

class Hook:
    def __init__(self, module):
        self.module = module
        self.input_size = None
        self.output_size = None
        self.inputs = []
        self.outputs = []

        self.weight_size = None
        self.weight = None
        
        self.active = True
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        if not self.active:
            return
        Input = input[0].clone().detach()
        Output = output.clone().detach()

        self.input_size = torch.tensor(Input.shape)
        self.output_size = torch.tensor(Output.shape)

        self.inputs.append(Input)
        self.outputs.append(Output)

        # Module bình thường, không phải LoRA
        if hasattr(module, 'weight') and module.weight is not None:
            self.weight_size = module.weight.shape
            self.weight = module.weight.clone().detach()


    def activate(self, active):
        self.active = active

    def remove(self):
        self.input_size = None
        self.output_size = None
        self.inputs.clear()
        self.outputs.clear()
        self.weight_size = None
        self.weight =  None

        self.active = False
        self.hook.remove()
        # print("Hook is removed")

def calculate_flops_SVD(size_1, size_2):
    if isinstance(size_1, torch.Tensor):
        m = torch.max(size_1, size_2)
        n = torch.min(size_1, size_2)
    else:
        m = max(size_1, size_2)
        n = min(size_1, size_2)
    return m*n**2
