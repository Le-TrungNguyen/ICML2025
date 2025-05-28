import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_grad_filter, register_normal_conv, register_ASI, register_measure_perplexity_HOSVD, register_HOSVD_var_filter

from custom_op.conv2d.conv_avg import Conv2dAvg
from custom_op.conv2d.conv_ASI import Conv2d_ASI


from tqdm import tqdm

from utils.util import get_all_conv_with_name, Hook, get_all_conv_with_name_and_previous_non_linearity, get_active_conv_with_name
from math import ceil
from models.encoders import get_encoder
from functools import reduce
import logging

from utils.perplexity import Perplexity


import inspect

# import time

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes,
                 learning_rate, weight_decay, set_bn_eval, load = None,

                num_of_finetune=None, 
                ##### which kind of filter will be used
                with_HOSVD_var=False, with_grad_filter=False, with_ASI=False, force_use_base = False, measure_perplexity_HOSVD_var=False,

                no_reuse = False, truncation_threshold=None, filt_radius=None, budget = None, perplexity_pkl=None,

                just_log = False, # only log activation size, flops ... no training

                use_sgd=False, momentum=0.9, anneling_steps=8008, scheduler_interval='step',
                lr_warmup=0, init_lr_prod=0.25, 
                setup=None):
       
        super(ClassificationModel, self).__init__()

        # Automatically capture all init arguments and their values
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.initial_state = {arg: values[arg] for arg in args if arg != 'self'}
        ############################################################

        self.setup = setup
        if self.setup not in ["A", "B"]:
            raise ValueError(f"Invalid setup value: {self.setup}. It must be 'A' or 'B'.")

        self.backbone_name = backbone
        self.backbone = get_encoder(backbone, self.setup, **backbone_args)
  
        if load != None and (measure_perplexity_HOSVD_var == True):
            state_dict = th.load(load)['state_dict']
            self.load_state_dict(state_dict, strict=False)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone._out_channels[-1], num_classes)
        ##    
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)

        ##
        self.all_conv_layers, self.name_conv_layers_with_relu, _, _ = get_all_conv_with_name_and_previous_non_linearity(self) # A dictionary contains all conv2d layers (value) and their names (key)
        if num_of_finetune == "all" or num_of_finetune > len(self.all_conv_layers):
            logging.info("[Warning] Finetuning all layers")
            self.num_of_finetune = len(self.all_conv_layers)
        else:
            self.num_of_finetune = num_of_finetune

        self.just_log = just_log
        self.measure_perplexity_HOSVD_var = measure_perplexity_HOSVD_var
        self.force_use_base = force_use_base
        self.with_ASI = with_ASI
        self.with_HOSVD_var = with_HOSVD_var
        self.with_grad_filter = with_grad_filter
        self.with_base = not (self.with_HOSVD_var or self.with_grad_filter or self.with_ASI)

        self.no_reuse = no_reuse
        self.truncation_threshold = truncation_threshold
        self.filt_radius = filt_radius

        if self.measure_perplexity_HOSVD_var:
            self.perplexity     = [None for layer_idx in range(len(self.all_conv_layers))]
            self.measured_rank  = [None for layer_idx in range(len(self.all_conv_layers))]
            self.layer_mem      = [None for layer_idx in range(len(self.all_conv_layers))]


        if self.with_ASI:
            self.perplexity_pkl = perplexity_pkl
            perplexity = Perplexity()
            perplexity.load(self.perplexity_pkl)
            best_memory, best_perplexity, best_indices, self.suitable_ranks = perplexity.find_best_combination(budget=budget, num_of_finetuned=self.num_of_finetune)
        
        ##
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod
        self.hook = {} # Hook being a dict: where key is the module name and value is the hook

        ###################################### Create configuration to modify model #########################################
        self.filter_cfgs = {"type": "conv", "backbone": backbone}
        self.handle_finetune()
        ###########################################################################################################
        

        if load != None and (self.measure_perplexity_HOSVD_var == False):
            state_dict = th.load(load)['state_dict']
            self.load_state_dict(state_dict)
        
        if self.force_use_base: #Bắt buộc dùng bản base
            register_normal_conv(self, self.filter_cfgs)
        elif self.measure_perplexity_HOSVD_var:
            register_measure_perplexity_HOSVD(self, self.filter_cfgs)
             
        elif self.with_ASI:
            register_ASI(self, self.filter_cfgs)
        elif self.with_HOSVD_var:
            register_HOSVD_var_filter(self, self.filter_cfgs)
        elif self.with_grad_filter:
            register_grad_filter(self, self.filter_cfgs)
        
        self.acc.reset()
    
    def attach_memory_list(self):
        self.k0_hosvd = []
        self.k1_hosvd = []
        self.k2_hosvd = []
        self.k3_hosvd = []
        self.raw_size = []
        self.output_size = []
        self.k_hosvd = [self.k0_hosvd, self.k1_hosvd, self.k2_hosvd, self.k3_hosvd, self.raw_size, self.output_size]

        self.filter_cfgs["k_hosvd"] = self.k_hosvd

        if self.with_HOSVD_var:
            register_HOSVD_var_filter(self, self.filter_cfgs)
      
        self.update_optimizer()


    
    def clear_measured_variables(self):
        for i in range(len(self.perplexity)):
            self.perplexity[i]     = None
            self.measured_rank[i]  = None
            self.layer_mem[i]      = None
            
    def reset(self):
        # Reset the model to its initial state
        self.__init__(**self.initial_state)

    def reset_k_hosvd(self):
        self.k0_hosvd.clear()
        self.k1_hosvd.clear()
        self.k2_hosvd.clear()
        self.k3_hosvd.clear()
        self.raw_size.clear()
        self.output_size.clear()

    
    def set_filter_configs(self, finetuned_layer):
        """ Helper function to set filter configurations based on conditions """
        if finetuned_layer == None: self.filter_cfgs = -1
        else:
            self.filter_cfgs["finetuned_layer"] = finetuned_layer
            if self.measure_perplexity_HOSVD_var:
                new_items = {"explain_variance_threshold": self.truncation_threshold, "perplexity": self.perplexity, "measured_rank": self.measured_rank, "layer_mem": self.layer_mem}

            elif self.with_ASI:
                new_items = {"truncation_threshold": self.suitable_ranks, "no_reuse": self.no_reuse}

            elif self.with_HOSVD_var:
                new_items = {"explained_variance_threshold": self.truncation_threshold, "k_hosvd": None}

            elif self.with_grad_filter:
                new_items = {"radius": self.filt_radius}
            else:
                new_items = {}
            self.filter_cfgs.update(new_items)
        
    def freeze_layers(self):
        """ Helper function to freeze layers that are not being finetuned """
        if self.num_of_finetune != 0 and self.num_of_finetune != None:
            if self.num_of_finetune != "all" and self.num_of_finetune <= len(self.all_conv_layers):
                self.all_conv_layers = self.all_conv_layers[-self.num_of_finetune:]  # Apply filter to last 'num_of_finetune' layers

                self.name_conv_layers_with_relu = self.name_conv_layers_with_relu[-self.num_of_finetune:]
            
            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0:
                    if name not in self.all_conv_layers and name != '':
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False  # Freeze layer
                    elif name in self.all_conv_layers:
                        break
            return self.all_conv_layers
        else:
            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0:
                    if name != '':
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False  # Freeze layer
            return None
    
    def handle_finetune(self):
        if not self.measure_perplexity_HOSVD_var:
            """ Handle the logic for finetuning based on num_of_finetune """
            if self.num_of_finetune != 0 and self.num_of_finetune != "all":
                self.all_conv_layers = self.freeze_layers()
            elif self.num_of_finetune == 0 or self.num_of_finetune == None: # If no finetune => freeze all
                logging.info("[Warning] number of finetuned layers is 0 => Freeze all layers !!")
                self.all_conv_layers = self.freeze_layers()
            else:
                logging.info("[Warning] Missing configuration !!")
                self.all_conv_layers = None
            self.set_filter_configs(self.all_conv_layers)
        else:
            self.set_filter_configs(self.all_conv_layers)

            
    def activate_hooks(self, is_activated=True):
        for h in self.hook:
            self.hook[h].activate(is_activated)

    def remove_hooks(self):
        for h in self.hook:
            self.hook[h].remove()
        self.hook.clear()
        logging.info("Hook is removed")

    def reset_activation_maps(self):
        self.activation_maps.clear()
        for layer in range(self.num_of_finetune):
            self.activation_maps.append([])

    def attach_hooks_for_conv(self, consider_active_only=True):
        if not consider_active_only:
            conv_layers = get_all_conv_with_name(self)
        else:
            conv_layers = get_active_conv_with_name(self)
        assert conv_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

        for name, mod in  conv_layers.items():
            self.hook[name] = Hook(mod)

    def get_activation_size(self, consider_active_only=True, element_size=4, unit="MB", register_hook=False): # For VanillaBP and Gradient Filter
        if self.with_HOSVD_var:
            return
        # Register hook to log input/output size
        if register_hook:
            self.attach_hooks_for_conv(consider_active_only=consider_active_only)
            self.activate_hooks(True)
        else:
            #############################################################################
            _, first_hook = next(iter(self.hook.items()))
            if first_hook.active: logging.info("Hook is activated")
            else: logging.info("[Warning] Hook is not activated !!")

            num_element = 0
            num_flops_fw = 0
            num_flops_bw = 0

            for layer_index, name in enumerate(self.hook): # through each layer
                input_size = self.hook[name].input_size

                B, C, H, W = input_size
                _, C_prime, H_prime, W_prime = self.hook[name].output_size
                K_H, K_W = self.hook[name].module.kernel_size

                if isinstance(self.hook[name].module, Conv2dAvg):
                    stride = self.hook[name].module.stride
                    x_h, x_w = input_size[-2:]
                    h, w = self.hook[name].output_size[-2:] 

                    p_h, p_w = ceil(h / self.filt_radius), ceil(w / self.filt_radius)
                    x_order_h, x_order_w = self.filt_radius * stride[0], self.filt_radius * stride[1]
                    x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

                    x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
                    x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

                    num_element += int(input_size[0] * input_size[1] * x_sum_height * x_sum_width)

                    ############### FLOPs
                    forward_overhead = (H/self.filt_radius)*(W/self.filt_radius)*(self.filt_radius**2-1)*C*B
                    conv_forward = K_H*K_W*C*C_prime*B*H*W

                    gradient_filering_overhead = B*C_prime*(h/self.filt_radius)*(w/self.filt_radius)
                    weight_sum_overhead = C_prime*C*(K_H*K_W-1)
                    scalar_mult_backward = B*C_prime*(h/self.filt_radius)*(w/self.filt_radius)
                    frobenius_backward = K_H*K_W*C*C_prime*B*H*W

                    num_flops_fw += forward_overhead + conv_forward
                    num_flops_bw += gradient_filering_overhead + weight_sum_overhead + scalar_mult_backward + frobenius_backward

                elif isinstance(self.hook[name].module, Conv2d_ASI):
                    from custom_op.compression.hosvd_subspace_iteration import hosvd_subspace_iteration
                    S, u_list = hosvd_subspace_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index])

                    K0, K1, K2, K3 = S.shape

                    num_element += S.numel() + sum(u.numel() for u in u_list)

                    fw_overhead = 0
                    for K in S.shape:
                        fw_overhead += 2*B*C*H*W*K + K**3
                    vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B

                    bw = (K0*C_prime*H_prime*W_prime*B + H*K0*K1*K2*K3 + H*W*K0*K1*K3 + C_prime*K0*K1*K_H*K_W*H_prime*W_prime + C_prime*C*K_H*K_W*K1)

                    num_flops_fw += fw_overhead + vanilla_fw
                    num_flops_bw += bw
                    
                elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d) and self.with_base:
                    num_element += int(input_size[1] * input_size[2] * input_size[3] * input_size[0])

                    vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B
                    vanilla_bw = (K_H*K_W*C*C_prime*H_prime*W_prime)*B

                    num_flops_fw += vanilla_fw
                    num_flops_bw += vanilla_bw



            self.remove_hooks()

            if unit == "Byte":
                res = str(num_element*element_size)
            if unit == "MB":
                res = str((num_element*element_size)/(1024*1024))
            elif unit == "KB":
                res = str((num_element*element_size)/(1024))
            else:
                raise ValueError("Unit is not suitable")
            
            with open(os.path.join(self.logger.log_dir, f'activation_memory_{unit}.log'), "a") as file:
                file.write(f"Activation memory is {res} {unit}\n")
        
            with open(os.path.join(self.logger.log_dir, f'total_FLOPs.log'), "a") as file:
                file.write(f"Forward: {num_flops_fw}\n")
                file.write(f"Backward: {num_flops_bw}\n")
                file.write(f"Total FLOPs: {num_flops_fw + num_flops_bw}\n")
            
    
    def get_activation_size_hosvd(self, num_batches, element_size=4, unit="MB"):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        k_hosvd_tensor = th.tensor(self.k_hosvd[:4], device=device).float() # Shape: (4 k, #batches * num_of_finetune)
        k_hosvd_tensor = k_hosvd_tensor.view(4, num_batches, self.num_of_finetune) # Shape: (4 k, #batches, num_of_finetune)
        k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: (num_of_finetune, #batch, 4 k)


        input_shapes = th.tensor(self.k_hosvd[4], device=device).reshape(num_batches, self.num_of_finetune, 4) # Shape: (#batch, num_of_finetune, 4 shapes)
        input_shapes = input_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shapes)

        output_shapes = th.tensor(self.k_hosvd[5], device=device).reshape(num_batches, self.num_of_finetune, 4) # Shape: (#batch, num_of_finetune, 4 shapes)
        output_shapes = output_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shapes)

        '''
        Iterate through each layer (dimension 1: num_of_finetune) 
        -> Iterate through each batch (dimension 2: #batches), calculate the number of elements here, then infer the average number of elements per batch for each layer 
        -> Sum everything to get the average number of elements per batch across all layers.
        '''
        num_element_all = th.sum(
            k_hosvd_tensor[:, :, 0] * k_hosvd_tensor[:, :, 1] * k_hosvd_tensor[:, :, 2] * k_hosvd_tensor[:, :, 3]
            + k_hosvd_tensor[:, :, 0] * input_shapes[:, :, 0]
            + k_hosvd_tensor[:, :, 1] * input_shapes[:, :, 1]
            + k_hosvd_tensor[:, :, 2] * input_shapes[:, :, 2]
            + k_hosvd_tensor[:, :, 3] * input_shapes[:, :, 3],
            dim=1
        )
        num_element = th.sum(num_element_all) / k_hosvd_tensor.shape[1]

        if unit == "Byte":
            res = num_element*element_size
        elif unit == "MB":
            res = (num_element*element_size)/(1024*1024)
        elif unit == "KB":
            res = (num_element*element_size)/(1024)
        else:
            raise ValueError("Unit is not suitable")
        
        with open(os.path.join(self.logger.log_dir, f'activation_memory_{unit}.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str(float(res)) + "\n")

        #################### FLOPs
        num_flops_fw = 0
        num_flops_bw = 0

        for layer_idx, name in enumerate(self.filter_cfgs["finetuned_layer"]):
            path_seq = name.split('.')
            target = reduce(getattr, path_seq, self)
            K_H, K_W = target.kernel_size

            K0, K1, K2, K3 = k_hosvd_tensor[layer_idx, :, 0], k_hosvd_tensor[layer_idx, :, 1], k_hosvd_tensor[layer_idx, :, 2], k_hosvd_tensor[layer_idx, :, 3]
            B, C, H, W = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2], input_shapes[layer_idx, :, 3]
            C_prime, H_prime, W_prime = output_shapes[layer_idx, :, 1], output_shapes[layer_idx, :, 2], output_shapes[layer_idx, :, 3]
            
            if self.current_epoch == 0: # forward flops:
                term1 = th.max(B, C * H * W) ** 2 * th.min(B, C * H * W)
                term2 = th.max(C, B * H * W) ** 2 * th.min(C, B * H * W)
                term3 = th.max(H, B * C * W) ** 2 * th.min(H, B * C * W)
                term4 = th.max(W, B * C * H) ** 2 * th.min(W, B * C * H)

                vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B # Shape: (#batch, 1)

                hosvd_neurips_fw_overhead = (term1 + term2 + term3 + term4)
                hosvd_neurips_fw = hosvd_neurips_fw_overhead + vanilla_fw
                num_flops_fw += hosvd_neurips_fw # Shape: (#batch, 1)

            hosvd_neurips_FLOPs_bw = (K0*C_prime*H_prime*W_prime*B + H*K0*K1*K2*K3 + H*W*K0*K1*K3 + C_prime*K0*K1*K_H*K_W*H_prime*W_prime + C_prime*C*K_H*K_W*K1)
            num_flops_bw += hosvd_neurips_FLOPs_bw # Shape: (#batch, 1)
        
        num_flops_bw = (th.sum(num_flops_bw, dim=0)/num_flops_bw.shape[0]).item()

        if self.current_epoch == 0:
            self.num_flops_fw = (th.sum(num_flops_fw, dim=0)/num_flops_fw.shape[0]).item()
            with open(os.path.join(self.logger.log_dir, f'fw_FLOPs.log'), "a") as file:
                file.write(f"Forward FLOPs (same for all epoch): {self.num_flops_fw}\n")

        with open(os.path.join(self.logger.log_dir, f'bw_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str(num_flops_bw) + "\n")

        with open(os.path.join(self.logger.log_dir, f'total_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str((self.num_flops_fw + num_flops_bw)) + "\n")

    def configure_optimizers(self):
        if self.use_sgd:
            optimizer = th.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            if self.lr_warmup == 0:
                scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.anneling_steps, eta_min=0.1 * self.learning_rate)
            else:
                def _lr_fn(epoch):
                    if epoch < self.lr_warmup:
                        lr = self.init_lr_prod + (1 - self.init_lr_prod) / (self.lr_warmup - 1) * epoch
                    else:
                        e = epoch - self.lr_warmup
                        es = self.anneling_steps - self.lr_warmup
                        lr = 0.5 * (1 + np.cos(np.pi * e / es))
                    return lr
                scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_fn)
            sch = {
                "scheduler": scheduler,
                'interval': self.scheduler_interval,
                'frequency': 1
            }
            return [optimizer], [sch]
        optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.8, 0.9))
        return [optimizer]
   
    def update_optimizer(self):
        # Get the current optimizer
        optimizer = self.trainer.optimizers[0]

        for name in self.filter_cfgs["finetuned_layer"]:
            path_seq = name.split('.')
            target = reduce(getattr, path_seq, self) # Turn on gradient
            

            optimizer.add_param_group({
                'params': filter(lambda p: p.requires_grad, target.parameters())
            })


    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
        ):
            optimizer_closure()

            if not self.measure_perplexity_HOSVD_var:
                optimizer.step()
                optimizer.zero_grad()
    
    def bn_eval(self):
        def f(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
            m.momentum = 1.0
        self.apply(f)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.pooling(feat)
        feat = feat.flatten(start_dim=1)
        logit = self.classifier(feat)
        return logit
    
    def forward_data(self, dataloader, tilte='Forwarding data'):
        self.eval()
        with th.no_grad():
            for input_data in tqdm(dataloader, desc=tilte, unit="batch"):
                img, _ = input_data['image'].to(self.device), input_data['label'].to(self.device)
                _ = self.forward(img)

    def training_step(self, train_batch, batch_idx):
        if self.set_bn_eval:
            self.bn_eval()
        img, label = train_batch['image'], train_batch['label']
        if img.shape[1] == 1:
            img = th.cat([img] * 3, dim=1)
        logits = self.forward(img)
        pred_cls = th.argmax(logits, dim=-1)

        acc = th.sum(pred_cls == label) / label.shape[0]
        
        loss = self.loss(logits, label)
        self.log("Train/Loss", loss)
        self.log("Train/Acc", acc)

        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        with open(os.path.join(self.logger.log_dir, 'train_loss.log'), 'a') as f:
            mean_loss = th.stack([o['loss'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_loss}")
            f.write("\n")

        with open(os.path.join(self.logger.log_dir, 'train_acc.log'), 'a') as f:
            mean_acc = th.stack([o['acc'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_acc}")
            f.write("\n")

    def validation_step(self, val_batch, batch_idx):
        img, label = val_batch['image'], val_batch['label']
        if img.shape[1] == 1:
            img = th.cat([img] * 3, dim=1)

        logits = self.forward(img)
        probs = logits.softmax(dim=-1)
        pred = th.argmax(logits, dim=1)
        self.acc(probs, label)
        loss = self.loss(logits, label)
        self.log("Val/Loss", loss)
        return {'pred': pred, 'prob': probs, 'label': label}

    def validation_epoch_end(self, outputs):
        f = open(os.path.join(self.logger.log_dir, 'val.log'), 'a') if self.logger is not None else None
        acc = self.acc.compute()
        if self.logger is not None:
            f.write(f"{self.current_epoch} {acc}\n")
            f.close()
        self.log("Val/Acc", acc)
        self.log("val-acc", acc)
        self.acc.reset()
