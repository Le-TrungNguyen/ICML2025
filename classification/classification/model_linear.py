import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_HOSVD_var_filter, register_ASI, register_measure_perplexity_HOSVD
from utils.util import get_all_linear_with_name, get_active_linear_with_name, Hook, calculate_flops_SVD
from models.encoders import get_encoder
from functools import reduce
import logging
import inspect
# from utils.perplexity import Perplexity
from utils.perplexity_dp import Perplexity

from custom_op.linear.linear_ASI import Linear_ASI

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes,
                 learning_rate, weight_decay, set_bn_eval, load = None,

                 num_of_finetune=None,

                 with_HOSVD_var = False, with_ASI=False, truncation_threshold=None, measure_perplexity_HOSVD_var=False,

                 no_reuse = False, just_log = False, budget=None, perplexity_pkl=None,
                 checkpoint=None,
                 
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
        self.backbone = get_encoder(backbone, self.setup, checkpoint, **backbone_args)
        if self.backbone_name == "swinT":
            self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True) # Change classifier
        elif self.backbone_name == "vit_b_32":
            self.backbone.heads = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True)) # Change classifier
        ##
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)

        self.all_linear_layers = get_all_linear_with_name(self) # A dictionary contains all linear layers (value) and their names (key)
        if num_of_finetune == "all" or num_of_finetune > len(self.all_linear_layers):
            logging.info("[Warning] Finetuning all layers")
            self.finetune_all = True
            self.num_of_finetune = len(self.all_linear_layers)
        else:
            self.num_of_finetune = num_of_finetune
            self.finetune_all = False

        ##
        self.measure_perplexity_HOSVD_var = measure_perplexity_HOSVD_var
        if self.measure_perplexity_HOSVD_var:
            self.perplexity     = [None for layer_idx in range(len(self.all_linear_layers))]
            self.measured_rank  = [None for layer_idx in range(len(self.all_linear_layers))]
            self.layer_mem      = [None for layer_idx in range(len(self.all_linear_layers))]
        
        self.just_log = just_log

        ##
        self.with_HOSVD_var = with_HOSVD_var
        self.with_ASI = with_ASI
        self.with_base = not (self.with_HOSVD_var or self.with_ASI)
        
        self.no_reuse = no_reuse
        self.truncation_threshold = truncation_threshold

        if self.with_ASI:
            self.perplexity_pkl = perplexity_pkl
            perplexity = Perplexity()
            perplexity.load(self.perplexity_pkl)
            best_memory, best_perplexity, best_indices, self.suitable_ranks = perplexity.find_best_ranks_dp(budget=budget, num_of_finetuned=self.num_of_finetune)
            del perplexity
        

        ##
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod
        self.hook = {} # Hook being a dict: where key is the module name and value is the hook

        ###################################### Create configuration to modify model #########################################
        self.filter_cfgs = {"type": "linear", "backbone": backbone}
        self.handle_finetune()
        #######################################################################
        

        if load != None:
            state_dict = th.load(load)['state_dict']
            self.load_state_dict(state_dict)
        
        if self.measure_perplexity_HOSVD_var:
            register_measure_perplexity_HOSVD(self, self.filter_cfgs)
        elif self.with_HOSVD_var:
            register_HOSVD_var_filter(self, self.filter_cfgs)
        elif self.with_ASI:
            register_ASI(self, self.filter_cfgs)


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

            else:
                new_items = {}
            self.filter_cfgs.update(new_items)
    
    def freeze_layers(self):
        """ Helper function to freeze layers that are not being finetuned """
        if self.num_of_finetune != 0 and self.num_of_finetune != None:
            if self.num_of_finetune != "all" and self.num_of_finetune <= len(self.all_linear_layers):
                self.all_linear_layers = dict(list(self.all_linear_layers.items())[-self.num_of_finetune:])

            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0:
                    if name not in self.all_linear_layers and name != '':
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False  # Freeze layer
                    elif name in self.all_linear_layers:
                        break
            return self.all_linear_layers
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
            if self.num_of_finetune != 0 and self.num_of_finetune != "all" and self.finetune_all == False:
                self.all_linear_layers = self.freeze_layers()
            elif self.num_of_finetune == 0 or self.num_of_finetune == None: # If no finetune => freeze all
                logging.info("[Warning] number of finetuned layers is 0 => Freeze all layers !!")
                self.all_linear_layers = self.freeze_layers()
            elif self.finetune_all:
                logging.info("[Warning] Finetune all model!!")
            else:
                logging.info("[Warning] Missing configuration !!")
                self.all_linear_layers = None
            self.set_filter_configs(self.all_linear_layers)
        else:
            self.set_filter_configs(self.all_linear_layers)

    def activate_hooks(self, is_activated=True):
        for h in self.hook:
            self.hook[h].activate(is_activated)

    def remove_hooks(self):
        for h in self.hook:
            self.hook[h].remove()
        self.hook.clear()
        logging.info("Hook is removed")

    def attach_hooks_for_linear(self, consider_active_only=False):
        if not consider_active_only:
            linear_layers = get_all_linear_with_name(self)
        else:
            linear_layers = get_active_linear_with_name(self)
        assert linear_layers != -1, "[Warning] Consider activate linear only but no linear is finetuned => No hook is attached !!"

        for name, mod in  linear_layers.items():
            self.hook[name] = Hook(mod)

    def get_activation_size(self, consider_active_only=True, element_size=4, unit="MB", register_hook=False): # element_size = 4 bytes
        if self.with_HOSVD_var:
            return
        # Register hook to log input/output size
        if register_hook:
            self.attach_hooks_for_linear(consider_active_only=consider_active_only)
            self.activate_hooks(True)
        #############################################################################
        else:
            _, first_hook = next(iter(self.hook.items()))
            if first_hook.active: logging.info("Hook is activated")
            else: logging.info("[Warning] Hook is not activated !!")
            #############################################################################
            # Feed one sample of data into model to record activation size
            num_element_activation = 0
            self.num_flops_fw = 0
            num_flops_bw = 0

            if self.backbone_name == "swinT":
                for layer_index, name in enumerate(self.hook): # through each layer
                    input_size = self.hook[name].input_size
                    B, H, W, C = [x.item() for x in input_size]

                    _, _, _, C_prime  = [x.item() for x in self.hook[name].output_size]
                    
                    if isinstance(self.hook[name].module, Linear_ASI):
                        # Calculate activation size
                        from custom_op.compression.hosvd_subspace_iteration import hosvd_subspace_iteration
                        S, u_list = hosvd_subspace_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)])
                        num_element_activation += S.numel() + sum(u.numel() for u in u_list)

                        # FLOPs
                        K1, K2, K3, K4 = S.shape

                        # Forward = overhead of ASI + vanilla cost
                        fw_overhead = 0
                        for K in S.shape:
                            fw_overhead += 2*B*C*H*W*K + K**3
                        vanilla_fw = (B*H*C_prime*W*(2*C-1))

                        # Backward = cost of calculating gradient for weight in low rank
                        bw = (B*K1*H*W*C_prime + H*K2*K1*K3*K4 + W*K3*K1*H*C_prime + C*K4*K1*H*K3 + K1*K3*H*C*C_prime)


                        self.num_flops_fw += fw_overhead + vanilla_fw
                        num_flops_bw += bw

                    elif isinstance(self.hook[name].module, nn.modules.linear.Linear):
                        num_element_activation += int(input_size.prod())

                        vanilla_fw = int(B*H*C_prime*W*(2*C-1))
                        vanilla_bw = int(C_prime*C*(2*B*H*W - 1))


                        self.num_flops_fw += vanilla_fw
                        num_flops_bw += vanilla_bw

            elif self.backbone_name == "vit_b_32":
                for layer_index, name in enumerate(self.hook): # through each layer
                    input_size = self.hook[name].input_size
                    B, N, I = [x.item() for x in input_size]
                    _, _, O  = [x.item() for x in self.hook[name].output_size]

                    if isinstance(self.hook[name].module, Linear_ASI):
                        # Log explained var of each layer
                        from custom_op.compression.hosvd_var import hosvd_var
                        S, u_list, ex_var_list = hosvd_var(self.hook[name].inputs[0], 0.9, return_full_rank=True)
                        with open(os.path.join(self.logger.log_dir, f'layer_explain_vars.txt'), "a") as file: # First column is layer, the remains are rank of each layer at current epoch
                            for mode, ex_var in enumerate(ex_var_list):
                                file.write(str(layer_index) + f"_mode_{mode}")
                                for var in ex_var:
                                    file.write("\t" + str(var.item()))
                                file.write("\n")
                            file.write("\n")
                        #######################################################

                        # Calculate activation size
                        from custom_op.compression.hosvd_subspace_iteration import hosvd_subspace_iteration
                        S, u_list = hosvd_subspace_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)])
                        num_element_activation += S.numel() + sum(u.numel() for u in u_list)

                        ########################## Tính FLOPs ######################
                        K1, K2, K3 = S.shape
                        
                        # Forward = overhead của ASI + vanilla cost
                        fw_overhead_activation = 0
                        for K in S.shape:
                            fw_overhead_activation += 2*B*N*I*K + K**3
                        vanilla_fw = int(B*N*O*(2*I-1))
                        # Backward = cost để tính gradient cho weight ở low rank
                        bw = (B*N*O*K1 + K1*K2*K3*N + K1*K3*I*N + I*O*N*K1)

                        self.num_flops_fw += fw_overhead_activation + vanilla_fw
                        num_flops_bw += bw


                    elif isinstance(self.hook[name].module, nn.modules.linear.Linear):
                        num_element_activation += int(input_size.prod())

                        # Tính FLOPs
                        vanilla_fw = int(B*N*O*(2*I-1))

                        vanilla_bw = int(O*I*(2*B*N-1))

                        self.num_flops_fw += vanilla_fw
                        num_flops_bw += vanilla_bw

            self.remove_hooks()

            if unit == "Byte":
                res_activation = str(num_element_activation*element_size)
            if unit == "MB":
                res_activation = str((num_element_activation*element_size)/(1024*1024))
            elif unit == "KB":
                res_activation = str((num_element_activation*element_size)/(1024))
            else:
                raise ValueError("Unit is not suitable")

            with open(os.path.join(self.logger.log_dir, f'activation_memory_{unit}.log'), "a") as file:
                file.write(f"Activation memory is {res_activation} {unit}\n")

            with open(os.path.join(self.logger.log_dir, f'total_FLOPs.log'), "a") as file:
                file.write(f"Forward: {self.num_flops_fw}\n")
                file.write(f"Backward: {num_flops_bw}\n")
                file.write(f"Total FLOPs: {self.num_flops_fw + num_flops_bw}\n")


    def get_activation_size_hosvd(self, num_batches, element_size=4, unit="MB"):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        if self.backbone_name == "swinT":
            k_hosvd_tensor = th.tensor(self.k_hosvd[:4], device=device).float() # Shape: (4 k, #batches * num_of_finetune)
            k_hosvd_tensor = k_hosvd_tensor.view(4, num_batches, -1) # Shape: (4 k, #batches, num_of_finetune)

            input_shapes = th.tensor(self.k_hosvd[4], device=device).reshape(num_batches, -1, 4) # Shape: (#batch, num_of_finetune, 4 shapes)
            output_shapes = th.tensor(self.k_hosvd[5], device=device).reshape(num_batches, -1, 4) # Shape: (#batch, num_of_finetune, 4 shapes)

        elif self.backbone_name == "vit_b_32":
            k_hosvd_tensor = th.tensor(self.k_hosvd[:3], device=device).float() # Shape: (3 k, #batches * num_of_finetune)
            k_hosvd_tensor = k_hosvd_tensor.view(3, num_batches, -1) # Shape: (3 k, #batches, num_of_finetune)

            input_shapes = th.tensor(self.k_hosvd[3], device=device).reshape(num_batches, -1, 3) # Shape: (#batch, num_of_finetune, 3 shapes)
            output_shapes = th.tensor(self.k_hosvd[4], device=device).reshape(num_batches, -1, 3) # Shape: (#batch, num_of_finetune, 3 shapes)

        k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: (num_of_finetune, #batch, 4 k) (hoặc 3 k)
        input_shapes = input_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shapes) (hoặc 3 shapes)
        output_shapes = output_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shapes) (hoặc 3 shapes)

        '''
        Iterate through each layer (dimension 1: num_of_finetune) 
        -> Iterate through each batch (dimension 2: #batches), calculate the number of elements here, then infer the average number of elements per batch for each layer 
        -> Sum everything to get the average number of elements per batch across all layers.
        '''

        num_element_all = th.sum(th.prod(k_hosvd_tensor, dim=2) + sum(k_hosvd_tensor[:, :, i] * input_shapes[:, :, i] for i in range(k_hosvd_tensor.shape[2])), dim=1)

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

        #################### Tính FLOPs
        num_flops_fw = 0
        num_flops_bw = 0

        if self.backbone_name == "swinT":
            for layer_idx in range(k_hosvd_tensor.shape[0]): # Duyệt qua từng layer active
                K1, K2, K3, K4 = k_hosvd_tensor[layer_idx, :, 0], k_hosvd_tensor[layer_idx, :, 1], k_hosvd_tensor[layer_idx, :, 2], k_hosvd_tensor[layer_idx, :, 3]
                B, H, W, C = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2], input_shapes[layer_idx, :, 3]
                C_prime = output_shapes[layer_idx, :, 3]
                
                if self.current_epoch == 0: # forward flops:
                    term1 = calculate_flops_SVD(B, C*H*W)
                    term2 = calculate_flops_SVD(C, B*H*W)
                    term3 = calculate_flops_SVD(H, B*C*W)
                    term4 = calculate_flops_SVD(W, B*C*H)

                    vanilla_fw = (B*H*C_prime*W*(2*C-1)) # Shape: (#batch, 1)

                    hosvd_neurips_fw_overhead = (term1 + term2 + term3 + term4)
                    hosvd_neurips_fw = hosvd_neurips_fw_overhead + vanilla_fw
                    num_flops_fw += hosvd_neurips_fw # Shape: (#batch, 1)

                hosvd_neurips_FLOPs_bw = (B*H*K3*C_prime*W + K1*H*K3*K4*K2 + B*H*K3*K4*K1 + B*H*K3*C*K4 + B*H*C*C_prime*K3)
                num_flops_bw += hosvd_neurips_FLOPs_bw # Shape: (#batch, 1)
        elif self.backbone_name == "vit_b_32":
            for layer_idx in range(k_hosvd_tensor.shape[0]):
                K1, K2, K3 = k_hosvd_tensor[layer_idx, :, 0], k_hosvd_tensor[layer_idx, :, 1], k_hosvd_tensor[layer_idx, :, 2]
                B, N, I = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2]
                O = output_shapes[layer_idx, :, 2]
                
                if self.current_epoch == 0: # forward flops:
                    term1 = calculate_flops_SVD(B, N*I)
                    term2 = calculate_flops_SVD(N, B*I)
                    term3 = calculate_flops_SVD(I, B*N)

                    vanilla_fw = (B*I*O*(2*N-1)) # Shape: (#batch, 1)

                    hosvd_neurips_fw_overhead = (term1 + term2 + term3)
                    hosvd_neurips_fw = hosvd_neurips_fw_overhead + vanilla_fw
                    num_flops_fw += hosvd_neurips_fw # Shape: (#batch, 1)

                hosvd_neurips_FLOPs_bw = (B*N*O*K1 + K1*K2*K3*N + K1*K3*I*N + I*O*N*K1)
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
        logit = self.backbone(x)
        return logit

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
        f = open(os.path.join(self.logger.log_dir, 'val.log'),
                 'a') if self.logger is not None else None
        acc = self.acc.compute()
        if self.logger is not None:
            f.write(f"{self.current_epoch} {acc}\n")
            f.close()
        self.log("Val/Acc", acc)
        self.log("val-acc", acc)
        self.acc.reset()