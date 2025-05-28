from pytorch_lightning.callbacks import Callback
import torch
from custom_op.register import register_measure_perplexity_HOSVD
from pytorch_lightning import seed_everything


class LogActivationMemoryCallback(Callback):

    def __init__(self, log_activation_mem=False, perplexity=None):
        self.log_activation_mem             = log_activation_mem    # if True: Log estimation of activation memory
        self.first_train_batch_start_logged = False
        self.first_train_batch_end_logged   = False                 # a flag indicating that training of the 1st batch of the 1st epoch is finish
        self.training_begin                 = False                 # a flag indicating the beginning of training
        self.num_train_batches              = None                  # number of batch of data for training
        self.num_val_batches                = None                  # number of batch of data for validating

        if perplexity is not None:
            self.perplexity = perplexity
            self.total_epsilon = len(self.perplexity.perplexity[0])
            self.total_layer = len(self.perplexity.perplexity)
            self.epsilon_idx = 0
            self.layer_idx = 0

    def on_train_epoch_start(self, trainer, model):
        if not self.training_begin:
            if self.log_activation_mem:
                if hasattr(model, 'with_HOSVD_var') and model.with_HOSVD_var:
                    model.attach_memory_list()
                
            self.training_begin = True
        

        if hasattr(model, 'measure_perplexity_HOSVD_var') and model.measure_perplexity_HOSVD_var:
            model.filter_cfgs["explain_variance_threshold"] = self.perplexity.set_of_epsilons[self.epsilon_idx]
            print(f"For epsilon is {self.perplexity.set_of_epsilons[self.epsilon_idx]}")
            seed_everything(233)
            register_measure_perplexity_HOSVD(model, model.filter_cfgs)

            model.update_optimizer()

    def on_train_epoch_end(self, trainer, model):

        if self.log_activation_mem:
            if hasattr(model, 'with_HOSVD_var') and model.with_HOSVD_var:
                model.get_activation_size_hosvd(self.num_train_batches) # Decomposition only occurs during training 
                model.reset_k_hosvd()
            
        
    
    def on_train_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx):

        if self.log_activation_mem:
            if not self.first_train_batch_start_logged: # Log in the first epoch with the first train batch because the activation memory of these methods is stable.
                    model.get_activation_size(register_hook=True)
                    self.first_train_batch_start_logged = True

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        """
        Called at the end of a training batch.
        Logs activation memory for the first batch if applicable (for Vanilla Training and Gradient Filter)
        """
        self.num_train_batches = batch_idx + 1
        if self.log_activation_mem:
            if not self.first_train_batch_end_logged: # Log in the first epoch with the first train batch because the activation memory of these methods is stable.
                model.get_activation_size(register_hook=False)
                self.first_train_batch_end_logged = True
        
        if model.just_log:
            trainer.should_stop = True
            trainer.limit_val_batches = 0

        if hasattr(model, 'measure_perplexity_HOSVD_var') and model.measure_perplexity_HOSVD_var:
            for i in range(self.total_layer):
                self.perplexity.perplexity[i][self.epsilon_idx] = model.perplexity[i].item() if isinstance(model.perplexity[i], torch.Tensor) else model.perplexity[i]
                self.perplexity.ranks[i][self.epsilon_idx]      = model.measured_rank[i]
                self.perplexity.layer_mems[i][self.epsilon_idx] = model.layer_mem[i].item() if isinstance(model.layer_mem[i], torch.Tensor) else model.layer_mem[i]

            model.clear_measured_variables()

            self.epsilon_idx += 1
        

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        self.num_val_batches = batch_idx + 1