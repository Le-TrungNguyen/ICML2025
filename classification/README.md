
# Logging Resource Consumption

To log the resource consumption for each experiment, set the flag `log_activation_mem` to `True` in the `run()` function of either `trainer_cls.py` (for convolutional models) or `trainer_cls_linear.py` (for transformer models) or `trainer_cls_llm.py` (for TinyLlama).

The output will be saved in the experiment’s result folder. (For example:
`runs/setupA/mcunet/flowers102/HOSVD/var0.8/HOSVD_l2_var0.8_full_pretrain_imagenet_flowers102/version_0/activation_memory_MB.log`  for experiment that finetunes 2 last convolutional layers of MCUNet using HOSVD with $\varepsilon=0.8$ following setup A and FLowers102 dataset).