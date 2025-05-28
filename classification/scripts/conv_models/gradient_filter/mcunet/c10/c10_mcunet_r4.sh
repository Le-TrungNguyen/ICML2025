setup="A"
dataset="cifar10"
num_classes="10"
radius="4"
usr_group_kl="full_pretrain_imagenet"

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/mcu/pretrain_15.29_cifar10/version_0/checkpoints/epoch=49-val-acc=0.950.ckpt"

general_config_args="--config configs/mcunet_config.yaml"
logger_args="--logger.save_dir runs/setup$setup/mcunet/$dataset/gradfilt/$radius"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.setup $setup --model.with_grad_filter True --model.filt_radius $radius --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

# python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r${radius}_${usr_group_kl} --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r${radius}_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r${radius}_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r${radius}_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name filt_l5_r${radius}_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name filt_l6_r${radius}_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name filt_l7_r${radius}_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name filt_l8_r${radius}_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name filt_l9_r${radius}_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls.py ${common_args} --logger.exp_name filt_l10_r${radius}_${usr_group_kl} --model.num_of_finetune 10
# python trainer_cls.py ${common_args} --logger.exp_name filt_l11_r${radius}_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name filt_l12_r${radius}_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name filt_l13_r${radius}_${usr_group_kl} --model.num_of_finetune 13
# python trainer_cls.py ${common_args} --logger.exp_name filt_l14_r${radius}_${usr_group_kl} --model.num_of_finetune 14
# python trainer_cls.py ${common_args} --logger.exp_name filt_l15_r${radius}_${usr_group_kl} --model.num_of_finetune 15
# python trainer_cls.py ${common_args} --logger.exp_name filt_l16_r${radius}_${usr_group_kl} --model.num_of_finetune 16
# python trainer_cls.py ${common_args} --logger.exp_name filt_l17_r${radius}_${usr_group_kl} --model.num_of_finetune 17
# python trainer_cls.py ${common_args} --logger.exp_name filt_l18_r${radius}_${usr_group_kl} --model.num_of_finetune 18
# python trainer_cls.py ${common_args} --logger.exp_name filt_l19_r${radius}_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name filt_l20_r${radius}_${usr_group_kl} --model.num_of_finetune 20
# python trainer_cls.py ${common_args} --logger.exp_name filt_l21_r${radius}_${usr_group_kl} --model.num_of_finetune 21
# python trainer_cls.py ${common_args} --logger.exp_name filt_l22_r${radius}_${usr_group_kl} --model.num_of_finetune 22
# python trainer_cls.py ${common_args} --logger.exp_name filt_l23_r${radius}_${usr_group_kl} --model.num_of_finetune 23
# python trainer_cls.py ${common_args} --logger.exp_name filt_l24_r${radius}_${usr_group_kl} --model.num_of_finetune 24
# python trainer_cls.py ${common_args} --logger.exp_name filt_l25_r${radius}_${usr_group_kl} --model.num_of_finetune 25
# python trainer_cls.py ${common_args} --logger.exp_name filt_l26_r${radius}_${usr_group_kl} --model.num_of_finetune 26
# python trainer_cls.py ${common_args} --logger.exp_name filt_l27_r${radius}_${usr_group_kl} --model.num_of_finetune 27
# python trainer_cls.py ${common_args} --logger.exp_name filt_l28_r${radius}_${usr_group_kl} --model.num_of_finetune 28
# python trainer_cls.py ${common_args} --logger.exp_name filt_l29_r${radius}_${usr_group_kl} --model.num_of_finetune 29
# python trainer_cls.py ${common_args} --logger.exp_name filt_l30_r${radius}_${usr_group_kl} --model.num_of_finetune 30
# python trainer_cls.py ${common_args} --logger.exp_name filt_l31_r${radius}_${usr_group_kl} --model.num_of_finetune 31
# python trainer_cls.py ${common_args} --logger.exp_name filt_l32_r${radius}_${usr_group_kl} --model.num_of_finetune 32
# python trainer_cls.py ${common_args} --logger.exp_name filt_l33_r${radius}_${usr_group_kl} --model.num_of_finetune 33
# python trainer_cls.py ${common_args} --logger.exp_name filt_l34_r${radius}_${usr_group_kl} --model.num_of_finetune 34
# python trainer_cls.py ${common_args} --logger.exp_name filt_l35_r${radius}_${usr_group_kl} --model.num_of_finetune 35
# python trainer_cls.py ${common_args} --logger.exp_name filt_l36_r${radius}_${usr_group_kl} --model.num_of_finetune 36
# python trainer_cls.py ${common_args} --logger.exp_name filt_l37_r${radius}_${usr_group_kl} --model.num_of_finetune 37
# python trainer_cls.py ${common_args} --logger.exp_name filt_l38_r${radius}_${usr_group_kl} --model.num_of_finetune 38
# python trainer_cls.py ${common_args} --logger.exp_name filt_l39_r${radius}_${usr_group_kl} --model.num_of_finetune 39
# python trainer_cls.py ${common_args} --logger.exp_name filt_l40_r${radius}_${usr_group_kl} --model.num_of_finetune 40
# python trainer_cls.py ${common_args} --logger.exp_name filt_l41_r${radius}_${usr_group_kl} --model.num_of_finetune 41
# python trainer_cls.py ${common_args} --logger.exp_name filt_l42_r${radius}_${usr_group_kl} --model.num_of_finetune 42