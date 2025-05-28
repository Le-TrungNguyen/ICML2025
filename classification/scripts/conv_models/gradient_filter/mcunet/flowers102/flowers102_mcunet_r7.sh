setup="A"
dataset="flowers102"
num_classes="102"
radius="7"
usr_group_kl="full_pretrain_imagenet"

general_config_args="--config configs/mcunet_config.yaml"
logger_args="--logger.save_dir runs/setup$setup/mcunet/$dataset/gradfilt/$radius"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.setup $setup --model.with_grad_filter True --model.filt_radius $radius --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r${radius}_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r${radius}_${usr_group_kl} --model.num_of_finetune 4