saved_location="runs"

setup="B"
dataset="imagenet"
num_classes="1000"
model="resnet18"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/res18/pretrain_13.10_imagenet/version_0/checkpoints/epoch=179-val-acc=0.753.ckpt"

imagenet_link="/home/infres/lnguyen-23/imagenet/"

perplexity_pkl="--model.perplexity_pkl ${saved_location}/setup$setup/$model/imagenet/perplexity_HOSVD_var/perplexity_test_var_0.4to0.9_imagenet/perplexity.pkl"


# # Set this kiable if want to resume training
# checkpoint="--checkpoint xxx"

general_config_args="--config configs/${model}_config.yaml"
logger_args="--logger.save_dir ${saved_location}/setup$setup/$model/$dataset/ASI"
data_args="--data.setup $setup --data.name $dataset --data.data_dir $imagenet_link --data.train_workers 5 --data.val_workers 5 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gradient_clip_val 2.0"
model_args="--model.setup $setup --model.with_ASI True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.005 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $perplexity_pkl $trainer_args $data_args $model_args $load_args $logger_args $seed_args $checkpoint"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name ASI_l2_k${k}_${usr_group_kl} --model.num_of_finetune 2 --model.budget 0.97306615114212
python trainer_cls.py ${common_args} --logger.exp_name ASI_l4_k${k}_${usr_group_kl} --model.num_of_finetune 4 --model.budget 2.88905882835388