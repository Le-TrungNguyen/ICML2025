setup="A"
dataset="BoolQ"
num_classes="2"


general_config_args="--config configs/TinyLlama_config.yaml"
logger_args="--logger.save_dir runs_tinyllama_ddp_/setup$setup/TinyLlama/$dataset/base"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50" # --trainer.gpus 2 --trainer.strategy ddp"
model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233 --data.batch_size 8"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args


python trainer_cls_llm.py ${common_args} --logger.exp_name base_l5_${usr_group_kl} --model.num_of_finetune 5
python trainer_cls_llm.py ${common_args} --logger.exp_name base_l4_${usr_group_kl} --model.num_of_finetune 4
python trainer_cls_llm.py ${common_args} --logger.exp_name base_l3_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls_llm.py ${common_args} --logger.exp_name base_l2_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls_llm.py ${common_args} --logger.exp_name base_l1_${usr_group_kl} --model.num_of_finetune 1
# python trainer_cls_llm.py ${common_args} --logger.exp_name base_all_${usr_group_kl} --model.num_of_finetune all