#!/bin/bash
saved_location="runs"
# Common setup
setup="A"
var="0.8"

# Dataset configurations
declare -A datasets
datasets=(
  ["pets"]=37
  ["flowers102"]=102
  ["cub200"]=200
  ["cifar10"]=10
  ["cifar100"]=100
)

# Model names
models=("mcunet" "mbv2" "resnet18" "resnet34")

# List of num_of_finetune values
num_of_finetune_list=(2 4)

# Loop through models and datasets
for model in "${models[@]}"; do
  echo "Processing model: $model"
  common_config_args="--config configs/${model}_config.yaml"
  common_data_args="--data.train_workers 24 --data.val_workers 24"
  common_model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch"
  common_trainer_args="--trainer.max_epochs 50 --trainer.gradient_clip_val 2.0"
  common_seed_args="--seed_everything 233"
  common_methods="--model.with_HOSVD_var True --model.truncation_threshold $var"
  common_args="$common_config_args $common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"

  for dataset in "${!datasets[@]}"; do
    num_classes=${datasets[$dataset]}
    echo "  Processing dataset: $dataset with num_classes: $num_classes"

    specific_logger_args="--logger.save_dir ${saved_location}/setup$setup/$model/$dataset/HOSVD_var/var$var"
    specific_data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset"
    specific_model_args="--model.num_classes $num_classes"
    specific_args="$specific_logger_args $specific_data_args $specific_model_args"

    all_args="$common_args $specific_args"
    echo $all_args

    for num_of_finetune in "${num_of_finetune_list[@]}"; do
      echo "    Running with num_of_finetune: $num_of_finetune"
      python trainer_cls.py ${all_args} --logger.exp_name HOSVD_var${var}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune
    done
  done
done
