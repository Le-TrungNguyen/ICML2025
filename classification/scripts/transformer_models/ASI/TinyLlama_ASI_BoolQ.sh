#!/bin/bash

rank=20
saved_location="runs"

# Dataset configurations
declare -A datasets
datasets=(
  ["BoolQ"]=2
  # ["AG_News"]=4
)

# Model names
models=("TinyLlama")

num_of_finetune_list=(1)


# Common setup
setup="A"
common_data_args="--data.train_workers 24 --data.val_workers 24"
common_model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch"
common_trainer_args="--trainer.max_epochs 50 --trainer.gradient_clip_val 2.0" # --trainer.gpus 2 --trainer.strategy ddp"
common_seed_args="--seed_everything 233 --data.batch_size 8"
common_methods="--model.with_ASI True --model.no_reuse False --model.truncation_threshold $rank" # --data.num_train_batch 5 --data.num_val_batch 5"

common_args="$common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"

# Loop through models and datasets
for model in "${models[@]}"; do
  echo "Processing model: $model"

  model_config_args="--config configs/${model}_config.yaml"

  for dataset in "${!datasets[@]}"; do
    num_classes=${datasets[$dataset]}
    echo "  Processing dataset: $dataset with num_classes: $num_classes"
    specific_logger_args="--logger.save_dir ${saved_location}/setup$setup/$model/$dataset/ASI/"
    specific_data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset"
    specific_model_args="--model.num_classes $num_classes"
    specific_args="$specific_logger_args $specific_data_args $specific_model_args"

    all_args="$model_config_args $common_args $specific_args"
    # echo $all_args

    for i in "${!num_of_finetune_list[@]}"; do
      num_of_finetune="${num_of_finetune_list[i]}"

      python trainer_cls_llm.py ${all_args} --logger.exp_name ASI_rank${rank}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune
    done
  done
done
