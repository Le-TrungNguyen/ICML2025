#!/bin/bash

saved_location="runs"

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
models=("mbv2")

# List of num_of_finetune values and corresponding budgets
num_of_finetune_list=(2 4)


declare -A budgets=(
  ["pets"]="0.207318842 0.769021571"
  ["flowers102"]="0.301968187 0.755280137"
  ["cub200"]="0.267010778 0.67493391"
  ["cifar10"]="0.259025842 0.677683592"
  ["cifar100"]="0.15810889 0.705279648"
)

# Common setup
setup="A"
var="0.8"
perplexity_pkl="--model.perplexity_pkl $saved_location/setupA/mbv2/imagenet/perplexity_HOSVD_var/perplexity_test_var_0.4to0.9_imagenet/perplexity.pkl"
common_data_args="--data.train_workers 24 --data.val_workers 24"
common_model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch"
common_trainer_args="--trainer.max_epochs 50 --trainer.gradient_clip_val 2.0"
common_seed_args="--seed_everything 233"
common_methods="--model.with_ASI True --model.no_reuse False"

common_args="$perplexity_pkl $common_data_args $common_model_args $common_trainer_args $common_seed_args $common_methods"

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

    # Get budgets for the current dataset
    IFS=' ' read -r -a dataset_budgets <<< "${budgets[$dataset]}"
    for i in "${!num_of_finetune_list[@]}"; do
      num_of_finetune="${num_of_finetune_list[i]}"
      budget="${dataset_budgets[i]}"

      echo "    Running with num_of_finetune: $num_of_finetune, budget: $budget"
      python trainer_cls.py ${all_args} --logger.exp_name ASI_budget${budget}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune --model.budget $budget
    done
  done
done
