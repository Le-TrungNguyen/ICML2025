#!/bin/bash

saved_location="runs"

# Dataset configurations
declare -A datasets
datasets=(
  ["pets"]=37
)

# Model names
models=("swinT")

# List of num_of_finetune values and corresponding budgets
num_of_finetune_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)


declare -A budgets=(
  ["pets"]="4.1121521 4.776542664 6.669616699 7.71157074 8.082237244 8.355361938 15.87916946 17.50820541 34.87229156 37.60500336 54.24913788 56.57781601 73.05618286 76.95783234 89.0435791 88.48342896 136.9461365 146.8933411 205.8098602 210.5010376 267.6213684 280.3143616 308.72052 311.1638794"
)


# Common setup
setup="A"
perplexity_pkl="--model.perplexity_pkl $saved_location/setupA/swinT/imagenet/perplexity_HOSVD_var/perplexity_test_var_0.4to0.9_imagenet/perplexity.pkl"
common_data_args="--data.train_workers 24 --data.val_workers 24"
common_model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch"
common_trainer_args="--trainer.max_epochs 50 --trainer.gradient_clip_val 2.0"
common_seed_args="--seed_everything 233"
common_methods="--model.with_ASI True --model.no_reuse False  --model.just_log False"

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
      python trainer_cls_linear.py ${all_args} --logger.exp_name ASI_budget${budget}_l${num_of_finetune} --model.num_of_finetune $num_of_finetune --model.budget $budget
    done
  done
done
