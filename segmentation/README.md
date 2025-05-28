# Running ASI

Suppose we want to run ASI on DeepLabV3 (dlv3).

**Step 1:** First, we need to measure perplexity by feeding 1 batch of pretrained data (Cityscapes). Modify the following configuration in [schedule_20k.py](mmsegmentation/configs/_base_/schedules/schedule_20k.py):

```python
test_times = 1  # Number of batches to feed into each checkpoint for testing
runner = dict(type='IterBasedRunner', max_iters=test_times)
checkpoint_config = dict(by_epoch=False, interval=test_times + 1)  # Save checkpoint after every 'test_times + 1' steps
evaluation = dict(interval=test_times + 1, metric='mIoU', pre_eval=True)  # Evaluate after every 'test_times + 1' steps
```

**Step 2:** Measure the perplexity by running:

```bash
bash scripts/dlv3/measure_ASI_dlv3.sh
```

**Step 3:** Revert the [schedule_20k.py](mmsegmentation/configs/_base_/schedules/schedule_20k.py) file to its original state.

**Step 4:** Run the experiment with ASI:

```bash
bash scripts/dlv3/ASI_dlv3.sh
```

**Note:** For other experiments, simply run the corresponding bash files as usual.


# Logging Resource Consumption:

**Step 1**: Run any experiment, for example, HOSVD for DeepLabV3 with MobileNetV2 as the backbone and variance 0.8. Use the following command:

```bash 
bash scripts/dlv3m/HOSVD/hosvd_dlv3m_0.8.sh
```

**Step 2**: The results will be saved in the directory:

`runs/HOSVD/0.8/hosvd_5L_deeplabv3_mv2_512x512_20k_voc12aug` 

This folder contains multiple checkpoint files named as `iter_{step}.pth`, where `step` represents the point in time at which the checkpoint was saved.

**Step 3**: Modify the following configuration in `mmsegmentation/configs/_base_/schedules/schedule_20k.py`:

```
test_times = 10  # Number of batches to feed into each checkpoint for testing
runner = dict(type='IterBasedRunner', max_iters=test_times)
checkpoint_config = dict(by_epoch=False, interval=test_times + 1)  # Save checkpoint after every 'test_times + 1' steps
evaluation = dict(interval=test_times + 1, metric='mIoU', pre_eval=True)  # Evaluate after every 'test_times + 1' steps
```

**Step 4**: In the experiment script, replace `train.py` with `train_count_mem.py`.

For example, the content of the file `scripts/dlv3m/HOSVD/hosvd_dlv3m_0.8.sh` should be updated to:

```
### Cityscapes -> VOC12Aug: Train the last 5 layers of DeepLabV3-ResNet18 with HOSVD
python train_count_mem.py configs/deeplabv3mv2/0.8/hosvd_5L_deeplabv3_mv2_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_mv2_512x512_5k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233

# Cityscapes -> VOC12Aug: Train the last 10 layers of DeepLabV3-ResNet18 with HOSVD
python train_count_mem.py configs/deeplabv3mv2/0.8/hosvd_10L_deeplabv3_mv2_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_mv2_512x512_5k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233
```

**Step 6**: Run the modified script:

```bash 
bash scripts/dlv3m/HOSVD/hosvd_dlv3m_0.8.sh
```

The result will be saved in: `runs/HOSVD/0.8/hosvd_5L_deeplabv3_mv2_512x512_20k_voc12aug/version_0/mem_log/activation_memory_MB.log`.