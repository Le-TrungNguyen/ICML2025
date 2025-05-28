# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with HOSVD
python train.py configs/fcn/0.8/hosvd_5L_fcn_r18-d8_512x512_voc12aug.py --load-from calib/calib_fcn_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with HOSVD
python train.py configs/fcn/0.8/hosvd_10L_fcn_r18-d8_512x512_voc12aug.py --load-from calib/calib_fcn_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233