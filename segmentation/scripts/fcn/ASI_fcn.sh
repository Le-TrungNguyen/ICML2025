python train.py configs/fcn/0.8/hosvd_5L_fcn_r18-d8_512x512_voc12aug.py \
    --load-from calib/calib_fcn_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_fcn_r18-d8_512x512_cityscapes/perplexity_combined.pkl"\
    --budget 1.43386590480804

python train.py configs/fcn/0.8/hosvd_10L_fcn_r18-d8_512x512_voc12aug.py \
    --load-from calib/calib_fcn_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_fcn_r18-d8_512x512_cityscapes/perplexity_combined.pkl"\
    --budget 3.76530504226685