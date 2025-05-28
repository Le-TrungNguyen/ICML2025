python train.py configs/pspnetmv2/0.8/hosvd_5L_pspnet_mv2-d8_512x512_20k_voc12aug.py \
    --load-from calib/calib_pspnet_mv2-d8_512x512_5k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_pspnet_mv2-d8_512x512_20k_cityscapes/perplexity_combined.pkl"\
    --budget 0.0315906517207623

python train.py configs/pspnetmv2/0.8/hosvd_10L_pspnet_mv2-d8_512x512_20k_voc12aug.py \
    --load-from calib/calib_pspnet_mv2-d8_512x512_5k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_pspnet_mv2-d8_512x512_20k_cityscapes/perplexity_combined.pkl"\
    --budget 1.46979808807373