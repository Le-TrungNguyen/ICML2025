python train.py configs/upernet/0.8/hosvd_5L_upernet_r18_512x512_20k_voc12aug.py \
    --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_upernet_r18_512x512_20k_cityscapes/perplexity_combined.pkl"\
    --budget 1.34823191165924

python train.py configs/upernet/0.8/hosvd_10L_upernet_r18_512x512_20k_voc12aug.py \
    --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth \
    --cfg-options data.samples_per_gpu=8 \
    --seed 233 \
    --with_ASI True \
    --perplexity_pkl "perplexity/perplexity_hosvd_10L_upernet_r18_512x512_20k_cityscapes/perplexity_combined.pkl"\
    --budget 1.67557716369628