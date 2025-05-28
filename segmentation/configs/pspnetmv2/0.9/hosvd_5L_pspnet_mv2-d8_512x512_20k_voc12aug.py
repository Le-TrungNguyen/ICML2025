_base_ = "./../full_pspnet_mv2-d8_512x512_20k_voc12aug.py"

freeze_layers = ["backbone"]

hosvd_var = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', SVD_var=0.9),
    ]
)
