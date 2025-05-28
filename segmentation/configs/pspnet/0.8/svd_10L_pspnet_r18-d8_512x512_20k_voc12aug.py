_base_ = "./../full_pspnet_r18-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "~backbone.layer4",
]


svd_var = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', SVD_var=0.8),
        dict(path="backbone.layer4.1", type='resnet_basic_block', SVD_var=0.8),
        dict(path="backbone.layer4.0", type='resnet_basic_block', SVD_var=0.8),
    ]
)
