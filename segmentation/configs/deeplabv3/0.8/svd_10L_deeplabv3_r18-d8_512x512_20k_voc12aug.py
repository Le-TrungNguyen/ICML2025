_base_ = "../full_deeplabv3_r18-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "decode_head", "~decode_head.conv_seg",
    "~decode_head.bottleneck", "~decode_head.aspp_modules",
    "~backbone.layer4.1.conv2",
    "~backbone.layer4.1.conv1",
    "~backbone.layer4.0.conv2",
    "~backbone.layer4.0.conv1",
]

svd_var = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', SVD_var=0.8),
        dict(path="decode_head.aspp_modules.0", type='cbr', SVD_var=0.8),
        dict(path="decode_head.aspp_modules.1", type='cbr', SVD_var=0.8),
        dict(path="decode_head.aspp_modules.2", type='cbr', SVD_var=0.8),
        dict(path="decode_head.aspp_modules.3", type='cbr', SVD_var=0.8),
        dict(path="backbone.layer4.1", type='resnet_basic_block', SVD_var=0.8),
        dict(path="backbone.layer4.0", type='resnet_basic_block', SVD_var=0.8),
    ]
)
