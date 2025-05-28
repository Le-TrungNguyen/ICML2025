_base_ = '../full_fcn_r18-d8_512x512_20k_voc12aug.py'

freeze_layers = [
    "backbone",
    "~backbone.layer4",
    "~backbone.layer3.1",
]

svd_var = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.conv_cat", type='cbr', SVD_var=0.9),
        dict(path="decode_head.convs.0", type='cbr', SVD_var=0.9),
        dict(path="decode_head.convs.1", type='cbr', SVD_var=0.9),
        dict(path="backbone.layer4.1", type='resnet_basic_block', SVD_var=0.9),
        dict(path="backbone.layer4.0", type='resnet_basic_block', SVD_var=0.9),
        dict(path="backbone.layer4.0.downsample.0", type='conv', SVD_var=0.9),
        dict(path="backbone.layer3.1", type='resnet_basic_block', SVD_var=0.9),
    ]
)
