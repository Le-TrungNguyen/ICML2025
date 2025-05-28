_base_ = 'full_fcn_r18-d8_512x512_20k_voc12aug.py'

freeze_layers = [
    "backbone",
    "~backbone.layer4.1",
]
base = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.conv_cat", type='cbr'),
        dict(path="decode_head.convs.0", type='cbr'),
        dict(path="decode_head.convs.1", type='cbr'),
        dict(path="backbone.layer4.1", type='resnet_basic_block'),
    ]
)

