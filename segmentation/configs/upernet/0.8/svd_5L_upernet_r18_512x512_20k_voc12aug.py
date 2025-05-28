_base_ = '../full_upernet_r18_512x512_20k_voc12aug.py'

freeze_layers = [
    'backbone', 'decode_head', '~decode_head.conv_seg',
    '~decode_head.fpn_bottleneck', '~decode_head.fpn_convs', '~decode_head.bottleneck'
]

svd_var = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.fpn_bottleneck", type='cbr', SVD_var=0.8),
        dict(path="decode_head.fpn_convs.0", type='cbr', SVD_var=0.8),
        dict(path="decode_head.fpn_convs.1", type='cbr', SVD_var=0.8),
        dict(path="decode_head.fpn_convs.2", type='cbr', SVD_var=0.8),
        dict(path="decode_head.bottleneck", type='cbr', SVD_var=0.8),
    ]
)
