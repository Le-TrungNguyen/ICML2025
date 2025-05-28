_base_ = 'full_upernet_r18_512x512_20k_voc12aug.py'

freeze_layers = [
    'backbone', 'decode_head', '~decode_head.conv_seg',
    '~decode_head.fpn_bottleneck', '~decode_head.fpn_convs', '~decode_head.bottleneck'
]

base = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.fpn_bottleneck", type='cbr'),
        dict(path="decode_head.fpn_convs.0", type='cbr'),
        dict(path="decode_head.fpn_convs.1", type='cbr'),
        dict(path="decode_head.fpn_convs.2", type='cbr'),
        dict(path="decode_head.bottleneck", type='cbr'),
    ]
)
