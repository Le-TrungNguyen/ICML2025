# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings

# Comment these lines if you want to measure perplexity or resource consumption
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

# test_times = 1 # Number of batches to feed into each checkpoint for testing
# runner = dict(type='IterBasedRunner', max_iters=test_times)
# checkpoint_config = dict(by_epoch=False, interval=test_times+1)
# evaluation = dict(interval=test_times+1, metric='mIoU', pre_eval=True)
