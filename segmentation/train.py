# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)

from custom_op.register import register_filter, register_HOSVD_filter, register_SVD_filter, register_measure_perplexity_HOSVD, register_HOSVD_power4_budget_filter
from functools import reduce
import torch.nn as nn
from tools.perplexity import Perplexity, merged_perplexity
from tools.utils import delete_junk_folder

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--measure_perplexity', help='Measure perplexity or not', default=False)
    parser.add_argument('--SVD_var', help='SVD_var', default=0.8)
    parser.add_argument('--with_ASI', help='use ASI or not', default=False)
    parser.add_argument('--budget', help='budget for ASI', default=None)
    parser.add_argument('--perplexity_pkl', help='link to saved perplexity')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    parser.add_argument(
        '--log-postfix', default=''
    )
    parser.add_argument('--collect-moment', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def get_moment_logger(model, name):
    model.moment1[name] = 0.0
    model.moment2[name] = 0.0
    model.moment_step[name] = 0

    def _logger(grad):
        model.moment1[name] += (grad - model.moment1[name]) / (model.moment_step[name] + 1)
        model.moment2[name] += (grad.square() - model.moment2[name]) / (model.moment_step[name] + 1)
        model.moment_step[name] += 1

    return _logger


def get_latest_log_version(path):
    max_version = -1
    for p in os.walk(path):
        p = p[0].split('/')[-1]
        if p.startswith('version_') and p[8:].isdigit():
            v = int(p[8:])
            if v > max_version:
                max_version = v
    return max_version + 1


def main(SVD_var_measure_perplexity=None):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    if cfg.get("gradient_filter", None) is None: cfg.gradient_filter = dict(enable=False)
    if cfg.get("hosvd_var", None) is None: cfg.hosvd_var = dict(enable=False)
    if cfg.get("svd_var", None) is None: cfg.svd_var = dict(enable=False)
    if cfg.get("base", None) is None: cfg.base = dict(enable=False)
    
    if (cfg.gradient_filter["enable"] == False and 
        cfg.hosvd_var["enable"] == False and
        cfg.svd_var["enable"] == False and
        cfg.base["enable"] == False):
        cfg.full = dict(enable=True)
    else: cfg.full = dict(enable=False)

    if cfg.get("freeze_layers", None) is None: cfg.freeze_layers = []
    

    # work_dir is determined in this priority: CLI > segment in file > filename
    work_dir = './runs'
    postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
    log_name = osp.splitext(osp.basename(args.config))[0] + postfix
    if args.work_dir is not None:
        work_dir = args.work_dir
    work_dir = osp.join(work_dir, log_name)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if cfg.gradient_filter.enable:
        work_dir = osp.join(osp.join(osp.dirname(work_dir), 'gradient_filter'), osp.basename(work_dir))
        register_filter(model, cfg.gradient_filter)
    elif cfg.base.enable: work_dir = osp.join(osp.join(osp.dirname(work_dir), 'base'), osp.basename(work_dir))
    elif cfg.full.enable: work_dir = osp.join(osp.join(osp.dirname(work_dir), 'full'), osp.basename(work_dir))
    elif cfg.hosvd_var.enable:
        if args.measure_perplexity:
            work_dir = osp.join(osp.join(osp.dirname(work_dir), f'perplexity_{osp.basename(work_dir)}/'))
            work_dir = os.path.join('./perplexity', *work_dir.split(os.sep)[2:])
            total_conv_layer = 0
            for cf in cfg.hosvd_var['filter_install']:
                if cf['type'] == 'cbr' or cf['type'] == 'conv': total_conv_layer += 1
                elif cf['type'] == 'resnet_basic_block': total_conv_layer += 2

            perplexity=     [None for layer in range(total_conv_layer)]
            measured_rank=  [None for layer in range(total_conv_layer)]
            layer_mem=      [None for layer in range(total_conv_layer)]
            layer_name = [None for layer in range(total_conv_layer)]

            new_items = {"perplexity": perplexity, "measured_rank": measured_rank, "layer_mem": layer_mem, "layer_name": layer_name}
            cfg.hosvd_var["SVD_var"] = SVD_var_measure_perplexity
            cfg.hosvd_var.update(new_items)
            register_measure_perplexity_HOSVD(model, cfg.hosvd_var)
        elif args.with_ASI:
            work_dir = osp.join(osp.join(osp.dirname(work_dir), f'ASI_{osp.basename(work_dir)}/'))
            total_conv_layer = 0
            for cf in cfg.hosvd_var['filter_install']:
                if cf['type'] == 'cbr' or cf['type'] == 'conv': total_conv_layer += 1
                elif cf['type'] == 'resnet_basic_block': total_conv_layer += 2

            perplexity = Perplexity()
            perplexity.load(args.perplexity_pkl)
            best_memory, best_perplexity, best_indices, suitable_ranks = perplexity.find_best_combination(budget=float(args.budget), num_of_finetuned=total_conv_layer)
            new_items = {"rank": suitable_ranks, "layer_names": perplexity.layer_names[-total_conv_layer:]}
            cfg.hosvd_var.update(new_items)
            register_HOSVD_power4_budget_filter(model, cfg.hosvd_var)
        else:
            work_dir = osp.join(osp.join(osp.dirname(work_dir), 'HOSVD/' + str(cfg.hosvd_var['filter_install'][0]['SVD_var'])), osp.basename(work_dir))
            register_HOSVD_filter(model, cfg.hosvd_var)
    elif cfg.svd_var.enable:
        work_dir = osp.join(osp.join(osp.dirname(work_dir), 'SVD/' + str(cfg.hosvd_var['filter_install'][0]['SVD_var'])), osp.basename(work_dir))
        register_SVD_filter(model, cfg.svd_var)



    max_version = get_latest_log_version(work_dir)
    work_dir = osp.join(work_dir, f"version_{max_version}")
    cfg.work_dir = work_dir
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                # dash_line)
    meta['env_info'] = env_info

    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    # logger.info(f'Set random seed to {seed}, '
    #             f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

 

    for layer_path in cfg.freeze_layers:
        active = layer_path[0] == '~'
        if active:
            layer_path = layer_path[1:]
            # logger.info(f"Unfreeze: {layer_path}")
        else:
            # logger.info(f"Freeze: {layer_path}")
            pass
        path_seq = layer_path.split('.')
        target = reduce(getattr, path_seq, model)
        for p in target.parameters():
            p.requires_grad = active
    if args.collect_moment:
        model.moment1 = {}
        model.moment2 = {}
        model.moment_step = {}
        conv_layer_names = []
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                conv_layer_names.append(n)
                m.weight.register_hook(get_moment_logger(model, n))
        # logger.info(f"Layers to be scaned:\n{conv_layer_names}")

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    if args.collect_moment:
        moments = [model.moment1, model.moment2]
        torch.save(moments, osp.join(cfg.work_dir, f"moment_log_{timestamp}"))
    
    if args.measure_perplexity:
        perplexity_object = Perplexity(set_of_epsilons=[cfg.hosvd_var['SVD_var']], # Chỉ xét 1 epsilon mỗi lần test
                                perplexity=[[None for epsilon_idx in range(1)] for layer in range(total_conv_layer)],
                                ranks=[[None for epsilon_idx in range(1)] for layer in range(total_conv_layer)],
                                layer_mems=[[None for epsilon_idx in range(1)] for layer in range(total_conv_layer)])

        for i in range(total_conv_layer):
            perplexity_object.perplexity[i][0] = perplexity[i].item() if isinstance(perplexity[i], torch.Tensor) else perplexity[i]
            perplexity_object.ranks[i][0]      = measured_rank[i]
            perplexity_object.layer_mems[i][0] = layer_mem[i].item() if isinstance(layer_mem[i], torch.Tensor) else layer_mem[i]
        perplexity_object.layer_names = layer_name
        perplexity_object.save(osp.join(osp.dirname(cfg.work_dir), f"perplexity_var{cfg.hosvd_var['SVD_var']}.pkl"))
        # delete_junk_folder(osp.dirname(cfg.work_dir))
        return osp.join(osp.dirname(cfg.work_dir), f"perplexity_var{cfg.hosvd_var['SVD_var']}.pkl")

if __name__ == '__main__':
    print("Train script")
    args = parse_args()
    if args.measure_perplexity:
        perplexity_files = []
        for SVD_var in args.SVD_var.strip().split():
            perplexity_file = main(float(SVD_var))
            perplexity_files.append(perplexity_file)
        merged_perplexity(*perplexity_files)
        # delete_junk_folder(osp.dirname(perplexity_files[0]))
    else:
        main()
        
