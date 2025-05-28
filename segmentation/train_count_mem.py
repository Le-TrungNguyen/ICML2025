# File used to count memory
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

from tools.perplexity import Perplexity

from custom_op.register import register_filter, register_HOSVD_filter, register_SVD_filter, attach_hook_for_base_conv, register_HOSVD_power4_budget_filter
from functools import reduce
import torch.nn as nn

def get_all_conv_with_name(model):
    conv_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d): # Ngầm log lại hết cả các lớp kế thừa luôn
            conv_layers.append({"path": name, "type": "conv"})
    return conv_layers


def count_elements(elements):
    count = 0
    for element in elements:
        element_type = element.get('type')
        if element_type in ['cbr', 'conv']:
            count += 1
        elif element_type == 'resnet_basic_block':
            count += 2
    return count

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
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


def get_memory(link, args, unit='Byte'):
    cfg = Config.fromfile(args.config)
    if cfg.get("gradient_filter", None) is None: cfg.gradient_filter = dict(enable=False)
    if cfg.get("base", None) is None: cfg.base = dict(enable=False)
    if cfg.get("hosvd_var", None) is None: cfg.hosvd_var = dict(enable=False)
    if cfg.get("svd_var", None) is None: cfg.svd_var = dict(enable=False)

    if (cfg.gradient_filter["enable"] == False and 
        cfg.hosvd_var["enable"] == False and
        cfg.svd_var["enable"] == False and
        cfg.base["enable"] == False):
        cfg.full = dict(enable=True)
    else:
        cfg.full = dict(enable=False)

    if cfg.get("freeze_layers", None) is None: cfg.freeze_layers = []

    if cfg.hosvd_var.enable or cfg.svd_var.enable:
        if args.with_ASI:
            work_dir = './runs'
            postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
            log_name = osp.splitext(osp.basename(args.config))[0] + postfix
            if args.work_dir is not None:
                work_dir = args.work_dir
            work_dir = osp.join(work_dir, log_name)
        else:
            args.load_from = link

            work_dir = osp.dirname(link)
            work_dir = osp.join(work_dir, 'mem_log')
            work_dir = osp.join(work_dir, 'delete')
    
    else: # VanillaBP or GradientFilter
        work_dir = './runs'
        postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
        log_name = osp.splitext(osp.basename(args.config))[0] + postfix
        if args.work_dir is not None:
            work_dir = args.work_dir
        work_dir = osp.join(work_dir, log_name)
        
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

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    hook = {}

    if cfg.gradient_filter.enable:
        work_dir = osp.join(osp.join(osp.dirname(work_dir), 'gradient_filter'), osp.basename(work_dir))
        # logger.info("Install Gradient Filter")
        register_filter(model, cfg.gradient_filter, hook)

    elif cfg.base.enable:
        work_dir = osp.join(osp.join(osp.dirname(work_dir), 'base'), osp.basename(work_dir))
        attach_hook_for_base_conv(model, cfg.base, hook)

    elif cfg.full.enable:

        work_dir = osp.join(osp.join(osp.dirname(work_dir), 'full'), osp.basename(work_dir))
        cfg.full['filter_install'] = get_all_conv_with_name(model)
        attach_hook_for_base_conv(model, cfg.full, hook)
    
    elif cfg.hosvd_var.enable:
        if args.with_ASI:
            work_dir = osp.join(osp.join(osp.dirname(work_dir), f'ASI_{osp.basename(work_dir)}/'))
            total_conv_layer = 0
            for cf in cfg.hosvd_var['filter_install']:
                if cf['type'] == 'cbr' or cf['type'] == 'conv': total_conv_layer += 1
                elif cf['type'] == 'resnet_basic_block': total_conv_layer += 2

            perplexity = Perplexity()
            perplexity.load(args.perplexity_pkl)
            best_memory, best_perplexity, best_indices, suitable_ranks = perplexity.find_best_combination(budget=float(args.budget), num_of_finetuned=total_conv_layer)


            print("Best memory là: ", best_memory, best_indices)
            # Đo sẽ khác vì pretrained data dùng size 512x1024, còn data finetune thì dùng 512x512

            new_items = {"rank": suitable_ranks, "layer_names": perplexity.layer_names[-total_conv_layer:]}
            cfg.hosvd_var.update(new_items)
            register_HOSVD_power4_budget_filter(model, cfg.hosvd_var, hook)

        else:
            num_of_finetune = count_elements(cfg.hosvd_var['filter_install'])
            cfg.hosvd_var["k_hosvd"] = [[], [], [], [], [], []]
            # logger.info("Install HOSVD with variance")
            register_HOSVD_filter(model, cfg.hosvd_var)

    elif cfg.svd_var.enable:
        num_of_finetune = count_elements(cfg.svd_var['filter_install'])
        cfg.svd_var["svd_size"] = []
        # logger.info("Install SVD with variance")
        register_SVD_filter(model, cfg.svd_var)
    

    if cfg.gradient_filter.enable or cfg.base.enable or args.with_ASI or cfg.full.enable:
        work_dir = osp.join(work_dir, 'delete')

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
    #             dash_line)
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
        # else:
            # logger.info(f"Freeze: {layer_path}")
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

    # logger.info(model)

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
        validate=False,
        timestamp=timestamp,
        meta=meta)

    if args.collect_moment:
        moments = [model.moment1, model.moment2]
        torch.save(moments, osp.join(cfg.work_dir, f"moment_log_{timestamp}"))

    # get_activation_mem:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from math import ceil
    from custom_op.conv2d.conv_avg import Conv2dAvg
    from segmentation.custom_op.conv2d.conv_ASI import Conv2d_ASI
    from segmentation.custom_op.compression.hosvd_subspace_iteration import hosvd_subspace_iteration
    num_element = 0
    num_flops_fw = 0
    num_flops_bw = 0
    element_size=4
    if cfg.gradient_filter.enable or cfg.base.enable or args.with_ASI or cfg.full.enable:
        for layer_index, name in enumerate(hook):
            input_size = hook[name].input_size
            B, C, H, W = input_size
            _, C_prime, H_prime, W_prime = hook[name].output_size
            K_H, K_W = hook[name].module.kernel_size
            
            if isinstance(hook[name].module, Conv2dAvg):
                stride = hook[name].module.stride
                x_h, x_w = input_size[-2:]
                output_size = hook[name].output_size
                h, w = output_size[-2:]

                filt_radius = hook[name].special_param

                p_h, p_w = ceil(h / filt_radius), ceil(w / filt_radius)
                x_order_h, x_order_w = filt_radius * stride[0], filt_radius * stride[1]
                x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

                x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
                x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

                num_element += int(input_size[0] * input_size[1] * x_sum_height * x_sum_width)

                ############### FLOPs
                forward_overhead = (H/filt_radius)*(W/filt_radius)*(filt_radius**2-1)*C*B
                conv_forward = K_H*K_W*C*C_prime*B*H*W

                gradient_filering_overhead = B*C_prime*(h/filt_radius)*(w/filt_radius)
                weight_sum_overhead = C_prime*C*(K_H*K_W-1)
                scalar_mult_backward = B*C_prime*(h/filt_radius)*(w/filt_radius)
                frobenius_backward = K_H*K_W*C*C_prime*B*H*W

                num_flops_fw += forward_overhead + conv_forward
                num_flops_bw += gradient_filering_overhead + weight_sum_overhead + scalar_mult_backward + frobenius_backward

            elif isinstance(hook[name].module, Conv2d_ASI):
                S, u_list = hosvd_subspace_iteration(hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=suitable_ranks[layer_index])
                K0, K1, K2, K3 = S.shape
                
                num_element += S.numel() + sum(u.numel() for u in u_list)
                #################  Tính flops
                fw_overhead = 0
                for K in S.shape:
                    fw_overhead += 2*B*C*H*W*K + K**3
                vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B
                bw = (K0*C_prime*H_prime*W_prime*B + H*K0*K1*K2*K3 + H*W*K0*K1*K3 + C_prime*K0*K1*K_H*K_W*H_prime*W_prime + C_prime*C*K_H*K_W*K1)

                num_flops_fw += fw_overhead + vanilla_fw
                num_flops_bw += bw
                
            elif isinstance(hook[name].module, nn.modules.conv.Conv2d):
                num_element += input_size[0]*input_size[1]*input_size[2]*input_size[3]

                vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B
                vanilla_bw = (K_H*K_W*C*C_prime*H_prime*W_prime)*B

                num_flops_fw += vanilla_fw
                num_flops_bw += vanilla_bw

        if unit == "Byte":
            mem = str(num_element*element_size)
        if unit == "MB":
            mem = str((num_element*element_size)/(1024*1024))
        elif unit == "KB":
            mem = str((num_element*element_size)/(1024))
        else:
            raise ValueError("Unit is not suitable")
        
        with open(os.path.join(osp.dirname(work_dir), f'activation_memory_{unit}.log'), "w") as file:
                file.write(f"Activation memory is {mem} {unit}\n")
        
        with open(os.path.join(osp.dirname(work_dir), f'total_FLOPs.log'), "w") as file:
                file.write(f"Forward: {num_flops_fw}\n")
                file.write(f"Backward: {num_flops_bw}\n")
                file.write(f"Total FLOPs: {num_flops_fw + num_flops_bw}\n")

    elif cfg.svd_var.enable:
        svd_size_tensor= torch.stack(cfg.svd_var['svd_size']).t().float() # Shape: shape of svd components, number of batchs * num_of_finetune
        svd_size_tensor = svd_size_tensor.view(3, -1, num_of_finetune) # Shape: shape of svd components, number of batchs, num_of_finetune
        svd_size_tensor = svd_size_tensor.permute(2, 1, 0) # Shape: num_of_finetune, number of batchs, shape of svd components
        num_element_all = torch.mean(svd_size_tensor[:, :, 0] * svd_size_tensor[:, :, 1] + svd_size_tensor[:, :, 1] * svd_size_tensor[:, :, 2], dim=1)
        num_element = torch.sum(num_element_all)

        
        if unit == "Byte":
            mem = num_element*element_size
        if unit == "MB":
            mem = (num_element*element_size)/(1024*1024)
        elif unit == "KB":
            mem = (num_element*element_size)/(1024)
        else:
            raise ValueError("Unit is not suitable")
        
        with open(os.path.join(osp.dirname(work_dir), f'activation_memory_{unit}.log'), "a") as file:
            file.write(osp.basename(link) + "\t" + str(float(mem)) + "\n")

    elif cfg.hosvd_var.enable:
        k_hosvd_tensor = torch.tensor(cfg.hosvd_var["k_hosvd"][:4], device=device).float() # Shape: ranks, number of batch * num_of_finetune
        k_hosvd_tensor = k_hosvd_tensor.view(4, -1, num_of_finetune) # Shape: ranks, number of batch, num_of_finetune
        k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: num_of_finetune, number of batch, ranks

        input_shapes = torch.tensor(cfg.hosvd_var["k_hosvd"][4], device=device).reshape(-1, num_of_finetune, 4) # Shape: num of batch, num_of_finetune, activation map raw shape
        input_shapes = input_shapes.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, activation map raw shape

        output_shapes = torch.tensor(cfg.hosvd_var["k_hosvd"][5], device=device).reshape(-1, num_of_finetune, 4) # Shape: (#batch, num_of_finetune, 4 shapes)
        output_shapes = output_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shapes)

        num_element_all = torch.sum(
            k_hosvd_tensor[:, :, 0] * k_hosvd_tensor[:, :, 1] * k_hosvd_tensor[:, :, 2] * k_hosvd_tensor[:, :, 3]
            + k_hosvd_tensor[:, :, 0] * input_shapes[:, :, 0]
            + k_hosvd_tensor[:, :, 1] * input_shapes[:, :, 1]
            + k_hosvd_tensor[:, :, 2] * input_shapes[:, :, 2]
            + k_hosvd_tensor[:, :, 3] * input_shapes[:, :, 3],
            dim=1
        )
        
        num_element = torch.sum(num_element_all) / k_hosvd_tensor.shape[1]

        if unit == "Byte":
            mem = num_element*element_size
        if unit == "MB":
            mem = (num_element*element_size)/(1024*1024)
        elif unit == "KB":
            mem = (num_element*element_size)/(1024)
        else:
            raise ValueError("Unit is not suitable")
        with open(os.path.join(osp.dirname(work_dir), f'activation_memory_{unit}.log'), "a") as file:
            file.write(osp.basename(link) + "\t" + str(float(mem)) + "\n")

        #################### Tính FLOPs
        num_flops_fw = 0
        num_flops_bw = 0

        layer_names = []
        for layer_idx, layer in enumerate(cfg.hosvd_var['filter_install']):
            if layer['type'] == 'resnet_basic_block':
                layer_names.append(layer['path'] + ".conv1")
                layer_names.append(layer['path'] + ".conv2")
            else:
                layer_names.append(layer['path'])

        for layer_idx, name in enumerate(layer_names):
            path_seq = name.split('.')
            target = reduce(getattr, path_seq, model)
            K_H, K_W = target.kernel_size

            K0, K1, K2, K3 = k_hosvd_tensor[layer_idx, :, 0], k_hosvd_tensor[layer_idx, :, 1], k_hosvd_tensor[layer_idx, :, 2], k_hosvd_tensor[layer_idx, :, 3]
            B, C, H, W = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2], input_shapes[layer_idx, :, 3] # Shape: Mỗi phần tử (#batch, 1)
            C_prime, H_prime, W_prime = output_shapes[layer_idx, :, 1], output_shapes[layer_idx, :, 2], output_shapes[layer_idx, :, 3]
            # Tính FLOPs:
            # FW
            term1 = torch.max(B, C * H * W) ** 2 * torch.min(B, C * H * W)
            term2 = torch.max(C, B * H * W) ** 2 * torch.min(C, B * H * W)
            term3 = torch.max(H, B * C * W) ** 2 * torch.min(H, B * C * W)
            term4 = torch.max(W, B * C * H) ** 2 * torch.min(W, B * C * H)

            vanilla_fw = (K_H*K_W*C_prime*C*H*W)*B # Shape: (#batch, 1)

            hosvd_neurips_fw_overhead = (term1 + term2 + term3 + term4)
            hosvd_neurips_fw = hosvd_neurips_fw_overhead + vanilla_fw
            num_flops_fw += hosvd_neurips_fw # Shape: (#batch, 1)
            # BW
            hosvd_neurips_FLOPs_bw = (K0*C_prime*H_prime*W_prime*B + H*K0*K1*K2*K3 + H*W*K0*K1*K3 + C_prime*K0*K1*K_H*K_W*H_prime*W_prime + C_prime*C*K_H*K_W*K1)
            num_flops_bw += hosvd_neurips_FLOPs_bw # Shape: (#batch, 1)

        num_flops_bw = (torch.sum(num_flops_bw, dim=0)/num_flops_bw.shape[0]).item()
        num_flops_fw = (torch.sum(num_flops_fw, dim=0)/num_flops_fw.shape[0]).item()
        with open(os.path.join(osp.dirname(work_dir), f'fw_FLOPs.log'), "a") as file:
            file.write(f"Forward FLOPs (same for all epoch): {num_flops_fw}\n")
        with open(os.path.join(osp.dirname(work_dir), f'bw_FLOPs.log'), "a") as file:
            file.write(osp.basename(link) + "\t" + str(num_flops_bw) + "\n")
        with open(os.path.join(osp.dirname(work_dir), f'total_FLOPs.log'), "a") as file:
            file.write(osp.basename(link) + "\t" + str((num_flops_fw + num_flops_bw)) + "\n")
    
    print("Memory and FLOPs are logged at ", osp.dirname(work_dir)) 
    


import re
def find_checkpoint():
    args = parse_args()

    experiment_dir = './runs'
    postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
    log_name = osp.splitext(osp.basename(args.config))[0] + postfix
    experiment_dir = osp.join(experiment_dir, log_name)
    cfg = Config.fromfile(args.config)

    if cfg.get("gradient_filter", None) is None: cfg.gradient_filter = dict(enable=False)
    if cfg.get("hosvd_var", None) is None: cfg.hosvd_var = dict(enable=False)
    if cfg.get("svd_var", None) is None: cfg.svd_var = dict(enable=False)
    if cfg.get("base", None) is None:cfg.base = dict(enable=False)

    if (cfg.gradient_filter["enable"] == False and 
        cfg.hosvd_var["enable"] == False and
        cfg.svd_var["enable"] == False and
        cfg.base["enable"] == False):
        cfg.full = dict(enable=True)
    else:
        cfg.full = dict(enable=False)

    if args.with_ASI:
        experiment_dir = osp.join(osp.join(osp.dirname(experiment_dir), f'ASI_{osp.basename(experiment_dir)}/'))
    elif cfg.hosvd_var.enable:
        experiment_dir = osp.join(osp.join(osp.dirname(experiment_dir), 'HOSVD/' + str(cfg.hosvd_var['filter_install'][0]['SVD_var'])), osp.basename(experiment_dir))
    elif cfg.svd_var.enable:
        experiment_dir = osp.join(osp.join(osp.dirname(experiment_dir), 'SVD/' + str(cfg.svd_var['filter_install'][0]['SVD_var'])), osp.basename(experiment_dir))

    if cfg.base.enable or cfg.gradient_filter.enable or args.with_ASI or cfg.full.enable:
        get_memory(None, args, unit='MB')
    elif cfg.hosvd_var.enable or cfg.svd_var.enable:
        checkpoints = []
        def process_directory(current_directory):
            for entry in sorted(os.listdir(current_directory)):
                entry_path = os.path.join(current_directory, entry)
                if 'iter' in entry:
                    checkpoints.append(entry_path)
                elif os.path.isdir(entry_path):
                    process_directory(entry_path)

        process_directory(experiment_dir)

        def extract_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'iter_(\d+)', filename)
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=extract_number)

        for checkpoint in checkpoints:
            get_memory(checkpoint, args, unit='MB')
        

if __name__ == '__main__':
    find_checkpoint()
