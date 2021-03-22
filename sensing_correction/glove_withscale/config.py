import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data')
parser.add_argument('--knit_name', default='')
parser.add_argument('--date', default='2019-09-02')
parser.add_argument('--n_point', type=int, default=-1)
parser.add_argument('--n_round', type=int, default=-1)

'''
train options
'''
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--nf_hidden', type=int, default=16)
parser.add_argument('--obs_window', type=int, default=15)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_valid_ratio', type=float, default=0.9)

parser.add_argument('--height', type=float, default=32)
parser.add_argument('--width', type=float, default=32)

parser.add_argument('--log_per_iter', type=int, default=2000)
parser.add_argument('--ckp_per_iter', type=int, default=800)

parser.add_argument('--debug', type=int, default=0)

'''
eval & resume options
'''
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--iter', type=int, default=-1)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--eval_list', default='')
parser.add_argument('--vis', type=int, default=1)
parser.add_argument('--vis_scale', type=int, default=0)
parser.add_argument('--store', type=int, default=0)

'''
ablation
'''
parser.add_argument('--position_bias', type=int, default=0,
                    help="whether to use position-based bias")
parser.add_argument('--lam_recon', type=float, default=0.,
                    help="weight on reconstruction loss")
parser.add_argument('--scale_factor', type=float, default=1.0,
                    help="internal scaling factor in calibration network")

'''
super resolution
'''
parser.add_argument('--superres', type=float, default=1.0)


def gen_args():
    args = parser.parse_args()

    args.data_path_prefix = os.path.join(args.data_path, args.knit_name)

    if args.superres == 1.:
        args.vis_path = os.path.join('dump_' + args.knit_name, 'vis')
        args.ckp_path = os.path.join('dump_' + args.knit_name, 'ckp')
    else:
        args.vis_path = os.path.join('dump_' + args.knit_name, 'vis_res%.1f' % args.superres)
        args.ckp_path = os.path.join('dump_' + args.knit_name, 'ckp_res%.1f' % args.superres)

    assert(args.obs_window % 2 == 1)

    args.vis_path += '_scale%.2f' % args.scale_factor
    args.ckp_path += '_scale%.2f' % args.scale_factor

    args.vis_path += '_lamRec%.2f' % args.lam_recon
    args.ckp_path += '_lamRec%.2f' % args.lam_recon

    if args.position_bias == 1:
        args.vis_path += '_pBias'
        args.ckp_path += '_pBias'

    os.system('mkdir -p ' + args.vis_path)
    os.system('mkdir -p ' + args.ckp_path)

    return args

