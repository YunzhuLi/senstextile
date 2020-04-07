import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data/vest_letter')
parser.add_argument('--n_obj', type=int, default=10)
parser.add_argument('--n_round', type=int, default=1)
parser.add_argument('--n_train_per_obj', type=int, default=4000)
parser.add_argument('--n_valid_per_obj', type=int, default=500)
parser.add_argument('--n_test_per_obj', type=int, default=1000)
parser.add_argument('--calibrate', type=int, default=0)

parser.add_argument('--input_window_size', type=int, default=45)
parser.add_argument('--skip', type=int, default=2)
parser.add_argument('--subsample', type=int, default=1)

parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=10)

parser.add_argument('--log_per_iter', type=int, default=10000)


def gen_args():
    args = parser.parse_args()

    if args.n_obj == 10:
        args.object_list = [
            'I_ort1', 'I_ort2', 'M_ort1', 'M_ort2', 'M_ort3', 'M_ort4',
            'T_ort1', 'T_ort2', 'T_ort3', 'T_ort4']
    else:
        raise AssertionError("Unknown number of classes %d" % args.n_obj)


    args.rec_path = 'dump_vest_letter_nObj_%d_subsample_%d' % (
        args.n_obj, args.subsample)

    os.system('mkdir -p ' + args.rec_path)

    return args

