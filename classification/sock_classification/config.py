import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data/sock_classification')
parser.add_argument('--n_obj', type=int, default=9)
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

    if args.n_obj == 9:
        args.object_list = [
            'downstairs', 'jump', 'lean_left', 'lean_right',
            'stand', 'stand_toes', 'upstairs', 'walk', 'walk_fast']
        args.n_rounds = [24, 3, 3, 3, 3, 4, 24, 3, 3]
    else:
        raise AssertionError("Unknown number of classes %d" % args.n_obj)


    args.rec_path = 'dump_vest_classification_nObj_%d_subsample_%d' % (
        args.n_obj, args.subsample)

    os.system('mkdir -p ' + args.rec_path)

    return args

