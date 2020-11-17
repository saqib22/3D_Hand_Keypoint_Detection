# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import sys
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip
import torchvision.transforms as transforms
from CustomLoader import CustomLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--test_set', type=str, dest='test_set')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
        assert args.annot_subset, "Please set proper annotation subset. Select one of all, human_annot, machine_annot"
    else:
        args.test_set = 'test'
        args.annot_subset = 'all'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set, args.annot_subset)
    tester._make_model()

    preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')
            
            joint_coord_out = out['joint_coord'].cpu().numpy()
            rel_root_depth_out = out['rel_root_depth'].cpu().numpy()
            hand_type_out = out['hand_type'].cpu().numpy()
            inv_trans = out['inv_trans'].cpu().numpy()

            preds['joint_coord'].append(joint_coord_out)
            preds['rel_root_depth'].append(rel_root_depth_out)
            preds['hand_type'].append(hand_type_out)
            preds['inv_trans'].append(inv_trans)
            
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    tester._evaluate(preds)

def custom_main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
        assert args.annot_subset, "Please set proper annotation subset. Select one of all, human_annot, machine_annot"
    else:
        args.test_set = 'test'
        args.annot_subset = 'all'
    
    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set, args.annot_subset)
    tester._make_model()

    test_loader = CustomLoader(transforms.ToTensor(), 'my_hand', 1)

    with torch.no_grad():
        for filename in test_loader.filenames:
            inputs={}
            preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': []}
            img, img_path, inv_trans = test_loader.get_batch_from_txt_files_()
            
            inputs['img'] = torch.reshape(img, (1, 3, 256, 256))

            # forward
            out = tester.model(inputs)

            joint_coord_out = out['joint_coord'].cpu().numpy()
            rel_root_depth_out = out['rel_root_depth'].cpu().numpy()
            hand_type_out = out['hand_type'].cpu().numpy()


            preds['joint_coord'].append(joint_coord_out)
            preds['rel_root_depth'].append(rel_root_depth_out)
            preds['hand_type'].append(hand_type_out)
            
            # evaluate
            preds = {k: np.concatenate(v) for k,v in preds.items()}
            test_loader.visualize(preds, filename, img_path, inv_trans)


if __name__ == "__main__":
    custom_main()
