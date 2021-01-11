# import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse
import torch
import tqdm
import numpy as np
import time

from yolov5.train_dt import yolov5
from EfficientObjectDetection.train_new_reward import EfficientOD
from utils import load_filenames, load_dataset, load_dataloader, compute_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24, help="Total batch size for all gpus.")
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--step_batch_size', type=int, default=100)
    parser.add_argument('--save_path', default='save')
    parser.add_argument('--rl_weight', default=None)
    parser.add_argument('--h_detector_weight', default=' ')
    parser.add_argument('--l_detector_weight', default=' ')
    parser.add_argument('--fine_tr', default='config/fine_tr.yaml')
    parser.add_argument('--fine_eval', default='config/fine_eval.yaml')
    parser.add_argument('--coarse_tr', default='config/coarse_tr.yaml')
    parser.add_argument('--coarse_eval', default='config/coarse_eval.yaml')
    parser.add_argument('--EfficientOD', default='config/EfficientOD.yaml')
    parser.add_argument('--split', default=4)
    parser.add_argument('--split_train_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/sample/images')
    parser.add_argument('--split_val_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/train/images')
    parser.add_argument('--split_test_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/test/images')
    parser.add_argument('--original_img_path_train',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/sample/images')
    parser.add_argument('--original_img_path_val',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/train/images')
    parser.add_argument('--original_img_path_test',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/test/images')
    opt = parser.parse_args()

    # training option load from yaml files
    with open(opt.fine_tr) as f:
        fine_tr = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.fine_eval) as f:
        fine_eval = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.coarse_tr) as f:
        coarse_tr = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.coarse_eval) as f:
        coarse_eval = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.EfficientOD) as f:
        efficient_config = yaml.load(f, Loader=yaml.FullLoader)

    epochs = opt.epochs
    bs = opt.batch_size

    fine_detector = yolov5(fine_tr, fine_eval, epochs, bs)
    coarse_detector = yolov5(coarse_tr, coarse_eval, epochs, bs)
    rl_agent = EfficientOD(efficient_config)

    split_train_path = opt.split_train_path
    split_val_path = opt.split_val_path
    split_test_path = opt.split_test_path
    split = opt.split

    original_img_path_train = opt.original_img_path_train
    original_img_path_val = opt.original_img_path_val
    original_img_path_test = opt.original_img_path_test

    assert bs % split == 0, 'batch size should be divided with image split patch size'

    # Training
    for e in range(epochs):
        print('Starting training for %g epochs...' % e)
        train_imgs = load_filenames(split_train_path, split, bs).files_array()
        fine_train_dataset = load_dataset(train_imgs, fine_tr, bs)
        coarse_train_dataset = load_dataset(train_imgs, fine_tr, bs)

        fine_train_loader = load_dataloader(bs, fine_train_dataset)
        coarse_train_loader = load_dataloader(bs, coarse_train_dataset)

        fine_train_nb = len(fine_train_loader)
        coarse_train_nb = len(coarse_train_loader)
        assert fine_train_nb == coarse_train_nb, 'fine & coarse train batch number is not matched'
        nb = fine_train_nb

        for i, (fine_train, coarse_train) in tqdm.tqdm(enumerate(zip(fine_train_loader, coarse_train_loader)),
                                                       total=fine_train_nb):
            fine_detector.train(e, i, nb, fine_train_dataset, fine_train)
            coarse_detector.train(e, i, nb, coarse_train_dataset, coarse_train)

            # result = (source_path, paths[si], mp, mr, map50, nl, stats)
            fine_results = fine_detector.eval(fine_train)
            coarse_results = coarse_detector.eval(coarse_train)

            rl_agent.train(e, i, nb, fine_results, coarse_results, original_img_path_train)

        # Validation
        if e % 1 == 0:

            fine_dataset, coarse_dataset, policies = rl_agent.eval(split_val_path, original_img_path_val)
            fine_results, coarse_results = [], []
            s_time = time.time()
            if len(fine_dataset.tolist()) > 0:
                fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
                fine_val_loader = load_dataloader(bs, fine_val_dataset)
                fine_nb = len(fine_val_loader)
                for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
                    for j in fine_detector.eval(fine_val):
                        fine_results.append(j)

            print('len(coarse_dataset.tolist()): \n', len(coarse_dataset.tolist()))
            if len(coarse_dataset.tolist()) > 0:
                coarse_val_dataset = load_dataset(coarse_dataset, fine_tr, bs)
                coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
                coarse_nb = len(coarse_train_loader)
                for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
                    for j in coarse_detector.eval(coarse_val):
                        coarse_results.append(j)

            map50 = compute_map(fine_results, coarse_results)
            print('Validation mAP: \n', map50)
            print('Time for validation: \n', time.time() - s_time)

            with open('val_result.txt', 'a') as f:
                f.write(str(map50))

            with open('val_policies.txt', 'a') as f:
                f.write(str(policies))

    # Testing
    fine_dataset, coarse_dataset, policies = rl_agent.eval(split_test_path, original_img_path_test)
    fine_results, coarse_results = [], []

    if len(fine_dataset.tolist()) > 0:
        fine_test_dataset = load_dataset(fine_dataset, fine_tr, bs)
        fine_test_loader = load_dataloader(bs, fine_test_dataset)
        fine_nb = len(fine_test_loader)
        for i, fine_test in tqdm.tqdm(enumerate(fine_test_loader), total=fine_nb):
            for j in fine_detector.eval(fine_test):
                fine_results.append(j)

    if len(coarse_dataset.tolist()) > 0:
        coarse_test_dataset = load_dataset(coarse_dataset, fine_tr, bs)
        coarse_test_loader = load_dataloader(bs, coarse_test_dataset)
        coarse_nb = len(coarse_test_loader)
        for i, coarse_test in tqdm.tqdm(enumerate(coarse_test_loader), total=coarse_nb):
            for j in coarse_detector.eval(coarse_test):
                coarse_results.append(j)

    map50 = compute_map(fine_results, coarse_results)
    print('MAP: \n', map50)

    with open('val_result.txt', 'a') as f:
        f.write(str(map50))

    with open('val_policies.txt', 'a') as f:
        f.write(str(policies))