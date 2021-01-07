# import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse
import torch
import tqdm
import numpy as np

from yolov5.train_dt import yolov5
from EfficientObjectDetection.train_new_reward import EfficientOD
from utils import load_filenames, load_dataset, load_dataloader, compute_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
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

    split_train_path = '/home/SSDD/ICIP21_dataset/split_data_4_0/rl_ver/test/images'
    split_val_path = '/home/SSDD/ICIP21_dataset/split_data_4_0/rl_ver/test/images'
    split_test_path = '/home/SSDD/ICIP21_dataset/split_data_4_0/rl_ver/test/images'
    split = 4

    original_img_path = '/home/SSDD/ICIP21_dataset/origin_data/rl_ver/test/images'

    assert bs % split == 0, 'batch size should be divided with image split patch size'

    # Training
    for e in range(epochs):
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

            rl_agent.train(e, i, nb, fine_results, coarse_results)

        # Validation
        if e % 10 == 0:
            fine_dataset, coarse_dataset, policies = rl_agent.eval(split_val_path, original_img_path)
            fine_results, coarse_results = [], []

            print(len(fine_dataset.tolist()))
            print(len(coarse_dataset.tolist()))

            if len(fine_dataset.tolist()) > 0:
                fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
                fine_val_loader = load_dataloader(bs, fine_val_dataset)
                fine_nb = len(fine_val_loader)
                for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
                    fine_results.append(fine_detector.eval(fine_val))

            if len(coarse_dataset.tolist()) > 0:
                coarse_val_dataset = load_dataset(coarse_dataset, fine_tr, bs)
                coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
                coarse_nb = len(coarse_train_loader)
                for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
                    coarse_results.append(coarse_detector.eval(coarse_val))

            map50 = compute_map(fine_results, coarse_results)
            print('MAP: \n', map50)

    # Testing
    fine_dataset, coarse_dataset, policies = rl_agent.eval(split_test_path, original_img_path)

    fine_test_dataset = load_dataset(fine_dataset, fine_tr, bs)
    coarse_test_dataset = load_dataset(coarse_dataset, fine_tr, bs)

    # if len(fine_test_dataset.tolist()) > 0:
    #     fine_test_loader = load_dataloader(bs, fine_test_dataset)
    #
    #
    # if len(fine_test_dataset.tolist()) > 0:
    #     coarse_test_loader = load_dataloader(bs, coarse_test_dataset)
    #
    # fine_nb = len(fine_test_loader)
    # coarse_nb = len(coarse_test_loader)
    #
    # fine_results, coarse_results = [], []
    # for i, fine_test in tqdm.tqdm(enumerate(fine_test_loader), total=fine_nb):
    #     fine_results.append(fine_detector.eval(fine_test))
    #
    # for i, coarse_test in tqdm.tqdm(enumerate(coarse_test_loader), total=coarse_nb):
    #     coarse_results.append(coarse_detector.eval(coarse_test))
    #
    # map50 = compute_map(fine_results, coarse_results)
    # print('MAP: \n', map50)
