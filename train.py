# import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse
import torch
import tqdm
import numpy as np
import time
import os
import utils
import copy
import easydict

# torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

from yolov5.train_dt import yolov5
from EfficientObjectDetection.train_new_reward import EfficientOD
from utils import load_filenames, load_dataset, load_dataloader, compute_map, convert_yolo2coco, label2idx, label_matching, reduce_dict, make_results, make_results_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=40, help="Total batch size for all gpus.")
    parser.add_argument('--RL_batch_size', type=int, default=40, help="Total batch size for all gpus.")
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--split', type=int, default=4)
    parser.add_argument('--split_format', nargs='*')
    parser.add_argument('--split_train_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_80/rl_ver/train/images')
    parser.add_argument('--split_val_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_80/rl_ver/val/images')
    parser.add_argument('--split_test_path',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_80/rl_ver/test/images')
    parser.add_argument('--original_img_path_train',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/train/images')
    parser.add_argument('--original_img_path_val',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/val/images')
    parser.add_argument('--original_img_path_test',
                        default='/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/test/images')
    parser.add_argument('--fine_img_size', nargs='+', type=int, default=[480, 480])
    parser.add_argument('--coarse_img_size', nargs='+', type=int, default=[96, 96])
    parser.add_argument('--model', default='yolov5')
    parser.add_argument('--RL_train_start_epoch', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='save')
    opt = parser.parse_args()

    fine_tr = easydict.EasyDict({
        "cfg": "yolov5/models/yolov5x_custom.yaml",
        "data": "yolov5/data/HRSID_800_od.yaml",
        "hyp": '',
        "img_size": opt.fine_img_size,
        "resume": False,
        "bucket": '',
        "cache_images": False,
        "weights": '',
        "save_path": opt.save_path,
        "name": "yolo_fine",
        "device": opt.device,
        "multi_scale": False,
        "single_cls": True,
        "sync_bn": False,
        "local_rank": -1
    })

    fine_eval = easydict.EasyDict({
        "data": "yolov5/data/HRSID_800_rl.yaml",
        "batch_size": opt.batch_size,
        "conf_thres": 0.001,
        "iou_thres": 0.6,  # for NMS
        "augment": False
    })

    coarse_tr = easydict.EasyDict({
        "cfg": "yolov5/models/yolov5x_custom.yaml",
        "data": "yolov5/data/HRSID_800_od.yaml",
        "hyp": '',
        "img_size": opt.coarse_img_size,
        "resume": False,
        "bucket": '',
        "cache_images": False,
        "weights": "",
        "save_path": opt.save_path,
        "name": "yolo_coarse",
        "device": opt.device,
        "multi_scale": False,
        "single_cls": True,
        "sync_bn": False,
        "local_rank": -1
    })

    coarse_eval = easydict.EasyDict({
        "data": "yolov5/data/HRSID_800_rl.yaml",
        "batch_size": opt.batch_size,
        "conf_thres": 0.001,
        "iou_thres": 0.6,  # for NMS
        "augment": False
    })


    efficient_config = easydict.EasyDict({
        "gpu_id": opt.device,
        "lr": 1e-3,
        "cv_dir": opt.save_path,
        "save_name": 'yolo',
        "batch_size": opt.batch_size,
        "step_batch_size": opt.RL_batch_size,
        "img_size": opt.fine_img_size,
        "num_workers": 0,
        "parallel": False,
        "alpha": 0.8,
        "beta": 0.1,
        "sigma": 0.5,
        "load": None,
        "split": opt.split,
        "split_format": opt.split_format,
        "train_start": opt.RL_train_start_epoch
    })

    if opt.model == 'yolov5':
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
        train_imgs = load_filenames(split_train_path, split, bs).files_array()
        fine_train_dataset = load_dataset(train_imgs, fine_tr, bs)
        coarse_train_dataset = load_dataset(train_imgs, coarse_tr, bs)

        for e in range(epochs):
            print('Starting training for %g epochs...' % e)

            fine_train_loader = load_dataloader(bs, fine_train_dataset)
            coarse_train_loader = load_dataloader(bs, coarse_train_dataset)

            fine_train_nb = len(fine_train_loader)
            coarse_train_nb = len(coarse_train_loader)
            assert fine_train_nb == coarse_train_nb, 'fine & coarse train batch number is not matched'
            nb = fine_train_nb

            for i, (fine_train, coarse_train) in tqdm.tqdm(enumerate(zip(fine_train_loader, coarse_train_loader)),
                                                           total=fine_train_nb):
                fine_detector.train(e, i + 1, nb, fine_train_dataset, fine_train)
                coarse_detector.train(e, i + 1, nb, coarse_train_dataset, coarse_train)

                # result = (source_path, paths[si], mp, mr, map50, nl, stats)
                fine_results = fine_detector.eval(fine_train)
                coarse_results = coarse_detector.eval(coarse_train)

                rl_agent.train(e, i + 1, nb, fine_results, coarse_results, original_img_path_train)

            # Validation
            if e % 1 == 0:

                fine_dataset, coarse_dataset, policies = rl_agent.eval(split_val_path, original_img_path_val)
                fine_results, coarse_results = [], []
                s_time = time.time()
                print('len(fine_dataset.tolist()): \n', len(fine_dataset.tolist()))
                if len(fine_dataset.tolist()) > 0:
                    fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
                    fine_val_loader = load_dataloader(bs, fine_val_dataset)
                    fine_nb = len(fine_val_loader)
                    for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
                        for j in fine_detector.test(fine_val):
                            fine_results.append(j)

                print('len(coarse_dataset.tolist()): \n', len(coarse_dataset.tolist()))
                if len(coarse_dataset.tolist()) > 0:
                    coarse_val_dataset = load_dataset(coarse_dataset, coarse_tr, bs)
                    coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
                    coarse_nb = len(coarse_train_loader)
                    for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
                        for j in coarse_detector.test(coarse_val):
                            coarse_results.append(j)

                map50 = compute_map(fine_results, coarse_results)
                print('Validation mAP: \n', map50)
                print('Validation find mAP: \n', compute_map(fine_results, []))
                print('Validation coarse mAP: \n', compute_map([], coarse_results))
                print('Time for validation: \n', time.time() - s_time)

                with open(opt.save_path + '/val_result.txt', 'a') as f:
                    f.write(str(map50) + '\n')

                eff = 0
                for i in policies:
                    eff += int(i)
                with open(opt.save_path + '/val_policies.txt', 'a') as f:
                    f.write(str(eff / len(policies)) + '\n')

        # Testing
        fine_dataset, coarse_dataset, policies = rl_agent.eval(split_test_path, original_img_path_test)
        fine_results, coarse_results = [], []

        if len(fine_dataset.tolist()) > 0:
            fine_test_dataset = load_dataset(fine_dataset, fine_tr, bs)
            fine_test_loader = load_dataloader(bs, fine_test_dataset)
            fine_nb = len(fine_test_loader)
            for i, fine_test in tqdm.tqdm(enumerate(fine_test_loader), total=fine_nb):
                for j in fine_detector.test(fine_test):
                    fine_results.append(j)

        if len(coarse_dataset.tolist()) > 0:
            coarse_test_dataset = load_dataset(coarse_dataset, coarse_tr, bs)
            coarse_test_loader = load_dataloader(bs, coarse_test_dataset)
            coarse_nb = len(coarse_test_loader)
            for i, coarse_test in tqdm.tqdm(enumerate(coarse_test_loader), total=coarse_nb):
                for j in coarse_detector.test(coarse_test):
                    coarse_results.append(j)

        map50 = compute_map(fine_results, coarse_results)
        print('Test mAP: \n', map50)
        print('Test find mAP: \n', compute_map(fine_results, []))
        print('Test coarse mAP: \n', compute_map([], coarse_results))

        with open(opt.save_path + '/test_result.txt', 'a') as f:
            f.write(str(map50) + '\n')

        eff = 0
        for i in policies:
            eff += int(i)
        with open(opt.save_path + '/test_policies.txt', 'a') as f:
            f.write(str(eff / len(policies)) + '\n')

    elif opt.model == 'faster_rcnn':
        # GPU Device
        gpu_id = opt.device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        use_cuda = torch.cuda.is_available()
        print("GPU device ", use_cuda)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        efficient_config['load'] = None  # bug fix

        epochs = opt.epochs
        bs = opt.batch_size
        rl_agent = EfficientOD(efficient_config)

        split_train_path = opt.split_train_path
        split_val_path = opt.split_val_path
        split_test_path = opt.split_test_path
        split = opt.split

        original_img_path_train = opt.original_img_path_train
        original_img_path_val = opt.original_img_path_val
        original_img_path_test = opt.original_img_path_test

        assert bs % split == 0, 'batch size should be divided with image split patch size'

        num_classes = 2

        fine_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes,
                                                                          pretrained_backbone=False)
        coarse_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes,
                                                                            pretrained_backbone=False)

        # # # # replace the classifier with a new one, that has
        # # # # num_classes which is user-defined

        # # # get number of input features for the classifier
        fine_in_features = fine_model.roi_heads.box_predictor.cls_score.in_features
        coarse_in_features = coarse_model.roi_heads.box_predictor.cls_score.in_features

        # # # replace the pre-trained head with a new one
        fine_model.roi_heads.box_predictor = FastRCNNPredictor(fine_in_features, num_classes)
        coarse_model.roi_heads.box_predictor = FastRCNNPredictor(coarse_in_features, num_classes)

        for fine_p, coarse_p in zip(fine_model.parameters(), coarse_model.parameters()):
            fine_p.requires_grad = True
            coarse_p.requires_grad = True

        fine_model.to(device)
        coarse_model.to(device)

        # Optimizer
        fine_params = [p for p in fine_model.parameters() if p.requires_grad]
        coarse_params = [p for p in coarse_model.parameters() if p.requires_grad]

        fine_optim = torch.optim.SGD(fine_params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        coarse_optim = torch.optim.SGD(coarse_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        fine_lr_scheduler = torch.optim.lr_scheduler.StepLR(fine_optim, step_size=50)
        coarse_lr_scheduler = torch.optim.lr_scheduler.StepLR(coarse_optim, step_size=50)

        # label이 없더라도 loader에 image 생성
        train_imgs = load_filenames(split_train_path, split, bs).files_array()
        fine_train_dataset = load_dataset(train_imgs, fine_tr, bs)
        coarse_train_dataset = load_dataset(train_imgs, coarse_tr, bs)
        for e in range(epochs):
            print('Starting training for %g epochs...' % e)

            fine_train_loader = load_dataloader(bs, fine_train_dataset)
            coarse_train_loader = load_dataloader(bs, coarse_train_dataset)

            fine_train_nb = len(fine_train_loader)
            coarse_train_nb = len(coarse_train_loader)
            assert fine_train_nb == coarse_train_nb, 'fine & coarse train batch number is not matched'
            nb = fine_train_nb

            # Logger
            fine_metric_logger = utils.MetricLogger(delimiter="  ")
            fine_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            coarse_metric_logger = utils.MetricLogger(delimiter="  ")
            coarse_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            fine_header = 'Fine Epoch: [{}]'.format(e)
            coarse_header = 'Coarse Epoch: [{}]'.format(e)

            #     # warmup
            fine_lr_scheduler = None
            corase_lr_scheduler = None
            if e == 0:
                warmup_factor = 1. / 1000
                warmup_iters = min(1000, fine_train_nb - 1)
                fine_lr_scheduler = utils.warmup_lr_scheduler(fine_optim, warmup_iters, warmup_factor)
                coarse_lr_scheduler = utils.warmup_lr_scheduler(coarse_optim, warmup_iters, warmup_factor)

            for i, (fine_train, coarse_train) in enumerate(zip(fine_train_loader, coarse_train_loader)):
                # train
                fine_model.train()
                coarse_model.train()
                #### fine train ###
                # Label mathching
                fine_imgs, fine_labels = label_matching(fine_train, device)
                fine_imgs = fine_imgs.to(device) / 255.

                ## train: img normalization --> not, zerodivision err
                fine_loss_dict = fine_model(fine_imgs, copy.deepcopy(fine_labels))
                fine_losses = sum(loss for loss in fine_loss_dict.values())
                fine_loss_dict_reduced = reduce_dict(fine_loss_dict)
                fine_loss_reduced = sum(loss for loss in fine_loss_dict_reduced.values())
                fine_loss_val = fine_loss_reduced.item()

                # optimizer
                fine_optim.zero_grad()
                fine_losses.backward()
                fine_optim.step()

                if fine_lr_scheduler is not None:
                    fine_lr_scheduler.step()

                fine_metric_logger.update(loss=fine_loss_reduced, **fine_loss_dict_reduced)
                fine_metric_logger.update(lr=fine_optim.param_groups[0]["lr"])

                if i % 200 == 0:
                    space_fmt = ':' + str(len(str(fine_train_nb))) + 'd'
                    log_msg = fine_metric_logger.delimiter.join([fine_header, '[{0' + space_fmt + '}/{1}]', '{meters}'])
                    print(log_msg.format(i, fine_train_nb, meters=str(fine_metric_logger)))

                ### coarse train ###
                # Label mathching
                coarse_imgs, coarse_labels = label_matching(coarse_train, device)
                coarse_imgs = coarse_imgs.to(device) / 255.

                ## train: img normalization --> not, zerodivision err
                coarse_loss_dict = coarse_model(coarse_imgs, copy.deepcopy(coarse_labels))
                coarse_losses = sum(loss for loss in coarse_loss_dict.values())

                # utils
                coarse_loss_dict_reduced = reduce_dict(coarse_loss_dict)
                coarse_loss_reduced = sum(loss for loss in coarse_loss_dict_reduced.values())
                coarse_loss_val = coarse_loss_reduced.item()

                # optimizer
                coarse_optim.zero_grad()
                coarse_losses.backward()
                coarse_optim.step()

                if coarse_lr_scheduler is not None:
                    coarse_lr_scheduler.step()

                coarse_metric_logger.update(loss=coarse_loss_reduced, **coarse_loss_dict_reduced)
                coarse_metric_logger.update(lr=fine_optim.param_groups[0]["lr"])

                # if i % opt.print_freq == 0:
                if i % 200 == 0:
                    space_fmt = ':' + str(len(str(fine_train_nb))) + 'd'
                    log_msg = coarse_metric_logger.delimiter.join(
                        [coarse_header, '[{0' + space_fmt + '}/{1}]', '{meters}'])
                    print(log_msg.format(i, fine_train_nb, meters=str(coarse_metric_logger)))

                ## train eval
                # result = (source_path, paths[si], mp, mr, map50, nl, stats)
                # file_name, od_file_dir, mp=0(skip), ma=0(skip), map50(will be soon), objnum, stat
                # stat = 4

                # make_results(model, dataset, device)
                fine_results = make_results(fine_model, fine_train, device)
                coarse_results = make_results(coarse_model, coarse_train, device)

                # conf_thresh=0.001 / iou_thres=0.6
                rl_agent.train(e, i, nb, fine_results, coarse_results, original_data_path=original_img_path_train)

                ## Validation
            if e % 1 == 0:
                fine_dataset, coarse_dataset, policies = rl_agent.eval(split_val_path, original_img_path_val)
                fine_results, coarse_results = [], []

                if len(fine_dataset.tolist()) > 0:
                    fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
                    fine_val_loader = load_dataloader(bs, fine_val_dataset)
                    fine_nb = len(fine_val_loader)
                    for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
                        fine_results += make_results_test(fine_model, fine_val, device)

                if len(coarse_dataset.tolist()) > 0:
                    coarse_val_dataset = load_dataset(coarse_dataset, fine_tr, bs)
                    coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
                    coarse_nb = len(coarse_train_loader)
                    for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
                        coarse_results += make_results(coarse_model, coarse_val, device)

                map50 = compute_map(fine_results, coarse_results)
                print('Validation MAP: \n', map50)
                print('Validation find mAP: \n', compute_map(fine_results, []))
                print('Validation coarse mAP: \n', compute_map([], coarse_results))

                with open(opt.save_path + '/val_result_faster.txt', 'a') as f:
                    f.write(str(map50))

                eff = 0
                for i in policies:
                    eff += int(i)
                with open(opt.save_path + '/val_policies_faster.txt', 'a') as f:
                    f.write(str(eff / len(policies)) + '\n')

            # save
            if e % 1 == 0:
                os.makedirs(opt.save_path, exist_ok=True)
                torch.save(fine_model, os.path.join(opt.save_path, 'fine_model_{}'.format(e)))
                torch.save(coarse_model, os.path.join(opt.save_path, 'coarse_model_{}'.format(e)))

        # Testing
        fine_dataset, coarse_dataset, policies = rl_agent.eval(split_test_path, original_img_path_test)
        fine_results, coarse_results = [], []

        if len(fine_dataset.tolist()) > 0:
            fine_test_dataset = load_dataset(fine_dataset, fine_tr, bs)
            fine_test_loader = load_dataloader(bs, fine_test_dataset)
            fine_nb = len(fine_test_loader)
            for i, fine_test in tqdm.tqdm(enumerate(fine_test_loader), total=fine_nb):
                fine_results += make_results(fine_model, fine_test, device)

        if len(coarse_dataset.tolist()) > 0:
            coarse_test_dataset = load_dataset(coarse_dataset, fine_tr, bs)
            coarse_test_loader = load_dataloader(bs, coarse_test_dataset)
            coarse_nb = len(coarse_test_loader)
            for i, coarse_test in tqdm.tqdm(enumerate(coarse_test_loader), total=coarse_nb):
                coarse_results += make_results(coarse_model, coarse_test, device)

        map50 = compute_map(fine_results, coarse_results)
        print('MAP: \n', map50)
        print('Test find mAP: \n', compute_map(fine_results, []))
        print('Test coarse mAP: \n', compute_map([], coarse_results))

        with open(opt.save_path + '/test_result_faster.txt', 'a') as f:
            f.write(str(map50))

        eff = 0
        for i in policies:
            eff += int(i)
        with open(opt.save_path + '/test_policies_faster.txt', 'a') as f:
            f.write(str(eff / len(policies)) + '\n')


