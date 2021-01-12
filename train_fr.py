# import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse
import torch
import tqdm
import numpy as np
import copy

# torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

# from yolov5.train_dt import yolov5
from EfficientObjectDetection.train_new_reward import EfficientOD

# import fr_utils
import munch
import os
import utils
from utils import load_filenames, load_dataset, load_dataloader, compute_map, convert_yolo2coco, label2idx, label_matching, reduce_dict, make_results

opt = {'epochs':100,
      'batch_size':12,
      'device':1,
      'test_epoch':10,
      'eval_epoch':2,
      'step_batch_size':100,
      'save_path':'save',
       'save_freq': 5,
      'rl_weight':None,
      'print_freq': 50,
      'h_detector_weight':'',
      'l_detector_weight':'',
      'fine_tr':'config/fine_tr.yaml',
      'fine_eval':'config/fine_eval.yaml',
      'coarse_tr':'config/coarse_tr.yaml',
      'coarse_eval':'config/coarse_eval.yaml',
      'EfficientOD':'config/EfficientOD.yaml',
      'split': 4}
       
opt = munch.AutoMunch(opt)

# GPU Device
gpu_id = opt.device
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device " , use_cuda)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

efficient_config['load'] = None # bug fix

epochs = opt.epochs
bs = opt.batch_size
# fine_detector = yolov5(fine_tr, fine_eval, epochs, bs)
# coarse_detector = yolov5(coarse_tr, coarse_eval, epochs, bs)
rl_agent = EfficientOD(efficient_config)

split_train_path = '/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/train/images'
split_val_path = '/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/val/images'
split_test_path = '/home/SSDD/ICIP21_dataset/800_HRSID/split_data_4_0/rl_ver/test/images'
split = 4

original_img_path = '/home/SSDD/ICIP21_dataset/800_HRSID/origin_data/rl_ver/'
original_img_path_train = original_img_path + 'train/images'
original_img_path_val = original_img_path + 'val/images'
original_img_path_test = original_img_path + 'test/images'

assert bs % split == 0, 'batch size should be divided with image split patch size'

num_classes = 2
# anchor_generator = AnchorGenerator(sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),))
# fine_backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=1, pretrained_backbone=False).backbone
# coarse_backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=1, pretrained_backbone=False).backbone

# # roi
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)

# fine_model = FasterRCNN(fine_backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
#                        min_size=800, max_size=1333, rpn_nms_thresh=0.5, rpn_fg_iou_thresh=0.5)
# coarse_model = FasterRCNN(coarse_backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
#                          min_size=800, max_size=1333, rpn_nms_thresh=0.5, rpn_fg_iou_thresh=0.5)

fine_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, pretrained_backbone=False)
coarse_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, pretrained_backbone=False)

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

for e in range(epochs):
    # label이 없더라도 loader에 image 생성
    train_imgs = load_filenames(split_train_path, split, bs).files_array()
    fine_train_dataset = load_dataset(train_imgs, fine_tr, bs)
    coarse_train_dataset = load_dataset(train_imgs, fine_tr, bs)

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
        warmup_iters = min(1000, fine_train_nb-1)
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
        
        if i % opt.print_freq ==0:
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
        
        if i % opt.print_freq ==0:
            space_fmt = ':' + str(len(str(fine_train_nb))) + 'd'
            log_msg = coarse_metric_logger.delimiter.join([coarse_header, '[{0' + space_fmt + '}/{1}]', '{meters}'])
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
        print(len(fine_dataset.tolist()))
        print(len(coarse_dataset.tolist()))
        fine_results, coarse_results = [], []

        if len(fine_dataset.tolist()) > 0:
            fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
            fine_val_loader = load_dataloader(bs, fine_val_dataset)
            fine_nb = len(fine_val_loader)
            for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
                fine_results += make_results(fine_model, fine_val, device)

        if len(coarse_dataset.tolist()) > 0:
            coarse_val_dataset = load_dataset(coarse_dataset, fine_tr, bs)
            coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
            coarse_nb = len(coarse_train_loader)
            for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
                coarse_results += make_results(coarse_model, coarse_val, device)

        map50 = compute_map(fine_results, coarse_results)
        print('Validation MAP: \n', map50)
        
    # save
    if e % opt.save_freq == 0:
        torch.save(fine_model,  os.path.join(opt.save, 'fine_model'))
        torch.save(coarse_model, os.path.join(opt.save, 'coarse_model'))


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

with open('test_result.txt', 'a') as f:
    f.write(str(map50))

with open('test_policies.txt', 'a') as f:
    f.write(str(policies))