# import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse
import torch
import tqdm
import numpy as np

# from yolov5.train_dt import yolov5
from EfficientObjectDetection.train_new_reward import EfficientOD
from utils import load_filenames, load_dataset, load_dataloader, compute_map, convert_yolo2coco, label2idx, label_matching
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import fr_utils
import munch


opt = {'epochs':100,
      'batch_size':24,
      'device':2,
      'test_epoch':10,
      'eval_epoch':2,
      'step_batch_size':100,
      'save_path':'save',
      'rl_weight':None,
      'h_detector_weight':'',
      'l_detector_weight':'',
      'fine_tr':'config/fine_tr.yaml',
      'fine_eval':'config/fine_eval.yaml',
      'coarse_tr':'config/coarse_tr.yaml',
      'coarse_eval':'config/coarse_eval.yaml',
      'EfficientOD':'config/EfficientOD.yaml'}

opt = munch.AutoMunch(opt)

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

original_img_path = '/home/SSDD/ICIP21_dataset/origin_data/rl_ver/test/images'

assert bs % split == 0, 'batch size should be divided with image split patch size'


# load a model pre-trained pre-trained on COCO
fine_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
coarse_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# # replace the classifier with a new one, that has
# # num_classes which is user-defined
num_classes = 1  # 1 class (person) + background
# # get number of input features for the classifier
fine_in_features = fine_model.roi_heads.box_predictor.cls_score.in_features
coarse_in_features = coarse_model.roi_heads.box_predictor.cls_score.in_features

# # replace the pre-trained head with a new one
fine_model.roi_heads.box_predictor = FastRCNNPredictor(fine_in_features, num_classes)
coarse_model.roi_heads.box_predictor = FastRCNNPredictor(coarse_in_features, num_classes)

fine_model.cuda()
coarse_model.cuda()

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

    for i, (fine_train, coarse_train) in tqdm.tqdm(enumerate(zip(fine_train_loader, coarse_train_loader)),
                                                           total=fine_train_nb):

        # YOLOv5 train        
#         fine_detector.train(e, i, nb, fine_train_dataset, fine_train)
#         coarse_detector.train(e, i, nb, coarse_train_dataset, coarse_train)

        # Label mathching
        fine_imgs, fine_labels = label_matching(fine_train)
        coarse_imgs, coarse_labels = label_matching(coarse_train)
        
        fine_imgs.cuda()
        coarse_imgs.cuda()
        
        ## train: img normalization --> not, zerodivision err
        fine_loss_dict = fine_model(fine_imgs/255., fine_labels)
        coarse_loss_dict = coarse_model(coarse_imgs/255., coarse_labels)
        
        fine_losses = sum(loss for loss in fine_loss_dict.values())
        coarse_losses = sum(loss for loss in coarse_loss_dict.values())
        
        # utils
        fine_loss_dict_reduced = reduce_dict(fine_loss_dict)
        coarse_loss_dict_reduced = reduce_dict(coarse_loss_dict)
        fine_loss_reduced = sum(loss for loss in fine_loss_dict_reduced.values())
        coarse_loss_reduced = sum(loss for loss in coarse_loss_dict_reduced.values())
        fine_loss_val = fine_loss_reduced.item()
        coarse_loss_val = coarse_loss_reduece.item()
        
        # optimizer
        # loss backward
        # optim step
        
        break
        
        ## train eval
        
        
        
        
        
        # result = (source_path, paths[si], mp, mr, map50, nl, stats)
        # fine_results = fine_detector.eval(fine_train)
        # coarse_results = coarse_detector.eval(coarse_train)

#         rl_agent.train(e, i, nb, fine_results, coarse_results)

#         # Validation
#         if e % 10 == 0:
#             fine_dataset, coarse_dataset, policies = rl_agent.eval(split_val_path, original_img_path)
#             fine_results, coarse_results = [], []

#             print(len(fine_dataset.tolist()))
#             print(len(coarse_dataset.tolist()))

#             if len(fine_dataset.tolist()) > 0:
#                 fine_val_dataset = load_dataset(fine_dataset, fine_tr, bs)
#                 fine_val_loader = load_dataloader(bs, fine_val_dataset)
#                 fine_nb = len(fine_val_loader)
#                 for i, fine_val in tqdm.tqdm(enumerate(fine_val_loader), total=fine_nb):
#                     fine_results.append(fine_detector.eval(fine_val))

#             if len(coarse_dataset.tolist()) > 0:
#                 coarse_val_dataset = load_dataset(coarse_dataset, fine_tr, bs)
#                 coarse_val_loader = load_dataloader(bs, coarse_val_dataset)
#                 coarse_nb = len(coarse_train_loader)
#                 for i, coarse_val in tqdm.tqdm(enumerate(coarse_val_loader), total=coarse_nb):
#                     coarse_results.append(coarse_detector.eval(coarse_val))

        map50 = compute_map(fine_results, coarse_results)
        print('MAP: \n', map50)
        break
    break
