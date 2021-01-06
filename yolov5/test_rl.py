import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import copy

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets_rl import create_dataloader
from yolov5.utils.general_rl import (coco80_to_coco91_class, check_file, check_img_size, scale_coords, xyxy2xywh,
                                     clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target)
from yolov5.utils.utils import compute_loss, non_max_suppression, ap_per_class
from yolov5.utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=1,
         imgsz=480,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=True,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False,
         task='train'):
    # Initialize/load model and set device
    result_list = []

    training = model is not None
    # print('detector-eval function 확인 - model:', model)
    if training:  # called by train_backup.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    # if not training:
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    if task == 'val':
        path = data['val']
    elif task == 'train':
        path = data['train']  # path to val/test images
    elif task == 'test':
        path = data['test']
    # path = data['val'] if task == 'val' else data['train']  # path to val/test images
    # path = data['train'] # path to val/test images
    # print('path', path)
    dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(),
                                   augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    # s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'number of object')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    total_stats = []
    for batch_i, (img_list, targets_list, paths_list, shapes_list) in enumerate(tqdm(dataloader)):
        # print('\npaths_list', paths_list)
        # print('\ntargets_list', targets_list)

        # index_stats = []
        for number in range(batch_size):
            jdict, ap, ap_class = [], [], []
            img = img_list[number]  # 4, c, h, w
            targets = targets_list[targets_list[:, 0] == number, 1:]  # [x, 6] (6: img patch number, class number, bounding box corrd)
            # print('\ntargets', targets)
            paths = paths_list[number]
            shapes = shapes_list[number]

            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # 4, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                t = time_synchronized()
                inf_out_list = []
                train_out_list = []
                for i in range(4):
                    inf_out, train_out = model(img[i].unsqueeze(0), augment=augment)  # inference and training outputs
                    inf_out_list.append(inf_out)
                    train_out_list.append(train_out)
                t0 += time_synchronized() - t

                # Compute loss
                if training:  # if model has loss hyperparameters
                    loss_list = []
                    for i in range(4):
                        target_forloss = torch.zeros(targets[targets[:, 0]==i, :].shape, device=device)
                        target_forloss[:, 0] = 0
                        target_forloss[:, 1:] = targets[targets[:, 0]==i, 1:]
                        # print('\ntarget_forloss', target_forloss)
                        # loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls
                        loss = compute_loss([x.float() for x in train_out_list[i]], target_forloss, model)[1][:3]  # GIoU, obj, cls
                        loss_list.append(loss)

                # Run NMS
                t = time_synchronized()
                output_list = []
                for i in range(4):
                    output = non_max_suppression(inf_out_list[i], conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
                    output_list.append(output)
                t1 += time_synchronized() - t
                # print('\noutput_list', output_list)

            # Statistics per image
            nl_patch = []

            for i in range(4):
                p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
                stats, ap_class = [], []
                for si, pred in enumerate(output_list[i]):
                    # print('\noutput_list[i]', output_list[i])
                    # print('\nsi', si)
                    # print('\npred', pred)
                    labels = targets[targets[:, 0] == i, 1:]
                    nl = len(labels)
                    # print('\nlabels', labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if len(labels):
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                            total_stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Append to text file
                    if save_txt:
                        gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                        txt_path = str(out / Path(paths[si]).stem)
                        pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
                        for *xyxy, conf, cls in pred:
                            print('\ncls:\n', cls)
                            print('\n&xyxy:\n', *xyxy)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Append to pycocotools JSON dictionary
                    if save_json:
                        # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                        image_id = Path(paths[si]).stem
                        box = pred[:, :4].clone()  # xyxy
                        scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                        box = xyxy2xywh(box)  # xywh
                        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                        for p, b in zip(pred.tolist(), box.tolist()):
                            jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                          'category_id': coco91class[int(p[5])],
                                          'bbox': [round(x, 3) for x in b],
                                          'score': round(p[4], 5)})

                    # Assign all predictions as incorrect
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                    # print('\ncorrect', correct)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        # print('\nlabels.shape\n', labels.shape)
                        tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                        # print('\ntbox', tbox)

                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                            # print('\nti', ti)
                            # print('\npi', pi)

                            # Search for detections
                            if pi.shape[0]:
                                # Prediction to target ious
                                ious, ii = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                                # print('\nious', ious)
                                # print('\nii', ii)

                                # Append detections
                                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                    # print('\nious > iouv[0]', ious > iouv[0])
                                    d = ti[ii[j]]  # detected target
                                    # print('\nti[ii[j]]', ti[ii[j]])
                                    if d not in detected:
                                        if len(detected) != len(labels):
                                            detected.append(d)
                                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                        # if len(detected) == nl:  # all targets already located in image
                                        #     break

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                    total_stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                #
                # print(len(stats))
                # result_stats = copy.deepcopy(stats)

                temp = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
                if len(temp) and temp[0].any():
                    p, r, ap, f1, ap_class = ap_per_class(*temp)
                    p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                source_path = paths[i].split(os.sep)[-1][:-6]
                result_list.append((source_path, paths[i], mp, mr, float(map50), loss_list[i].mean().detach().cpu().item(), nl, stats))

    total_temp = [np.concatenate(x, 0) for x in zip(*total_stats)]

    # print('\nlen(total_temp) and total_temp[0].any()', len(total_temp), total_temp, total_temp[0])

    if len(total_temp) and total_temp[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*total_temp)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print('{} result - map: {}'.format(imgsz, map50))
    print('{} result - precison: {}'.format(imgsz, mp))
    print('{} result - recall: {}'.format(imgsz, mr))

    save_path = 'save/' + task + '_' + str(imgsz) + '_results.txt'
    with open(save_path, 'a') as f:
        f.write(str(map50)+' '+str(mp)+' '+str(mr)+'\n')

    # print('\nresult_list', result_list)

    return result_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres, opt.save_json,
             opt.single_cls, opt.augment, opt.verbose)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
