import glob
import math
import os
import random
import shutil
import time
import pickle
import datetime
from pathlib import Path
from threading import Thread
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.distributed as dist

import errno

from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import copy

from yolov5.utils.utils import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first, ap_per_class


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_dataloader(batch_size, dataset):
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // 1, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             sampler=train_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=load_dataset.collate_fn)
    return dataloader


class load_filenames(Dataset):  # for training/testing
    def __init__(self, path, split, bs):
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                if os.path.isdir(p):  # folder
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f])
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s.' % path
        bi = np.floor(np.arange(n) / (bs * split)).astype(np.int)  # batch index
        self.nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.split = split

    def files_array(self):
        self.img_files = np.array(self.img_files).reshape((-1, self.split))
        np.random.shuffle(self.img_files)
        return self.img_files


class load_dataset(Dataset):  # for training/testing
    def __init__(self, imgs, opt, batch_size, augment=False, hyp=None, flip=True,
                 cache_images=False, single_cls=False, stride=32, pad=0.0):
        self.img_size = opt['img_size'][0]
        self.img_files = imgs.reshape(-1).tolist()

        n = len(self.img_files)
        assert n > 0, 'No images found'

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image

        self.augment = augment
        self.hyp = hyp
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.stride = stride
        self.flip = flip

        # Define labels
        self.label_files = [x.replace('images', 'yolov5_txt').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != self.get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Cache labels
        extract_bounding_boxes, labels_loaded = False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]  # label
            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s' % (os.path.dirname(file) + os.sep)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def get_hash(self, files):
        # Returns a single hash value of a list of files
        return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            pass

        return s

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = self.exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = self.get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)

        # Letterbox
        shape = self.img_size  # final letterboxed shape
        img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.flip:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = True
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def compute_map(fine_result, coarse_result):
    final_stats = []

    if len(fine_result) > 0:
        for result in fine_result:
            final_stats.append((result[6][0], result[6][1], np.array(result[6][2]), np.array(result[6][3])))

    if len(coarse_result) > 0:
        for result in coarse_result:
            final_stats.append((result[6][0], result[6][1], np.array(result[6][2]), np.array(result[6][3])))

    final_stats = [np.concatenate(x, 0) for x in zip(*final_stats)]

    p, r, ap, f1, ap_class = ap_per_class(*final_stats)
    p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return map50

def yolo2coco(tensors, org_res, device):
    # cx, cy, w, h --> x0, y0, x1, y2
    # zero / max
    # must be x0 < x1, y0 < y1 --> if not, inf loss
    tensors *= org_res
    cx, cy, w, h = tensors[0], tensors[1], tensors[2], tensors[3]
    x0, y0 = int(max(0, min(cx-(w/2), org_res))), int(max(0, min(cy-(h/2), org_res)))
    x1, y1 = int(max(0, min(x0+w, org_res))), int(max(0, min(y0+h, org_res)))
    x0, y0, x1, y1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    if x0 == x1:
        x1 += 1
    if y0 == y1:
        y1 += 1
    tensor = torch.Tensor([[x0, y0, x1, y1]]).to(device).clone()
    return tensor

def convert_yolo2coco(targets, save_dict, org_res=480, device='cuda'):
    '''
    # inputs
        targets: (idx, cls, bb_info)
        save_dict: result dictionary {'image_id', 'labels', 'boxes'}
        org_res: original resolution
    # returns
        save_dict
    # function:
        convert tensor (yolo format) to dictionary (COCO format)
    '''
    # not first
    if save_dict:
        save_dict['labels'] = torch.cat([save_dict['labels'].to(device), targets[1].unsqueeze(0).type(torch.int64).to(device) + 1]) # label = 1
        save_dict['boxes'] = torch.cat([save_dict['boxes'].to(device), yolo2coco(targets[2:], org_res, device)])
    # first
    else:
#         save_dict['image_id'] = targets[0].unsqueeze(0).type(torch.int64).to(device)
        save_dict['labels'] = targets[1].unsqueeze(0).type(torch.int64).to(device) + 1 # label = 1
        save_dict['boxes'] = yolo2coco(targets[2:], org_res, device)
    return save_dict

def label2idx(labels):
    label_dict = {}
    for idx, label in enumerate(labels):
        label_dict[label] = idx
    length = len(labels)
    return length, label_dict

def label_matching(dataset, device='cuda'):
    '''
    # inputs:
        dataset (img, target, path, info)
    # outputs:
        img, label
    # function:
        label이 있는 image set만 matching return
    '''
    imgs = dataset[0]
    labels = dataset[1]
    
    # label length
#     length = len(np.unique(labels[:,0]))\
    # img label : dictionary label matching
    # {key=org_label:value=item_label}
    length, label_dict = label2idx(np.unique(labels[:,0]))
    
    # create dummy list
    # img --> tensor
    # label --> list[(dict)]
    data_label = [{} for _ in range(length)] # approach by item_idx 0,1,2,3...
    data_img = []
    
    for l in labels:
        # length = 6 (index, cls, cx, cy, w, h)
        item_idx = label_dict[int(l[0])] # apporach by item_idx 0,1,2,3...
        org_idx = int(l[0]) # apporach by org_idx (label idx)
        if not data_label[item_idx]:
            data_img.append(imgs[org_idx])
        # convert label
        data_label[item_idx] = convert_yolo2coco(l, data_label[item_idx], org_res=imgs.shape[-1], device=device)
    return torch.stack(data_img), data_label

def reduce_dict(input_dict, average=True):
    # ref: https://github.com/pytorch/vision/blob/3711754a508e429d0049df3c4a410c4cde08e4e6/references/detection/utils.py#L118
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    # https://github.com/pytorch/vision/blob/3711754a508e429d0049df3c4a410c4cde08e4e6/references/detection/utils.py#L239
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class MetricLogger(object):
    # https://github.com/pytorch/vision/blob/3711754a508e429d0049df3c4a410c4cde08e4e6/references/detection/utils.py
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    # https://github.com/pytorch/vision/blob/3711754a508e429d0049df3c4a410c4cde08e4e6/references/detection/utils.py
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
