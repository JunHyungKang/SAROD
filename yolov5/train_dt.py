import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import yolov5.test_original as test  # import test.py to get mAP after each epoch
import yolov5.test_rl as test_rl  # import test.py to get mAP after each epoch
from yolov5.models.yolo import Model
from yolov5.utils import google_utils
from yolov5.utils.datasets import *
from yolov5.utils.utils import *
from yolov5.models.experimental import *

mixed_precision = False

class yolov5():

    def __init__(self, opt_tr, opt_eval, epochs, bs):
        self.opt = opt_tr
        self.opt_eval = opt_eval
        self.dataloader = None
        self.dataset = None

        self.wdir = 'weights' + os.sep  # weights dir
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = 'results.txt'

        # Hyperparameters
        self.hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
                    'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'momentum': 0.937,  # SGD momentum/Adam beta1
                    'weight_decay': 5e-4,  # optimizer weight decay
                    'giou': 0.05,  # giou loss gain
                    'cls': 0.5,  # cls loss gain
                    'cls_pw': 1.0,  # cls BCELoss positive_weight
                    'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
                    'obj_pw': 1.0,  # obj BCELoss positive_weight
                    'iou_t': 0.20,  # iou training threshold
                    'anchor_t': 4.0,  # anchor-multiple threshold
                    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
                    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                    'degrees': 0.0,  # image rotation (+/- deg)
                    'translate': 0.0,  # image translation (+/- fraction)
                    'scale': 0.5,  # image scale (+/- gain)
                    'shear': 0.0}  # image shear (+/- deg)

        # resume training
        self.last = get_latest_run() if self.opt['resume'] == 'get_last' else self.opt['resume']  # resume from most recent run
        if self.last and not self.opt['weights']:
            print(f'Resuming training from {self.last}')
        self.opt['weights'] = self.last if self.opt['resume'] and not self.opt['weights'] else self.opt['weights']

        # git check?
        # if self.opt['local_rank'] in [-1, 0]:
        #     check_git_status()

        # config file check
        self.opt['cfg'] = check_file(self.opt['cfg'])
        self.opt['data'] = check_file(self.opt['data'])

        # check hyps, update if there is
        if self.opt['hyp']:
            self.opt['hyp'] = check_file(self.opt['hyp'])  # check file
            with open(self.opt['hyp']) as f:
                self.hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps

        # extend to 2 sizes (train, test)
        self.opt['img_size'].extend([self.opt['img_size'][-1]] * (2 - len(self.opt['img_size'])))

        # batch size
        self.batch_size = bs
        self.total_batch_size = self.batch_size

        # GPU setting
        self.device = torch_utils.select_device(str(self.opt['device']), apex=False, batch_size=self.batch_size)

        self.opt['world_size'] = 1

        if self.device.type == 'cpu':
            mixed_precision = False
        elif self.opt['local_rank'] != -1:
            # DDP mode
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            self.device = torch.device("cuda", self.opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

            self.opt.world_size = dist.get_world_size()
            assert self.opt.batch_size % self.opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size

        # Train
        if self.opt['local_rank'] in [-1, 0]:
            # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.opt['save_path'], self.opt['name']))
        else:
            self.tb_writer = None
        self.load_model(self.hyp, self.tb_writer, self.opt, self.device, epochs)


    def load_model(self, hyp, tb_writer, opt, device, epochs):
        self.hyp = hyp
        self.tb_writer = tb_writer
        self.opt = opt
        self.device = device
        self.log_dir = self.tb_writer.log_dir if self.tb_writer else 'runs/evolution'  # run directory
        self.wdir = str(Path(self.log_dir) / 'weights') + os.sep  # weights directory
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = self.log_dir + os.sep + 'results.txt'
        self.epochs = epochs
        weights = self.opt['weights']
        rank = self.opt['local_rank']

        # Save run settings
        with open(Path(self.log_dir) / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(Path(self.log_dir) / 'opt.yaml', 'w') as f:
            yaml.dump(self.opt, f, sort_keys=False)

        # data Configure
        init_seeds(2 + rank)
        with open(self.opt['data']) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        train_path = data_dict['train']
        test_path = data_dict['val']
        self.nc, self.names = (1, ['item']) if opt['single_cls'] else (
        int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt['data'])  # check

        # Remove previous results
        if rank in [-1, 0]:  # True
            for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_file):
                os.remove(f)

        # Create model
        self.model = Model(self.opt['cfg'], nc=self.nc).to(self.device)

        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride): 32?
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt['img_size']]  # verify imgsz are gs-multiples

        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.batch_size * self.accumulate / self.nbs  # scale weight_decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        if self.hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Load Model
        with torch_distributed_zero_first(rank):
            google_utils.attempt_download(weights)
        self.start_epoch, self.best_fitness = 0, 0.0
        if weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint

            # load model
            try:
                exclude = ['anchor']  # exclude keys
                ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                                 if k in self.model.state_dict() and not any(x in k for x in exclude)
                                 and self.model.state_dict()[k].shape == v.shape}
                self.model.load_state_dict(ckpt['model'], strict=False)
                print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(self.model.state_dict()), weights))
            except KeyError as e:
                s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                    "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                    % (weights, self.opt['cfg'], weights, weights)
                raise KeyError(s) from e

            # load optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (self.opt.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # DP mode
        if self.device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt['sync_bn'] and self.device.type != 'cpu' and rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            print('Using SyncBatchNorm()')

        # Exponential moving average
        # self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.ema = torch_utils.ModelEMA(self.model) if rank in [-1, 0] else None

        # DDP mode
        if self.device.type != 'cpu' and rank != -1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

    def train(self, epoch, i, nb, dataset, dataloader):
        self.dataloader = dataloader
        self.dataset = dataset
        self.labels = self.dataset.labels

        mlc = np.concatenate(self.labels, 0)[:, 0].max()  # max label class

        self.nb = nb  # number of batches
        assert mlc < self.nc, 'Label class %g exceeds nc=%g. Possible class labels are 0-%g' % (mlc, self.nc, self.nc - 1)

        # Model parameters
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        self.model.class_weights = labels_to_class_weights(self.labels, self.nc).to(self.device)  # attach class weights
        self.model.names = self.names
        rank = self.opt['local_rank']

        # Class frequency
        if rank in [-1, 0]:
            labels = np.concatenate(self.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=self.log_dir)
            if self.tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                self.tb_writer.add_histogram('classes', c, 0)

        self.nw = max(3 * self.nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.maps = np.zeros(self.nc)  # mAP per class
        self.results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        t0 = time.time()
        # if rank in [0, -1]:
        #     print('Image sizes %g train, %g test' % (self.imgsz, self.imgsz_test))
        #     print('Starting training for %g epochs...' % epoch)
        # torch.autograd.set_detect_anomaly(True)

        ##########################################################################################################
        # Start training
        self.model.train()

        self.mloss = torch.zeros(4, device=self.device)  # mean losses

        if rank != -1:
            self.dataloader.sampler.set_epoch(epoch)

        # pbar = enumerate(self.dataloader)
        # if rank in [-1, 0]:
            # print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            # pbar = tqdm(pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()

        # batch -------------------------------------------------------------
        # for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        imgs = self.dataloader[0]
        targets = self.dataloader[1]
        paths = self.dataloader[2]

        # print(paths)

        ni = i + self.nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

        # Warmup
        if ni <= self.nw:
            xi = [0, self.nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.9, self.hyp['momentum']])

        # Multi-scale
        if self.opt['multi_scale']:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in
                      imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward
        pred = self.model(imgs)

        # Loss
        loss, loss_items = compute_loss(pred, targets.to(self.device), self.model)  # scaled by batch_size
        if rank != -1:
            loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return self.results

        # Backward
        if mixed_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Optimize
        if ni % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema is not None:
                self.ema.update(self.model)

        # Print
        if rank in [-1, 0]:
            self.mloss = (self.mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            self.s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, self.epochs - 1), mem, * self.mloss, targets.shape[0], imgs.shape[-1])
            # pbar.set_description(self.s)

            # Plot
            if ni < 3:
                f = str(Path(self.log_dir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if self.tb_writer and result is not None:
                    self.tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

        # end batch --------------------------------------------------------------------------------------------

        # Scheduler
        self.scheduler.step()

        # Save model
        if i == nb:
            ckpt = {'epoch': epoch,
                    'model': self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema,
                    'optimizer': None}

            # Save last, best and delete
            torch.save(ckpt, self.opt['save_path'] + self.wdir + str(epoch) + '.pt')

            del ckpt

    def eval(self, dataloader):
        augment = self.opt_eval['augment']
        conf_thres = self.opt_eval['conf_thres']
        iou_thres = self.opt_eval['iou_thres']
        rank = self.opt['local_rank']
        self.results = []
        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if rank in [-1, 0]:
            # mAP
            if self.ema is not None:
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            # final_epoch = epoch + 1 == self.epochs
            final_epoch = True
            if final_epoch:  # Calculate mAP
                model = self.ema.ema.module if hasattr(self.ema.ema, 'module') else self.ema.ema
                save_dir = self.log_dir
                imgsz = self.imgsz_test

                training = model is not None
                device = next(model.parameters()).device  # get model device

                half = device.type != 'cpu'  # half precision only supported on CUDA
                if half:
                    model.half()

                # Configure
                model.eval()

                with open(self.opt['data']) as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
                nc = 1 if self.opt['single_cls'] else int(data['nc'])  # number of classes
                iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
                niou = iouv.numel()

                seen = 0
                names = model.names if hasattr(model, 'names') else model.module.names
                coco91class = coco80_to_coco91_class()
                s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
                p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
                loss = torch.zeros(3, device=device)
                jdict, stats, ap, ap_class = [], [], [], []
                # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

                img = dataloader[0]
                targets = dataloader[1]
                paths = dataloader[2]
                shapes = dataloader[3]

                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                nb, _, height, width = img.shape  # batch size, channels, height, width
                whwh = torch.Tensor([width, height, width, height]).to(device)

                # Disable gradients
                with torch.no_grad():
                    # Run model
                    t = torch_utils.time_synchronized()
                    inf_out, train_out = model(img, augment=augment)  # inference and training outputs
                    t0 += torch_utils.time_synchronized() - t

                    # Compute loss
                    if training:  # if model has loss hyperparameters
                        # loss_list = []
                        loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls
                        # loss_list.append(loss)

                    # Run NMS
                    t = torch_utils.time_synchronized()
                    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False)
                    t1 += torch_utils.time_synchronized() - t

                # Statistics per image

                # print('len(output): \n', len(output))
                for si, pred in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if nl:
                            stats = [torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls]
                            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

                            if len(stats) and stats[0].any():
                                p, r, ap, f1, ap_class = ap_per_class(*stats)
                                p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
                            else:
                                nt = torch.zeros(1)

                            source_path = str(paths[si].split(os.sep)[-1].split('__')[0])
                            self.results.append((source_path, paths[si], mp, mr, map50, nl, stats))
                        else:
                            source_path = str(paths[si].split(os.sep)[-1].split('__')[0])
                            self.results.append((source_path, paths[si], 1, 1, 1, 0, 0))
                        continue

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Assign all predictions as incorrect
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                            pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                            # Search for detections
                            if pi.shape[0]:
                                # Prediction to target ious
                                ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                # Append detections
                                for j in (ious > iouv[0]).nonzero():
                                    d = ti[i[j]]  # detected target
                                    if d not in detected:
                                        detected.append(d)
                                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                        if len(detected) == nl:  # all targets already located in image
                                            break

                    # Append statistics (correct, conf, pcls, tcls)
                    # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                    stats = [(correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)]

                    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

                    if len(stats) and stats[0].any():
                        p, r, ap, f1, ap_class = ap_per_class(*stats)
                        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
                    else:
                        nt = torch.zeros(1)

                    source_path = str(paths[si].split(os.sep)[-1].split('__')[0])
                    self.results.append((source_path, paths[si], mp, mr, map50, nl, stats))


        dist.destroy_process_group() if rank not in [-1, 0] else None
        torch.cuda.empty_cache()

        # print('len(self.results): \n', len(self.results))
        return self.results


    def test(self, dataloader):
        augment = self.opt_eval['augment']
        conf_thres = self.opt_eval['conf_thres']
        iou_thres = self.opt_eval['iou_thres']
        rank = self.opt['local_rank']
        self.results = []
        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if rank in [-1, 0]:
            # mAP
            if self.ema is not None:
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            # final_epoch = epoch + 1 == self.epochs
            final_epoch = True
            if final_epoch:  # Calculate mAP
                model = self.ema.ema.module if hasattr(self.ema.ema, 'module') else self.ema.ema
                save_dir = self.log_dir
                imgsz = self.imgsz_test

                training = model is not None
                device = next(model.parameters()).device  # get model device

                half = device.type != 'cpu'  # half precision only supported on CUDA
                if half:
                    model.half()

                # Configure
                model.eval()

                with open(self.opt['data']) as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
                nc = 1 if self.opt['single_cls'] else int(data['nc'])  # number of classes
                iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
                niou = iouv.numel()

                seen = 0
                names = model.names if hasattr(model, 'names') else model.module.names
                coco91class = coco80_to_coco91_class()
                s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
                p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
                loss = torch.zeros(3, device=device)
                jdict, stats, ap, ap_class = [], [], [], []
                # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

                img = dataloader[0]
                targets = dataloader[1]
                paths = dataloader[2]
                shapes = dataloader[3]

                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                nb, _, height, width = img.shape  # batch size, channels, height, width
                whwh = torch.Tensor([width, height, width, height]).to(device)

                # Disable gradients
                with torch.no_grad():
                    # Run model
                    t = torch_utils.time_synchronized()
                    inf_out, train_out = model(img, augment=augment)  # inference and training outputs
                    t0 += torch_utils.time_synchronized() - t

                    # Compute loss
                    if training:  # if model has loss hyperparameters
                        # loss_list = []
                        loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls
                        # loss_list.append(loss)

                    # Run NMS
                    t = torch_utils.time_synchronized()
                    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False)
                    t1 += torch_utils.time_synchronized() - t

                # Statistics per image

                # print('len(output): \n', len(output))
                for si, pred in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if nl:
                            stats_return = [(torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)]
                            stats = [np.concatenate(x, 0) for x in zip(*stats_return)]  # to numpy

                            if len(stats) and stats[0].any():
                                p, r, ap, f1, ap_class = ap_per_class(*stats)
                                p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
                            else:
                                nt = torch.zeros(1)

                            source_path = str(paths[si].split(os.sep)[-1].split('__')[0])
                            self.results.append((source_path, paths[si], mp, mr, map50, nl, stats_return))
                        continue

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Assign all predictions as incorrect
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                            pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                            # Search for detections
                            if pi.shape[0]:
                                # Prediction to target ious
                                ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                # Append detections
                                for j in (ious > iouv[0]).nonzero():
                                    d = ti[i[j]]  # detected target
                                    if d not in detected:
                                        detected.append(d)
                                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                        if len(detected) == nl:  # all targets already located in image
                                            break

                    # Append statistics (correct, conf, pcls, tcls)
                    # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                    stats_return = [(correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)]

                    stats = [np.concatenate(x, 0) for x in zip(*stats_return)]  # to numpy

                    if len(stats) and stats[0].any():
                        p, r, ap, f1, ap_class = ap_per_class(*stats)
                        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
                        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
                    else:
                        nt = torch.zeros(1)

                    source_path = str(paths[si].split(os.sep)[-1].split('__')[0])
                    self.results.append((source_path, paths[si], mp, mr, map50, nl, stats_return))

        dist.destroy_process_group() if rank not in [-1, 0] else None
        torch.cuda.empty_cache()

        return self.results