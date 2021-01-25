import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse

from yolov5.train_dt import *
from EfficientObjectDetection.train_new_reward import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--detector_batch_size', type=int, default=32, help="Total batch size for all gpus.")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--step_batch_size', type=int, default=100)
    parser.add_argument('--save_path', default='save')
    parser.add_argument('--rl_weight', default=None)
    parser.add_argument('--h_detector_weight', default=' ')
    parser.add_argument('--l_detector_weight', default=' ')
    parser.add_argument('--test_path', default=None)
    opt = parser.parse_args()

    fine_opt_tr = easydict.EasyDict({
        "cfg": "yolov5/models/yolov5x_custom.yaml",
        "data": "yolov5/data/HRSID_800_od.yaml",
        "hyp": '',
        "epochs": opt.epochs,
        "batch_size": opt.detector_batch_size,
        "img_size": [480, 480],
        "rect": False,
        "resume": False,
        "nosave": False,
        "notest": True,
        "noautoanchor": True,
        "evolve": False,
        "bucket": '',
        "cache_images": False,
        "weights": 'weights/'+opt.h_detector_weight,
        "name": "yolov5x_800_480_200epoch",
        "device": opt.device,
        "multi_scale": False,
        "single_cls": True,
        "sync_bn": False,
        "local_rank": -1
    })

    fine_opt_eval = easydict.EasyDict({
        "data": "yolov5/data/HRSID_800_rl.yaml",
        "batch_size": 1,
        "conf_thres": 0.001,
        "iou_thres": 0.6  # for NMS
    })

    coarse_opt_tr = easydict.EasyDict({
        "cfg": "yolov5/models/yolov5x_custom.yaml",
        "data": "yolov5/data/HRSID_800_od.yaml",
        "hyp": '',
        "epochs": opt.epochs,
        "batch_size": opt.detector_batch_size,
        "img_size": [96, 96],
        "rect": False,
        "resume": False,
        "nosave": False,
        "notest": True,
        "noautoanchor": True,
        "evolve": False,
        "bucket": '',
        "cache_images": False,
        "weights": 'weights/'+opt.l_detector_weight,
        "name": "yolov5x_800_96_200epoch",
        "device": opt.device,
        "multi_scale": False,
        "single_cls": True,
        "sync_bn": False,
        "local_rank": -1
    })

    coarse_opt_eval = easydict.EasyDict({
        "data": "yolov5/data/HRSID_800_rl.yaml",
        "batch_size": 1,
        "conf_thres": 0.001,
        "iou_thres": 0.6  # for NMS
    })

    EfficientOD_opt = easydict.EasyDict({
        "gpu_id": opt.device,
        "lr": 1e-3,
        "cv_dir": opt.save_path,
        "batch_size": 1,
        "step_batch_size": opt.step_batch_size,
        "img_size": 480,
        "epoch_step": 20,
        "max_epochs": opt.epochs,
        "num_workers": 8,
        "parallel": False,
        "alpha": 0.8,
        "beta": 0.1,
        "sigma": 0.5,
        "load": opt.rl_weight,
        "test_path": opt.test_path
    })


    fine_detector = yolov5(fine_opt_tr, fine_opt_eval)
    coarse_detector = yolov5(coarse_opt_tr, coarse_opt_eval)
    rl_agent = EfficientOD(EfficientOD_opt)

    epochs = opt.epochs

    fine_detector.main(epochs)
    coarse_detector.main(epochs)

    for e in range(epochs):
        # fine_detector.train(e)
        # coarse_detector.train(e)
        # fine_eval_results = fine_detector.eval('train')
        # coarse_eval_results = coarse_detector.eval('train')
        # rl_agent.train(e, fine_eval_results, coarse_eval_results)
        # if e % opt.eval_epoch == 0:
        #     eval_fine = fine_detector.eval('val')
        #     eval_coarse = coarse_detector.eval('val')
        #     rl_agent.eval(e, eval_fine, eval_coarse)
        test_fine = fine_detector.eval('test')
        test_coarse = coarse_detector.eval('test')
        rl_agent.test(e, test_fine, test_coarse)


