import easydict
from multiprocessing import Process
import yaml
from pathlib import Path
import argparse

from yolov5.train_dt import *
from EfficientObjectDetection.train_new_reward import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--rl_weight', default=None)
    parser.add_argument('--h_detector_weight', default=' ')
    parser.add_argument('--l_detector_weight', default=' ')
    parser.add_argument('--test_path', default=None)
    opt = parser.parse_args()

    fine_opt_tr = easydict.EasyDict({
        "cfg": "yolov5/models/yolov5x_custom.yaml",
        "data": "yolov5/data/HRSID_800_od.yaml",
        "hyp": '',
        "epochs": 1,
        "batch_size": 1,
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
        "epochs": 1,
        "batch_size": 1,
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
        "cv_dir": 'save',
        "batch_size": 1,
        "step_batch_size": 1,
        "img_size": 480,
        "epoch_step": 20,
        "max_epochs": 1,
        "num_workers": 0,
        "parallel": False,
        "alpha": 0.8,
        "beta": 0.1,
        "sigma": 0.5,
        "load": opt.rl_weight,
        "test_path": opt.test_path
    })

    rl_agent = EfficientOD(EfficientOD_opt)
    fine_detector = yolov5(fine_opt_tr, fine_opt_eval)
    coarse_detector = yolov5(coarse_opt_tr, coarse_opt_eval)

    fine_detector.main(0)
    coarse_detector.main(0)

    rl_agent.visualization(fine_detector, coarse_detector)


