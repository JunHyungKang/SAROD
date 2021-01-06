import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import numpy as np
import shutil
import json

from EfficientObjectDetection.utils import utils_detector
from EfficientObjectDetection.dataset.dataloader_ete import CustomDatasetFromImages, CustomDatasetFromImages_timetest, CustomDatasetFromImages_test
from EfficientObjectDetection.constants import base_dir_groundtruth, base_dir_detections_cd, base_dir_detections_fd, base_dir_metric_cd, base_dir_metric_fd
from EfficientObjectDetection.constants import num_windows, img_size_fd, img_size_cd

def save_args(__file__, args):
    shutil.copy('EfficientObjectDetection/' + os.path.basename(__file__), 'EfficientObjectDetection/'+ args.cv_dir)
    with open('EfficientObjectDetection/' + args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros(x.shape)
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, (f_path, c_path) in enumerate(file_dirs):
        counter = 0
        for i in range(4):
            # ---------------- Read Ground Truth ----------------------------------
            outputs_all = []
            # gt_path = '{}/{}_{}_{}.txt'.format(base_dir_groundtruth, file_dir_st, xind, yind)
            gt_path = f_path[i].replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(gt_path):
                gt = np.loadtxt(gt_path).reshape([-1, 5])
                targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                # ----------------- Read Detections -------------------------------
                if policy[index, counter] == 1:
                    preds_dir = '{}/{}_{}_{}.npy'.format(base_dir_detections_fd, file_dir_st, xind, yind)
                    targets[:, 2:] *= img_size_fd
                    if os.path.exists(preds_dir):
                        preds = np.load(preds_dir).reshape([-1,7])
                        outputs_all.append(torch.from_numpy(preds))
                else:
                    preds_dir = '{}/{}_{}_{}.npy'.format(base_dir_detections_cd, file_dir_st, xind, yind)
                    targets[:, 2:] *= img_size_cd
                    if os.path.exists(preds_dir):
                        preds = np.load(preds_dir).reshape([-1,7])
                        outputs_all.append(torch.from_numpy(preds))
                set_labels += targets[:, 1].tolist()
                metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
            else:
                continue
            counter += 1

    return metrics, set_labels

def read_offsets(image_ids, num_actions):
    offset_fd = torch.zeros((len(image_ids), num_actions)).cuda()
    offset_cd = torch.zeros((len(image_ids), num_actions)).cuda()
    for index, img_id in enumerate(image_ids):
        offset_fd[index, :] = torch.from_numpy(np.load('{}/{}'.format(base_dir_metric_fd, os.path.splitext(img_id)[0]+'.npy')).flatten())
        offset_cd[index, :] = torch.from_numpy(np.load('{}/{}'.format(base_dir_metric_cd, os.path.splitext(img_id)[0]+'.npy')).flatten())
    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set


def compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy, beta, sigma):
    """
    Args:
        offset_fd: np.array, shape [batch_size, num_actions]
        offset_cd: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    c_ap += 0.05
    reward_patch_diff = (f_ap - c_ap)*policy + -1*((f_ap - c_ap)*(1-policy))
    reward_patch_acqcost = (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    f_ob_1 = torch.where(f_ob > 0, torch.tensor(0), torch.tensor(1))
    r_penalty = torch.abs(f_ob_1 - policy)

    # print('\nf_ob', f_ob)
    # print('\nf_ob.mean(dim=1)', f_ob.mean(dim=1).unsqueeze(-1))
    # print('\nf_ob.std(dim=1)', f_ob.std(dim=1).unsqueeze(-1))
    # object_norm = (f_ob - torch.mean(f_ob)*0.1)/torch.max(f_ob)
    # reward_object = object_norm * policy + -1 * (object_norm * (1-policy))
    reward_img = reward_patch_diff.sum(dim=1) + 0.05 * r_penalty.sum(dim=1) + 0.2 * reward_patch_acqcost
    reward = reward_img.unsqueeze(1)

    return reward.float()


def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Scale(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test


def get_dataset(img_size, fine_data, coarse_data, task, origin_path):
    # data: source_path, paths[i], p, r, ap50, loss_list[i].mean(), nl
    transform_train, transform_test = get_transforms(img_size)
    if task == 'train':
        trainset = CustomDatasetFromImages(fine_data, coarse_data, transform_train, origin_path)
    else:
        trainset = CustomDatasetFromImages(fine_data, coarse_data, transform_test, origin_path)
    return trainset

def get_dataset_test(img_size, img_path):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages_test(img_path, transform_test)
    return trainset


def get_dataset_timetest(img_size, root='data/'):
    transform_train, transform_test = get_transforms(img_size)
    # trainset = CustomDatasetFromImages(root+'train.csv', transform_train)
    testset = CustomDatasetFromImages_timetest(root + 'test.csv', transform_test)

    return testset
    # return trainset, testset


def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent

import torch.nn as nn
import torch.nn.functional as F


def critic_model(num_output):
    critic = torchmodels.resnet34()
    set_parameter_requires_grad(critic, False)
    num_ftrs = critic.fc.in_features
    critic.fc = torch.nn.Linear(num_ftrs, num_output)

    return critic

def ppo_model(num_output):
    ppo = torchmodels.resnet34()
    set_parameter_requires_grad(ppo, False)
    num_ftrs = ppo.fc.in_features
    ppo.fc = torch.nn.Linear(num_ftrs, num_output)

    return ppo


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(32, 64, 7, )
        self.conv3 = nn.Conv2d(64, 128, 7, )

        self.fc1 = nn.Linear(28, 1)

    def v(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = nn.AdaptiveAvgPool2d()(x) # (bs, 128, 1, 1)

        x = F.tanh(self.fc1(x))

        return x

class PPO(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 9)
        self.conv2 = nn.Conv2d(1, 1, 9)
        self.conv3 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 1)

    def v(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x