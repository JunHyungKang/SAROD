import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import deque
import pickle
import pylab
cudnn.benchmark = True
import argparse
from torch.autograd import Variable
# from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli
from collections import deque
import random
import torchvision.transforms as transforms

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from EfficientObjectDetection.utils import utils_ete, utils_detector
from EfficientObjectDetection.constants import base_dir_metric_cd, base_dir_metric_fd
from EfficientObjectDetection.constants import num_actions
import yolov5.utils.utils as yoloutil

import warnings
warnings.simplefilter("ignore")
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

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

class EfficientOD():
    def __init__(self, opt):
        # GPU Device
        self.opt = opt
        self.result_fine = None
        self.result_coarse = None
        self.epoch = None
        self.original_data_path = None
        gpu_id = self.opt['gpu_id']
        self.buffer = deque(maxlen=20000)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        use_cuda = torch.cuda.is_available()
        print("GPU device for EfficientOD: ", use_cuda)

        if not os.path.exists(self.opt['cv_dir']):
            print(self.opt['cv_dir'])
            os.makedirs(self.opt['cv_dir'])
        # utils_ete.save_args(__file__, self.opt)

        self.agent = utils_ete.get_model(num_actions)
        self.critic = utils_ete.critic_model(1)

        # ---- Load the pre-trained model ----------------------
        if self.opt['load'] is not None:
            path = os.path.join('weights', self.opt['load'])
            checkpoint = torch.load(path)
            self.agent.load_state_dict(checkpoint['agent'])
            print('loaded agent from %s' % opt.load)

        # Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
        if self.opt['parallel']:
            agent = nn.DataParallel(self.agent)
        self.agent.cuda()
        self.critic.cuda()

        # Update the parameters of the policy network
        self.optimizer_agent = optim.Adam(self.agent.parameters(), lr=float(self.opt['lr']))
        self.optimizer_critic = optim.Adam(self.agent.parameters(), lr=float(self.opt['lr']))

    def train(self, epoch, batch_iter, nb, result_fine, result_coarse, original_data_path):
        # Start training and testing
        self.epoch = epoch
        self.result_fine = result_fine
        self.result_coarse = result_coarse
        self.original_data_path = original_data_path

        transform_train, _ = get_transforms(self.opt['img_size'])

        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.

        self.agent.train()
        if batch_iter == 1:
            self.rewards, self.rewards_baseline, self.policies, self.stats_list, self.efficiency = [], [], [], [], []

        assert len(result_fine)==len(result_coarse), 'result data size is different between fine & coarse'

        # ressults = (source_path, paths[si], mp, mr, map50, nl, stats)
        for i in range(int(len(result_fine)/self.opt['split'])):
            f_ap, c_ap, f_ob, c_ob, f_stats, c_stats = [], [], [], [], [], []
            img_path = os.path.join(self.original_data_path, result_fine[i][0] + '.png')
            img_as_img = Image.open(img_path)
            img_as_tensor = transform_train(img_as_img)
            for j in range(self.opt['split']):
                f_ap.append(result_fine[i * self.opt['split'] + j][4])
                c_ap.append(result_coarse[i * self.opt['split'] + j][4])

                f_ob.append(result_fine[i * self.opt['split'] + j][5])
                c_ob.append(result_coarse[i * self.opt['split'] + j][5])

                f_stats.append(result_fine[i * self.opt['split'] + j][6])
                c_stats.append(result_coarse[i * self.opt['split'] + j][6])

            if epoch > 10:
                self.buffer.append([img_as_tensor.numpy(), f_ap, c_ap, f_stats, c_stats, f_ob, c_ob])

        if len(self.buffer) >= 1000:
            # print('RL training is ongoing! buffer size is more than 1000')
            # pbar = range((epoch+1)*6)
            # for i in pbar:
            minibatch = random.sample(self.buffer, self.opt['step_batch_size'])
            minibatch = np.array(minibatch)

            inputs, f_ap, c_ap = minibatch[:, 0].tolist(), minibatch[:, 1].tolist(), minibatch[:, 2].tolist()
            f_stats, c_stats = minibatch[:, -4], minibatch[:, -3]
            f_ob, c_ob = minibatch[:, -2].tolist(), minibatch[:, -1].tolist()

            # inputs = Variable(inputs)
            # if not self.opt.parallel:
            inputs = torch.tensor(inputs).squeeze().cuda()
            # Actions by the Agent
            probs = F.sigmoid(self.agent.forward(inputs))
            alpha_hp = np.clip(self.opt['alpha'] + epoch * 0.001, 0.6, 0.95)
            probs = probs * alpha_hp + (1 - alpha_hp) * (1 - probs)

            # Sample the policies from the Bernoulli distribution characterized by agent
            distr = Bernoulli(probs)
            policy_sample = distr.sample()

            # Test time policy - used as baseline policy in the training step
            policy_map = probs.data.clone()
            policy_map[policy_map < 0.5] = 0.0
            policy_map[policy_map >= 0.5] = 1.0
            policy_map = Variable(policy_map)

            f_ap = torch.from_numpy(np.array(f_ap).reshape((-1, 4)))
            c_ap = torch.from_numpy(np.array(c_ap).reshape((-1, 4)))
            f_ob = torch.from_numpy(np.array(f_ob).reshape((-1, 4)))
            c_ob = torch.from_numpy(np.array(c_ob).reshape((-1, 4)))
            f_ob = f_ob.float()
            c_ob = c_ob.float()

            reward_map = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy_map.cpu().data, self.opt['beta'], self.opt['sigma'])
            reward_sample = utils_ete.compute_reward_sarod(f_ap, c_ap, f_ob, c_ob, policy_sample.cpu().data, self.opt['beta'],
                                                           self.opt['sigma'])
            advantage = reward_sample.cuda().float() - reward_map.cuda().float()

            # Find the loss for only the policy network
            loss = distr.log_prob(policy_sample)
            loss = loss * Variable(advantage).expand_as(policy_sample)
            # loss = loss.expand_as(policy_sample)
            loss = loss.mean()

            loss = loss + F.smooth_l1_loss(sum(self.critic(inputs)), sum(reward_map.cuda().float()))

            self.optimizer_agent.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_agent.step()
            self.optimizer_critic.step()

            self.rewards.append(reward_sample.cpu())
            self.rewards_baseline.append(reward_map.cpu())
            self.policies.append(policy_sample.data.cpu())

            if batch_iter == nb:

                reward, sparsity, variance, policy_set = utils_ete.performance_stats(self.policies, self.rewards)
                with open(self.opt.cv_dir+'/reward.txt', 'a') as f:
                    f.write(str(reward) + '\n')

                # save the model --- agent
                agent_state_dict = self.agent.module.state_dict() if self.opt['parallel'] else self.agent.state_dict()
                state = {
                    'agent': agent_state_dict,
                    'epoch': self.epoch,
                    'reward': reward,
                }

                torch.save(state, self.opt['cv_dir'] + '/{}_ckpt_E_{}'.format(self.opt['save_name'], self.epoch))

    def eval(self, split_val_path, original_img_path):

        self.agent.eval()

        testset = utils_ete.get_dataset_test(self.opt['img_size'], img_path=original_img_path)
        testloader = torchdata.DataLoader(testset, batch_size=self.opt['batch_size'], shuffle=False,
                                          num_workers=self.opt['num_workers'])

        p, r, f1, mp, mr, map50, map, t0, t1, c_map50, f_map50 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        stats_list, metrics, policies, set_labels, c_stats_list, f_stats_list, efficiency = [], [], [], [], [], [], []

        fine_dataset = []
        coarse_dataset = []

        for batch_idx, (inputs, label_path, img_path) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs = Variable(inputs, volatile=True)
            # if not self.opt.parallel:
            inputs = torch.tensor(inputs).cuda()

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs))

            # Sample the policy from the agents output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0
            policy = Variable(policy)

            for i, policy in enumerate(policy.cpu().data):
                for j, policy_element in enumerate(policy):
                    efficiency.append(policy_element)
                    if policy_element == 0:
                        coarse_dataset.append(
                            os.path.join(split_val_path, img_path[i].replace('.png',
                                                                             self.opt['split_format'][j] + '.png')))

                    elif policy_element == 1:
                        fine_dataset.append(
                            os.path.join(split_val_path, img_path[i].replace('.png',
                                                                             self.opt['split_format'][j] + '.png')))

        return np.array(fine_dataset), np.array(coarse_dataset), efficiency

