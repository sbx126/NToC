import numpy as np
from copy import deepcopy
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import torch
from torch.utils.data import Dataset
import random
from scipy.special import softmax
from copy import deepcopy

from hanoi_env import Hanoi


class HanoiSeqDataset(Dataset):
    def __init__(self, disk_size_candidates, seq_len, num_digits = 5, digit_img_size = (28, 28), train = True, 
                 dataset_size = None, sample_type = "full_img_state", num_aux_vars = 6):
        super(HanoiSeqDataset, self).__init__()

        self.disk_size_candidates = disk_size_candidates
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.digit_img_size = digit_img_size
        self.train = train
        self.dataset_size = dataset_size
        self.sample_type = sample_type
        self.num_aux_vars = num_aux_vars

        self.max_num_disks = max(*self.disk_size_candidates) if len(disk_size_candidates) > 1 else disk_size_candidates[0]

        self.action_encoding = dict()
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.action_encoding[(i,j)] = len(self.action_encoding)

        # hanoi states
        self.states = []
        self.last_actions = []
        self.actions = []
        self.terminate_idxs = []
        for num_disks in self.disk_size_candidates:
            s, la, a = self._get_hanoi_trajectory(num_disks)
            self.states.extend(s)
            self.last_actions.extend(la)
            self.actions.extend(a)
            self.terminate_idxs.append(len(self.states) - 1)

        # auxiliary variables
        self.aux_vars = np.ones([len(self.states), num_aux_vars]) * 0.5
        self.aux_vars_m = np.zeros_like(self.aux_vars)
        self.aux_vars_v = np.zeros_like(self.aux_vars)
        self.aux_vars_c = np.zeros(self.aux_vars.shape, dtype = np.int64)

        # mnist dataset
        self.mnist_dataset = datasets.MNIST(
            root = "../../data/",
            train = self.train,
            transform = ToTensor(),
            download = False
        )
        self.digit2idxs = [[] for _ in range(self.num_digits)]
        for i in range(len(self.mnist_dataset)):
            class_id = self.mnist_dataset.targets[i]
            if class_id < self.num_digits:
                self.digit2idxs[class_id].append(i)

        # canvas
        self.canvas_size = (3 * self.digit_img_size[0] + 16, (self.max_num_disks + 1) * self.digit_img_size[1])
        self.ancher_xids = (4, self.digit_img_size[0] + 8, 2 * self.digit_img_size[0] + 12)
        self.ancher_yids = [i * self.digit_img_size[1] for i in range(self.max_num_disks)]

        self.img_resize = Resize(self.digit_img_size)

    def __len__(self):
        return len(self.states) if self.dataset_size is None else self.dataset_size

    def __getitem__(self, idx, seq_len = None):
        if seq_len is None:
            seq_len = self.seq_len
        if self.dataset_size is not None:
            idx = idx % len(self.states)

        # avoid sampling across trajectories
        for i in self.terminate_idxs:
            if idx <= i and idx + seq_len - 1 > i:
                idx = i - seq_len + 1
                break

        samples = [self._sample_single_state(idx+i) for i in range(seq_len)]

        state_imgs = torch.stack(
            [item[0] for item in samples], dim = 0
        )
        last_actions = torch.tensor([item[1] for item in samples])
        actions = torch.tensor([item[2] for item in samples])
        raw_states = [item[3] for item in samples]
        aux_vars = torch.from_numpy(self.aux_vars[idx:idx+seq_len,:].copy())
        
        return state_imgs, last_actions, actions, raw_states, aux_vars, idx

    def update_aux_vars(self, sample_idxs, grads, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, skip_first = True):
        if isinstance(sample_idxs, torch.Tensor):
            sample_idxs = sample_idxs.detach().cpu().numpy()

        seq_len = grads.shape[1]

        if skip_first:
            grads = grads[:,1:,:]

        for i, data_idx1 in enumerate(sample_idxs):
            data_idx2 = data_idx1 + seq_len
            if skip_first:
                data_idx1 += 1
            self.aux_vars_c[data_idx1:data_idx2, :] += 1
            self.aux_vars_m[data_idx1:data_idx2, :] = beta1 * self.aux_vars_m[data_idx1:data_idx2, :] + (1.0 - beta1) * grads[i,:,:]
            self.aux_vars_v[data_idx1:data_idx2, :] = beta2 * self.aux_vars_v[data_idx1:data_idx2, :] + (1.0 - beta2) * (grads[i,:,:]**2)
            m_hat = self.aux_vars_m[data_idx1:data_idx2, :] / (1.0 - np.power(beta1, self.aux_vars_c[data_idx1:data_idx2, :]))
            v_hat = self.aux_vars_v[data_idx1:data_idx2, :] / (1.0 - np.power(beta2, self.aux_vars_c[data_idx1:data_idx2, :]))

            self.aux_vars[data_idx1:data_idx2, :] += lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.aux_vars[data_idx1:data_idx2, :] = np.clip(self.aux_vars[data_idx1:data_idx2, :], 0.0, 1.0)

    def _sample_single_state(self, idx):
        state = self.states[idx]
        last_action = self.action_encoding[self.last_actions[idx]]
        action = self.action_encoding[self.actions[idx]]

        num_disks = sum(map(len, state))
        disk_mapping = sorted(random.sample(range(self.num_digits), num_disks))
        
        with torch.no_grad():
            state_raw = torch.zeros(3, self.max_num_disks + 1, dtype = torch.int64)
            for i in range(len(state[0])):
                state_raw[0,i] = state[0][i]
            for i in range(len(state[1])):
                state_raw[1,i] = state[1][i]
            for i in range(len(state[2])):
                state_raw[2,i] = state[2][i]

            state_raw[0,-1] = len(state[0])
            state_raw[1,-1] = len(state[1])
            state_raw[2,-1] = len(state[2])

        if self.sample_type == "full_img_state":
            with torch.no_grad():
                state_img = torch.zeros(self.canvas_size)
                for i in range(3):
                    pillar = state[i]
                    for j in range(len(pillar)):
                        digit_class = disk_mapping[pillar[j]]
                        img_idx = random.choice(self.digit2idxs[digit_class])
                        img = self.img_resize(self.mnist_dataset.data[img_idx, :, :].unsqueeze(0))
                        x = self.ancher_xids[i]
                        y = self.ancher_yids[j]

                        state_img[x:x+self.digit_img_size[0],y:y+self.digit_img_size[1]] = img

            return state_img, last_action, action, state_raw

        elif self.sample_type == "groundtruth_meta_state":
            with torch.no_grad():
                state_imgs = torch.zeros([3, self.digit_img_size[0], 2 * self.digit_img_size[1]])
                for i in range(3):
                    pillar = state[i]
                    if len(pillar) > 0:
                        digit_class = disk_mapping[pillar[-1]]
                        img_idx = random.choice(self.digit2idxs[digit_class])
                        img = self.img_resize(self.mnist_dataset.data[img_idx, :, :].unsqueeze(0))
                        state_imgs[i,:,:self.digit_img_size[1]] = img
                        state_imgs[(i-1)%3,:,self.digit_img_size[1]:] = img
            
            return state_imgs, last_action, action, state_raw
        
        else:
            raise NotImplementedError("Unknown `sample_type`: {}".format(self.sample_type))

    def _get_hanoi_trajectory(self, num_disks):
        if num_disks >= 2:
            env = Hanoi()
            traj, actions = env.record_traj(num_disks = num_disks)
            traj = traj[:-1]
            last_actions = [(1, 2)] + deepcopy(actions[:-1])
        elif num_disks == 1: # fake samples for number comparison
            traj = []
            for i in range(-1, 3):
                for j in range(-1, 3):
                    for k in range(-1, 3):
                        l = sorted([i, j, k])
                        if l[0] == -1 and l[2] >= 2:
                            continue
                        if l[1] == -1 and l[2] >= 1:
                            continue
                        if i != j and i != k and j != k:
                            p1 = [i] if i >= 0 else []
                            p2 = [j] if j >= 0 else []
                            p3 = [k] if k >= 0 else []
                            traj.append([p1, p2, p3])
            actions = [(1, 2) for _ in range(len(traj))]
            last_actions = [(1, 2) for _ in range(len(traj))]
        else:
            raise ValueError("Invalid `num_disks` = {}".format(num_disks))
        
        return traj, last_actions, actions

    @staticmethod
    def compare_top_disks(x, pillar1, pillar2):
        x1 = x[:,pillar1,:].clone()
        x2 = x[:,pillar2,:].clone()
        cond1 = (x1[:,-1] == 0)
        cond2 = (x2[:,-1] == 0)
        x1 = torch.where(
            cond1.unsqueeze(1), 
            10000, 
            x1.gather(1, torch.maximum(x1[:,-1].type(torch.int64).unsqueeze(1) - 1, torch.tensor(0, dtype = torch.int64)))
        )
        x2 = torch.where(
            cond2.unsqueeze(1), 
            10000, 
            x2.gather(1, torch.maximum(x2[:,-1].type(torch.int64).unsqueeze(1) - 1, torch.tensor(0, dtype = torch.int64)))
        )

        return (x1 < x2).type(torch.int64).squeeze(1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = HanoiSeqDataset(disk_size_candidates = [5], seq_len = 3, num_digits = 9)

    print(len(dataset))
    for i in range(len(dataset)):
        state, last_action, action, raw_states, aux_vars, sample_idx = dataset[i]
        state = state.detach().cpu().numpy()
        print(raw_states)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(state[0,:,:])
        plt.subplot(1, 3, 2)
        plt.imshow(state[1,:,:])
        plt.subplot(1, 3, 3)
        plt.imshow(state[2,:,:])
        plt.show()
