from copy import deepcopy
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import torch
from torch.utils.data import Dataset
import random
from copy import deepcopy

from hanoi_env import Hanoi


class HanoiDataset(Dataset):
    def __init__(self, disk_size_candidates, num_digits = 5, digit_img_size = (28, 28), train = True, 
                 dataset_size = None, sample_type = "full_img_state"):
        super(HanoiDataset, self).__init__()

        self.disk_size_candidates = disk_size_candidates
        self.num_digits = num_digits
        self.digit_img_size = digit_img_size
        self.train = train
        self.dataset_size = dataset_size
        self.sample_type = sample_type

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
        for num_disks in self.disk_size_candidates:
            s, la, a = self._get_hanoi_trajectory(num_disks)
            self.states.extend(s)
            self.last_actions.extend(la)
            self.actions.extend(a)

        # mnist dataset
        self.mnist_dataset = datasets.MNIST(
            root = "../../data/",
            train = self.train,
            transform = ToTensor(),
            download = True
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

    def __getitem__(self, idx):
        if self.dataset_size is not None:
            idx = idx % len(self.states)

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

    debug = 0

    if debug == 0:
        dataset = HanoiDataset(disk_size_candidates = [1], num_digits = 9)

        for i in range(len(dataset)):
            state_imgs, last_action, action, state_raw = dataset[i]
            state_imgs = state_imgs.detach().cpu().numpy()
            #print(img_digit)
            plt.figure()
            plt.imshow(state_imgs)
            plt.show()
    
    elif debug == 1:
        dataset = HanoiDataset(disk_size_candidates = [1], num_digits = 9, sample_type = "groundtruth_meta_state")

        print(len(dataset))
        for i in range(len(dataset)):
            state_imgs, last_action, action, state_raw = dataset[i]

            print(HanoiDataset.compare_top_disks(state_raw.unsqueeze(0), 0, 1))
            print(HanoiDataset.compare_top_disks(state_raw.unsqueeze(0), 1, 2))
            print(HanoiDataset.compare_top_disks(state_raw.unsqueeze(0), 2, 0))
            print("======")
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(state_imgs[0,:,:])
            plt.subplot(1, 3, 2)
            plt.imshow(state_imgs[1,:,:])
            plt.subplot(1, 3, 3)
            plt.imshow(state_imgs[2,:,:])
            plt.show()


