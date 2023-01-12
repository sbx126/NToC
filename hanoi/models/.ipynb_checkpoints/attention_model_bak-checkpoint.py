import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


class VisAttnModel(nn.Module):
    def __init__(self, num_classes, attn_dim = 64):
        super(VisAttnModel, self).__init__()

        self.num_classes = num_classes
        self.attn_dim = attn_dim

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 128, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d((5, 5))
        )

        self.attn_layer = nn.MultiheadAttention(embed_dim = self.attn_dim, num_heads = 2)
        self.query_vector = nn.Parameter(torch.randn(1, self.attn_dim).type(torch.float32), requires_grad = True)
        self.key_net = nn.Sequential(
            nn.Linear(128, self.attn_dim),
            nn.ReLU(),
            nn.Linear(self.attn_dim, self.attn_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, self.attn_dim),
            nn.ReLU(),
            nn.Linear(self.attn_dim, self.attn_dim)
        )

        self.pred_model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.Linear(64, self.num_classes)
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, get_attn_map = False):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        b = x.size(0)
        x = self.feature_net(x) # (B, attn_dim, H, W) 
        h, w = x.size(2), x.size(3)

        pos_encoding = self._positionalencoding2d(128, h, w).to(x.device)
        x = (x + pos_encoding.unsqueeze(0)).reshape(b, 128, h * w).permute(2, 0, 1)

        query = self.query_vector.unsqueeze(0).repeat(1, b, 1) # (1, B, attn_dim)
        key = self.key_net(x) # (H*W, B, attn_dim)
        value = self.value_net(x) # (H*W, B, attn_dim)
        x, attn_map = self.attn_layer(query, key, value) # (1, B, attn_dim)

        logits = self.pred_model(x.squeeze(0))

        if not get_attn_map:
            return logits
        else:
            return logits, attn_map.reshape(b, h, w)

    def get_loss(self, logits, targets):
        return self.ce_loss(logits, targets)

    def _positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe



if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../env")
    
    from hanoi_dataset import HanoiDataset

    batch_size = 64

    tr_dataset = HanoiDataset(disk_size_candidates = [3, 5], num_digits = 9, dataset_size = 20000)
    train_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)

    ts_dataset = HanoiDataset(disk_size_candidates = [7, 9], num_digits = 9, dataset_size = 10000)
    test_loader = DataLoader(ts_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda:0")

    model = VisAttnModel(num_classes = 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    for epoch_idx in range(40):
        total_loss = 0.0
        total_acc = 0.0
        for state_img, last_action, action, state_raw in train_loader:
            state_img = state_img.to(device)
            labels = HanoiDataset.compare_top_disks(state_raw, 0, 1).to(device)

            logits = model(state_img)
            loss = model.get_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_acc += (torch.argmax(logits, dim = 1) == labels).float().mean().detach().cpu().numpy()
        
        print("Epoch {} - average train loss: {:.4f}; average acc: {:.4f}".format(
            epoch_idx + 1, total_loss / len(train_loader), total_acc / len(train_loader)
        ))

        total_acc = 0.0
        for state_img, last_action, action, state_raw in test_loader:
            state_img = state_img.to(device)
            labels = HanoiDataset.compare_top_disks(state_raw, 0, 1).to(device)

            logits = model(state_img)
            total_acc += (torch.argmax(logits, dim = 1) == labels).float().mean().detach().cpu().numpy()

        print("        - average test acc: {:.4f}".format(total_acc / len(test_loader)))

    for state_img, last_action, action, state_raw in test_loader:
        state_img = state_img.to(device)

        logits, attn_map = model(state_img, get_attn_map = True)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(state_img[0,:,:].detach().cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(attn_map[0,:,:].detach().cpu().numpy())
        plt.show()


