import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


class VisAttnModel(nn.Module):
    def __init__(self, num_classes, feature_dim = 64, attn_dim = 32):
        super(VisAttnModel, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, feature_dim, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d((5, 5))
        )

        self.attn_layer = nn.MultiheadAttention(embed_dim = self.attn_dim, num_heads = 2)
        self.query_vector = nn.Parameter(torch.randn(1, self.attn_dim).type(torch.float32), requires_grad = True)
        self.key_net = nn.Sequential(
            nn.Linear(feature_dim, self.attn_dim),
            nn.ReLU(),
            nn.Linear(self.attn_dim, self.attn_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.attn_dim),
            nn.ReLU(),
            nn.Linear(self.attn_dim, self.attn_dim)
        )

        self.pred_model = nn.Sequential(
            nn.Linear(self.attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, get_attn_map = False):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        b = x.size(0)
        x = self.feature_net(x) # (B, attn_dim, H, W) 
        h, w = x.size(2), x.size(3)

        pos_encoding = self._positionalencoding2d(self.feature_dim, h, w).to(x.device)
        x = (x + pos_encoding.unsqueeze(0)).reshape(b, self.feature_dim, h * w).permute(2, 0, 1)

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


class BindingLayer_3(nn.Module):
    def __init__(self, num_binds = 3, num_outs = 2):
        super(BindingLayer_3, self).__init__()

        self.num_binds = num_binds
        self.models = []
        for i in range(self.num_binds):
            model = VisAttnModel(num_classes = num_outs)
            self.add_module("model_{}".format(i), model)
            self.models.append(model)

    def forward(self, x):
        y = self.models[0](x)
        for i in range(1, self.num_binds):
            h = self.models[i](x)
            y = torch.cat((y, h), dim = 1)

        return y


# +
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../env")

    from hanoi_dataset import HanoiDataset
    
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    batch_size = 64

    tr_dataset = HanoiDataset(disk_size_candidates = [5], num_digits = 9, dataset_size = 20000)
    train_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)

    ts_dataset = HanoiDataset(disk_size_candidates = [7], num_digits = 9, dataset_size = 10000)
    test_loader = DataLoader(ts_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda:0")
    
    #model = BindingLayer_3(num_outs = 2).to(device)
    model = VisAttnModel(num_classes = 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 5e-4)

    for epoch_idx in range(50):
        total_loss = 0.0
        total_acc = 0.0
        for state_img, last_action, action, state_raw in train_loader:
            state_img = state_img.to(device)
            labels = HanoiDataset.compare_top_disks(state_raw, 0, 1).to(device)
            b = state_img.size(0)
            #labels = torch.stack(
            #    (HanoiDataset.compare_top_disks(state_raw, 0, 1),
            #     HanoiDataset.compare_top_disks(state_raw, 1, 2),
            #     HanoiDataset.compare_top_disks(state_raw, 2, 0)),
            #    dim = 1
            #).to(device)
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 3 * 2).float()
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 2).float()
            
            logits = model(state_img)
            loss = model.get_loss(logits, labels)
            #loss = bce_loss(logits, labels)
            #loss = ce_loss(logits, labels)
            
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









# +
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../env")
    import itertools
    from hanoi_dataset import HanoiDataset
    
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    batch_size = 64

    tr_dataset = HanoiDataset(disk_size_candidates = [5], num_digits = 9, dataset_size = 20000)
    train_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)

    ts_dataset = HanoiDataset(disk_size_candidates = [7], num_digits = 9, dataset_size = 10000)
    test_loader = DataLoader(ts_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda:0")
    
    #model = BindingLayer_3(num_outs = 2).to(device)
    model1 = VisAttnModel(num_classes = 2).to(device)
    model2 = VisAttnModel(num_classes = 2).to(device)
    model3 = VisAttnModel(num_classes = 2).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr = 5e-4)
    optimizer = optim.Adam(itertools.chain(model1.parameters(), model2.parameters(), model3.parameters()), lr = 5e-4)
    
    for epoch_idx in range(50):
        total_loss = 0.0
        total_acc = 0.0
        for state_img, last_action, action, state_raw in train_loader:
            state_img = state_img.to(device)
            labels1 = HanoiDataset.compare_top_disks(state_raw, 0, 1).to(device)
            labels2 = HanoiDataset.compare_top_disks(state_raw, 1, 2).to(device)
            labels3 = HanoiDataset.compare_top_disks(state_raw, 2, 0).to(device)
            
            b = state_img.size(0)
            #labels = torch.stack(
            #    (HanoiDataset.compare_top_disks(state_raw, 0, 1),
            #     HanoiDataset.compare_top_disks(state_raw, 1, 2),
            #     HanoiDataset.compare_top_disks(state_raw, 2, 0)),
            #    dim = 1
            #).to(device)
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 3 * 2).float()
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 2).float()
            
            logits1 = model1(state_img)
            logits2 = model2(state_img)
            logits3 = model3(state_img)
            
            loss = model1.get_loss(logits1, labels1) + model2.get_loss(logits2, labels2) + model3.get_loss(logits3, labels3)
            #loss = bce_loss(logits, labels)
            #loss = ce_loss(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_acc += (torch.argmax(logits1, dim = 1) == labels1).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits2, dim = 1) == labels2).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits3, dim = 1) == labels3).float().mean().detach().cpu().numpy()

            
        print("Epoch {} - average train loss: {:.4f}; average acc: {:.4f}".format(
            epoch_idx + 1, total_loss / len(train_loader), total_acc / (len(train_loader) * 3)
        ))

        total_acc = 0.0
        for state_img, last_action, action, state_raw in test_loader:
            state_img = state_img.to(device)
            labels1 = HanoiDataset.compare_top_disks(state_raw, 0, 1).to(device)
            labels2 = HanoiDataset.compare_top_disks(state_raw, 1, 2).to(device)
            labels3 = HanoiDataset.compare_top_disks(state_raw, 2, 0).to(device)
            
            logits1 = model1(state_img)
            logits2 = model2(state_img)
            logits3 = model3(state_img)
            
            total_acc += (torch.argmax(logits1, dim = 1) == labels1).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits2, dim = 1) == labels2).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits3, dim = 1) == labels3).float().mean().detach().cpu().numpy()

            
        print("        - average test acc: {:.4f}".format(total_acc / (3 * len(test_loader))))











# +
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../env")
    import itertools
    from hanoi_dataset import HanoiDataset_old
    
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    batch_size = 64

    tr_dataset = HanoiDataset_old(disk_size_candidates = [5], num_digits = 9, dataset_size = 20000)
    train_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)

    ts_dataset = HanoiDataset_old(disk_size_candidates = [7], num_digits = 9, dataset_size = 10000)
    test_loader = DataLoader(ts_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda:0")
    
    #model = BindingLayer_3(num_outs = 2).to(device)
    model1 = VisAttnModel(num_classes = 2).to(device)
    model2 = VisAttnModel(num_classes = 2).to(device)
    model3 = VisAttnModel(num_classes = 2).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr = 5e-4)
    optimizer = optim.Adam(itertools.chain(model1.parameters(), model2.parameters(), model3.parameters()), lr = 5e-4)
    
    for epoch_idx in range(50):
        total_loss = 0.0
        total_acc = 0.0
        for state_img, last_action, action, state_raw in train_loader:
            state_img = state_img.to(device)
            labels1 = HanoiDataset_old.compare_top_disks(state_raw, 0, 1).to(device)
            labels2 = HanoiDataset_old.compare_top_disks(state_raw, 1, 2).to(device)
            labels3 = HanoiDataset_old.compare_top_disks(state_raw, 2, 0).to(device)
            
            b = state_img.size(0)
            #labels = torch.stack(
            #    (HanoiDataset.compare_top_disks(state_raw, 0, 1),
            #     HanoiDataset.compare_top_disks(state_raw, 1, 2),
            #     HanoiDataset.compare_top_disks(state_raw, 2, 0)),
            #    dim = 1
            #).to(device)
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 3 * 2).float()
            #labels = F.one_hot(labels, num_classes = 2).reshape(b, 2).float()
            
            logits1 = model1(state_img)
            logits2 = model2(state_img)
            logits3 = model3(state_img)
            
            loss = model1.get_loss(logits1, labels1) + model2.get_loss(logits2, labels2) + model3.get_loss(logits3, labels3)
            #loss = bce_loss(logits, labels)
            #loss = ce_loss(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_acc += (torch.argmax(logits1, dim = 1) == labels1).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits2, dim = 1) == labels2).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits3, dim = 1) == labels3).float().mean().detach().cpu().numpy()

            
        print("Epoch {} - average train loss: {:.4f}; average acc: {:.4f}".format(
            epoch_idx + 1, total_loss / len(train_loader), total_acc / (len(train_loader) * 3)
        ))

        total_acc = 0.0
        for state_img, last_action, action, state_raw in test_loader:
            state_img = state_img.to(device)
            labels1 = HanoiDataset_old.compare_top_disks(state_raw, 0, 1).to(device)
            labels2 = HanoiDataset_old.compare_top_disks(state_raw, 1, 2).to(device)
            labels3 = HanoiDataset_old.compare_top_disks(state_raw, 2, 0).to(device)
            
            logits1 = model1(state_img)
            logits2 = model2(state_img)
            logits3 = model3(state_img)
            
            total_acc += (torch.argmax(logits1, dim = 1) == labels1).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits2, dim = 1) == labels2).float().mean().detach().cpu().numpy()
            total_acc += (torch.argmax(logits3, dim = 1) == labels3).float().mean().detach().cpu().numpy()

            
        print("        - average test acc: {:.4f}".format(total_acc / (3 * len(test_loader))))










# -

    for state_img, last_action, action, state_raw in test_loader:
        state_img = state_img.to(device)

        logits, attn_map = model(state_img, get_attn_map = True)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(state_img[0,:,:].detach().cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(attn_map[0,:,:].detach().cpu().numpy())
        plt.show()


#     for state_img, last_action, action, state_raw in test_loader:
#         state_img = state_img.to(device)
#
#         logits, attn_map = model(state_img, get_attn_map = True)
#         print(state_raw[0])
#         plt.figure()
#         #plt.subplot(1, 2, 1)
#         #plt.imshow(state_img[0,:,:].detach().cpu().numpy())
#         #plt.subplot(1, 2, 2)
#         plt.xticks([])
#         plt.yticks([])
#         #plt.imshow(attn_map[0,:,:].detach().cpu().numpy())
#         plt.imshow(np.rot90(attn_map[0,:,:].detach().cpu().numpy()))
#         plt.show()

# np.rot90(attn_map[0,:,:].detach().cpu().numpy(),1)

# logits 

# labels

class ResNet50ViT(nn.Module):
    def __init__(self, *, img_dim, pretrained_resnet=False,
                 resnet_layers=5,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=64,
                 dropout=0, transformer=None, classification=True
                 ):
        """
        ResNet50 + ViT for image classification
        Args:
            img_dim: the spatial image size
            pretrained_resnet: wheter to load pretrained weight from torch vision
            resnet_layers: use 5 or 6. the layer to keep from the resnet 50 backbone
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
        """
        super().__init__()
        assert 5 <= resnet_layers <= 6, f'Inser 5 or 6 resnet layers to keep'
        resnet_channels = 256 if resnet_layers == 5 else 512
        feat_dim = img_dim // 4 if resnet_layers == 5 else img_dim // 8

        resnet_layers = list(resnet50(pretrained=pretrained_resnet).children())[:resnet_layers]
        self.img_dim = img_dim

        res50 = nn.Sequential(*resnet_layers)

        vit = ViT(img_dim=feat_dim, in_channels=resnet_channels, patch_dim=1,
                  num_classes=num_classes, dim_linear_block=dim_linear_block, dim=dim,
                  dim_head=dim_head, dropout=dropout, transformer=transformer,
                  classification=classification, heads=heads, blocks=blocks)
        self.model = nn.Sequential(res50, vit)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == 3, f'Insert input with 3 inp channels, {c} received.'
        assert self.img_dim == h == w, f'Insert input with {self.img_dim} dimensions.'
        return self.model(x)
