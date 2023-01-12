# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
import sys
#sys.path.append("../../../")
sys.path.append("./")
import argparse
from tensorboardX import SummaryWriter
from datasets import get_mnist_dataloader
torch.multiprocessing.set_sharing_strategy('file_system')

# +
# parameters 
base = 10
batch_size = 256
num_samples, num_samples_test = 10000, 6000
pre_train_num = 10
seq_len = 3
seq_len_test = 5
num_auxs = 2
num_epochs = 70  
iters_per_epoch = num_samples // batch_size 
iters_per_epoch_test = num_samples_test // batch_size

pc_update_freq = 1
nn_update_freq = 1
device = torch.device("cuda:0")
celoss = nn.CrossEntropyLoss()

logger = SummaryWriter(log_dir="log")
# -

# parse setting
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default=256)
parser.add_argument('--base', default=10)
parser.add_argument('-t', '--test_len', default=5)
args = parser.parse_args()
print(args)
base = int(args.base)
batch_size = int(args.batch_size)
seq_len_test = int(args.test_len)



class SeqAddDataset():
    def __init__(self, data_loader, num_samples, base, num_auxs, seq_len):
        self.num_samples = num_samples
        self.base = base
        self.num_auxs = num_auxs
        self.seq_len = seq_len
        
        # Generate dataset
        self.dataset = []
        iterator = iter(data_loader)
        for sample_idx in range(self.num_samples):
            try:
                imgs, labels = iterator.next()
            except StopIteration:
                iterator = iter(data_loader)
                imgs, labels = iterator.next()
                
            if imgs.size(0) < seq_len * 2:
                iterator = iter(data_loader)
                imgs, labels = iterator.next()
                
            imgs, labels = imgs[:seq_len * 2, :, :, :], labels[:seq_len * 2]
            imgs = imgs.reshape(seq_len, 2, 1, 28, 28)
            labels = labels.reshape(seq_len, 2)
            
            actual_labels = torch.zeros([seq_len], dtype = torch.long)
            carry = 0
            for seq_idx in range(seq_len):
                res = labels[seq_idx, 0] + labels[seq_idx, 1] + carry
                actual_labels[seq_idx] = (res % base)
                carry = (res // base)
                
            aux_vars = torch.zeros([seq_len, num_auxs * 2]) + 0.5
            aux_vars[0, :num_auxs] = 0.0
            aux_vars[0, 0] = 1.0
            aux_vars_m = torch.zeros([seq_len, num_auxs * 2]) # Used by Adam
            aux_vars_v = torch.zeros([seq_len, num_auxs * 2]) # Used by Adam
            num_iters = 0
            
            self.dataset.append([imgs, actual_labels, aux_vars, labels, 
                                 aux_vars_m, aux_vars_v, num_iters])
            
        self.last_m = None
        
        self.m = None
        self.batch_start = 0
            
    def get_data(self, batch_size = 32, get_individual_labels = False, get_all = False):
        if self.m is None or self.batch_start + batch_size > self.num_samples:
            self.m = np.random.permutation(self.num_samples)
            self.batch_start = 0
        m = self.m[self.batch_start:self.batch_start+batch_size]
        self.batch_start += batch_size
        self.last_m = m
        
        with torch.no_grad():
            images = torch.empty([batch_size, self.seq_len, 2, 1, 28, 28], dtype = torch.float32)
            labels = torch.empty([batch_size, self.seq_len], dtype = torch.long)
            aux_vars = torch.empty([batch_size, self.seq_len, self.num_auxs * 2], dtype = torch.float32)
            indiv_labels = torch.empty([batch_size, self.seq_len, 2], dtype = torch.long)
            aux_vars_m = torch.empty([batch_size, self.seq_len, self.num_auxs * 2], dtype = torch.float32)
            aux_vars_v = torch.empty([batch_size, self.seq_len, self.num_auxs * 2], dtype = torch.float32)
            num_iters = torch.empty([batch_size], dtype = torch.long)
            
            for idx in range(batch_size):
                data = self.dataset[m[idx]]
                images[idx, :, :, :, :, :] = data[0]
                labels[idx, :] = data[1]
                aux_vars[idx, :, :] = data[2]
                indiv_labels[idx, :, :] = data[3]
                aux_vars_m[idx, :, :] = data[4]
                aux_vars_v[idx, :, :] = data[5]
                num_iters[idx] = data[6]
                
        if get_all:
            return images, labels, aux_vars, indiv_labels, aux_vars_m, aux_vars_v, num_iters
        elif get_individual_labels:
            return images, labels, aux_vars, indiv_labels
        else:
            return images, labels, aux_vars
    
    def update_aux_vars(self, aux_vars, aux_vars_m, aux_vars_v, num_iters):
        assert self.last_m.shape[0] == aux_vars.shape[0]
        batch_size = self.last_m.shape[0]
        
        for idx in range(batch_size):
            data_idx = self.last_m[idx]
            self.dataset[data_idx][2][:, :] = torch.from_numpy(aux_vars[idx, :, :])
            self.dataset[data_idx][4][:, :] = torch.from_numpy(aux_vars_m[idx, :, :])
            self.dataset[data_idx][5][:, :] = torch.from_numpy(aux_vars_v[idx, :, :])
            self.dataset[data_idx][6] = num_iters[idx]



# +
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        #return logits, probas
        return logits


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=10,
                   grayscale=True)
    return model


# +
# Import julia
from julia.api import Julia
jl = Julia(compiled_modules = False) # Low-level interface of Julia
try:
    jl.eval('using LogicCircuits') # For some reason this will raise an error
except:
    pass
try:
    jl.eval('include("recursive_adder_helper.jl")')
except:
    pass

# Julia pc helper functions
jl.eval('include("recursive_adder_helper.jl")')
pc_gradients = jl.eval('pc_gradients')
pc_condprob = jl.eval('pc_condprob')
get_marupdate_nodes = jl.eval('get_marupdate_nodes')
pc_em_update = jl.eval('pc_em_update!')
pc_em_update_with_inc = jl.eval('pc_em_update_with_inc!')
pc_em_update_with_exc = jl.eval('pc_em_update_with_exc!')
fix_pc_input_weights = jl.eval('fix_pc_input_weights!')
fix_pc_input_weights2 = jl.eval('fix_pc_input_weights2!')
get_tautology_pc = jl.eval('get_tautology_pc')
update_aux_variables = jl.eval('update_aux_variables')
update_aux_variables_adam = jl.eval('update_aux_variables_adam')
print_pc = jl.eval('print_pc')
# lc function
jl.eval('include("soft_rule_learner.jl")')
generate_true_formula_categorical = jl.eval('generate_true_formula_categorical')
model_count = jl.eval('model_count')
rule_learning = jl.eval('rule_learning')
check_precision_base = jl.eval('check_precision_base')


# +
def init_model(base):

    model = resnet18(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    model.train()
    return model, optimizer

def pretrain_model(model, optimizer, pretrain_loader, num_epochs = 200, verbose = False):
    total_step = len(pretrain_loader)
    for epoch in range(num_epochs):
        cumloss = 0.0
        cumaccu = 0.0
        for i, (images, labels) in enumerate(pretrain_loader):
            images, labels = images.to(device), labels.to(device)

            pred_y = model(images)
            loss = celoss(pred_y, labels)

            cumloss += loss.detach().cpu().numpy()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                accu = (torch.argmax(pred_y, dim = 1) == labels).float().mean()
                cumaccu += accu.detach().cpu().numpy()

        if verbose:
            print("Epoch {} - Aveg loss: {:.6f}; Train accu: {:.2f}%".format(
                  epoch + 1, cumloss / total_step, 100 * cumaccu / total_step))

    return model




# +
train_loader = get_mnist_dataloader(train = True, batch_size = 2 * seq_len,
                                    included_classes = [i for i in range(base)], root = "../data") 
test_loader = get_mnist_dataloader(train = False, batch_size = 2 * seq_len_test,
                                   included_classes = [i for i in range(base)], root = "../data")
pretrain_loader = get_mnist_dataloader(train = True, batch_size = batch_size,
                                       included_classes = [i for i in range(base)], root = "../data",
                                       num_samples = pre_train_num * base, balance_class = True)
paired_dataloader = SeqAddDataset(train_loader, num_samples = 10000, base = base,
                                  num_auxs = num_auxs, seq_len = seq_len)

paired_dataloader_test = SeqAddDataset(test_loader, num_samples = 6000, base = base,
                                   num_auxs = num_auxs, seq_len = seq_len_test)

#paired_dataloader_test_train = SeqAddDataset(train_loader, num_samples = 6000, base = base,
#                                   num_auxs = num_auxs, seq_len = seq_len_test)


# -
def dataset_evaluation(iters_per_epoch_test, paired_dataloader_test, seq_len_test, model, batch_size):
    cumaccu_test = 0.0
    cumaccu_test_seq = 0.0
    with torch.no_grad():
        for iter_idx in range(iters_per_epoch_test):
            imgs, labels, _, _, _, _, \
                _ = paired_dataloader_test.get_data(batch_size = batch_size, get_all = True)
            imgs = imgs.to(device)

            pred_y = torch.sigmoid(model(imgs.reshape(batch_size * seq_len_test * 2, 1, 28, 28)))
            pred_y = pred_y.reshape(batch_size, seq_len_test, 2 * base) # (B, seq_len, 2 * base)

            pred_labels = torch.zeros([batch_size, seq_len_test, base]) + 0.5
            aux_vars = torch.zeros([batch_size, seq_len_test, 2 * num_auxs]) + 0.5
            aux_vars[:, 0, :num_auxs] = 0.0 # Initialize auxiliary variables
            aux_vars[:, 0, 0] = 1.0 # Initialize auxiliary variables
            pc_inputs = torch.cat(
                (pred_y.detach().cpu(), pred_labels, aux_vars),
                dim = 2
            ).reshape(batch_size, seq_len_test, 3 * base + 2 * num_auxs).detach().cpu().numpy()

            cond_vars = [i for i in range(2*base+1, 3*base+1)] + \
                [i for i in range(3*base+num_auxs+1, 3*base+2*num_auxs+1)]
            for seq_idx in range(seq_len_test):
                pc_outs = pc_condprob(pc, pc_inputs[:,seq_idx,:], cond_vars)
                pc_inputs[:,seq_idx,2*base:3*base] = pc_outs[:,:base]
                pc_inputs[:,seq_idx,3*base+num_auxs:] = pc_outs[:,base:]
                if seq_idx < seq_len_test - 1:
                    pc_inputs[:,seq_idx+1,3*base:3*base+num_auxs] = pc_outs[:,base:]
            preds = torch.from_numpy(pc_inputs[:,:,2*base:3*base])

            acc_flag = (torch.argmax(preds, dim = 2, keepdims = False) == labels).float()
            accu = torch.mean(acc_flag)
            accu_seq = torch.mean(torch.prod(acc_flag, 1))
            
            cumaccu_test_seq += accu_seq.detach().cpu().numpy()
            cumaccu_test += accu.detach().cpu().numpy()
    return cumaccu_test, cumaccu_test_seq


model, optimizer = init_model(base)
model = pretrain_model(model, optimizer, pretrain_loader, num_epochs = 30, verbose = True)
pc = get_tautology_pc(base, num_auxs = num_auxs)
marupdate_ns = get_marupdate_nodes(pc)

train_data = np.zeros([0, 3 * base + 4])
iter_count = 0
for epoch in range(num_epochs):
    cumloss = 0.0
    cumaccu = 0.0
    for iter_idx in range(iters_per_epoch):
        iter_count += 1

        imgs, labels, aux_vars, actual_labels, \
            aux_vars_m, aux_vars_v, num_iters = paired_dataloader.get_data(
            batch_size = batch_size, get_individual_labels = True, get_all = True
        )
        imgs = imgs.to(device)
        actual_labels = F.one_hot(actual_labels.reshape(
            batch_size, seq_len, 2), num_classes = base).reshape(batch_size, seq_len, 2 * base)
        pred_y = torch.sigmoid(model(imgs.reshape(batch_size * seq_len * 2, 1, 28, 28)))
        pred_y = pred_y.reshape(batch_size, seq_len, 2 * base) # (B, seq_len, 2 * base)
        with torch.no_grad():
            labels = F.one_hot(labels.reshape(batch_size, seq_len), num_classes = base) # (B, seq_len, base)
            aux_vars = aux_vars.reshape(batch_size, seq_len, 2 * num_auxs) # (B, seq_len, 2 * num_auxs)
            pc_inputs = torch.cat(
                (pred_y.detach().cpu(), labels.float(), aux_vars),
                dim = 2
            ).reshape(batch_size * seq_len, 3 * base + 2 * num_auxs).detach().cpu().numpy()
            
            pc_grads1, pc_lls, pc_flow = pc_gradients(pc, pc_inputs, log_grad = True)
            
            _, _, pc_flow2 = pc_gradients(pc, pc_inputs, log_grad = True, no_prob = True,
                                          marginalized_vars = [i for i in range(3*base+1, 3*base+2*num_auxs+1)])
            pc_grads2, _, _ = pc_gradients(pc, pc_inputs, log_grad = True,
                                          marginalized_vars = [i for i in range(2*base+1, 3*base+1)])
            pc_grads3, _, _ = pc_gradients(pc, pc_inputs, log_grad = True,
                                        marginalized_vars = [i for i in range(2*base+1, 3*base+2*num_auxs+1)])
            
            grads = pc_grads1 - pc_grads2
            grads_for_nn = pc_grads1 - pc_grads3

            # Update the PC
            if iter_count % pc_update_freq == 0:
                pc_em_update_with_exc(pc, pc_flow, step_size = 0.02, exclude_nodes = marupdate_ns)
                pc_em_update_with_inc(pc, pc_flow2, step_size = 0.02, update_nodes = marupdate_ns)
                fix_pc_input_weights2(pc)

            pc_grads = torch.from_numpy(grads_for_nn[:, :2*base]).to(
                device).reshape(batch_size, seq_len, 2*base)

        # Update neural network
        if iter_count % nn_update_freq == 0:
            optimizer.zero_grad()
            pred_y.backward(gradient = -pc_grads)
            optimizer.step()

        # Update auxiliary variables
        with torch.no_grad():
            aux_vars = aux_vars.reshape(batch_size, seq_len, 2 * num_auxs).detach().cpu().numpy()
            aux_vars_m = aux_vars_m.reshape(batch_size, seq_len, 2 * num_auxs).detach().cpu().numpy()
            aux_vars_v = aux_vars_v.reshape(batch_size, seq_len, 2 * num_auxs).detach().cpu().numpy()
            num_iters = num_iters.reshape(batch_size).detach().cpu().numpy()
            grads = grads.reshape(batch_size, seq_len, 3*base+2*num_auxs)[:,:,3*base:]

            aux_vars, aux_vars_m, aux_vars_v, \
                num_iters = update_aux_variables_adam(aux_vars, aux_vars_m, aux_vars_v,
                                                      num_iters, grads, lr = 0.01)
            paired_dataloader.update_aux_vars(aux_vars, aux_vars_m, aux_vars_v, num_iters)
        
        if epoch == num_epochs - 1:
            train_data = np.concatenate((train_data, pc_inputs), axis = 0)
        
        cumloss += -pc_lls.mean()

        # Compute accuracy (not entirely correct)
        with torch.no_grad():
            pred_out = pc_condprob(pc, pc_inputs, [i for i in range(2*base+1, 3*base+1)])
            pred_out = torch.from_numpy(pred_out).reshape(batch_size, seq_len, base)

            accu = torch.mean((torch.argmax(pred_out, dim = 2) == torch.argmax(labels, dim = 2)).float())
            cumaccu += accu.detach().cpu().numpy()
    
    # Evaluation
    if epoch % 10 == 9:
        batch_size_test = 256
        iters_per_epoch_test = 6000 // batch_size_test
        cumaccu_test, cumaccu_test_seq = dataset_evaluation(iters_per_epoch_test, paired_dataloader_test, seq_len_test, model, batch_size_test)    
    else:
        cumaccu_test, cumaccu_test_seq = 0.0, 0.0
    print("Epoch {} - Aveg loss: {:.6f}; Train accu: {:.2f}%; Test accu: {:.2f}%; Test_seq Seq accu: {:.2f}%;".format(
          epoch + 1, cumloss / iters_per_epoch, 100 * cumaccu / iters_per_epoch,
          100 * cumaccu_test / iters_per_epoch_test, 100 * cumaccu_test_seq/ iters_per_epoch_test,)
         )

num_cats = [base, base, base, 2, 2]
lc = generate_true_formula_categorical(num_cats, type = "flat")

model_count(lc)

count = 0
for i in range(29952):
    x = np.argmax(train_data[i,:10]) + np.argmax(train_data[i,10:20]) + np.argmax(train_data[i,30:32]) 
    a = x // 10 
    b = x % 10 
    if b == np.argmax(train_data[i,20:30]) and a == np.argmax(train_data[i,32:34]):
        count += 1

lc_new = lc


lc_new = rule_learning(lc_new, train_data, user_threshold = 0.002, num_targets = base + 4, maxiters = 2, verbose = True) 

check_precision_base(lc_new, 10)


