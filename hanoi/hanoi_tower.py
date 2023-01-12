import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import functools
import os
import sys
sys.path.append("./")
sys.path.append("./env")
sys.path.append("./models")
sys.path.append("./utils")

from hanoi_dataset import HanoiDataset
from hanoi_seq_dataset import HanoiSeqDataset
from attention_model import VisAttnModel
from ProgressBar import ProgressBar

from julia import Main as JL

JL.include("prep.jl")

#num_samples = 1000
batch_size = 128
num_epochs = 40
predict_dim = 6
train_len = 36
test_len = 5

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train_size', nargs='+', type=int, default=[3, 5])
parser.add_argument('-ts', '--test_size', nargs='+', type=int, default=[7, 9])
parser.add_argument('-trl', '--train_len', type=int, default=3)
parser.add_argument('-tsl', '--test_len', type=int, default=5)
parser.add_argument('-ps', '--pretrain_samples', type=int, default=500)
parser.add_argument('-e', '--num_epochs', type=int, default=40)
args = parser.parse_args()
print(args)
train_size = args.train_size 
test_size = args.test_size
train_len = args.train_len
test_len = args.test_len
num_samples = args.pretrain_samples
num_epochs = args.num_epochs


class BindingLayer(nn.Module):
    def __init__(self, num_binds = 1, num_outs = 6):
        super(BindingLayer, self).__init__()

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


def pretrain(model, optimizer, device, num_samples = num_samples, num_epochs = 200, use_ckpt = False):
    if use_ckpt and os.path.exists("./ckpts/seq_attn_pc.pt"):
        model.load_state_dict(torch.load("./ckpts/seq_attn_pc.pt"))
        print("> Pretrained model loaded")
        return

    tr_dataset = HanoiDataset(disk_size_candidates = [1], num_digits = 9, dataset_size = num_samples, sample_type = "full_img_state")
    train_loader = DataLoader(tr_dataset, batch_size = 32, shuffle = True)

    bce_loss = nn.BCELoss()

    for epoch_idx in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        for state_img, last_action, action, state_raw in train_loader:
            b = state_img.size(0)
            state_img = state_img.to(device)

            actual_symbols = torch.stack(
                (HanoiSeqDataset.compare_top_disks(state_raw, 0, 1),
                 HanoiSeqDataset.compare_top_disks(state_raw, 1, 2),
                 HanoiSeqDataset.compare_top_disks(state_raw, 2, 0)),
                dim = 1
            ).to(device)
            actual_symbols_onehot = F.one_hot(actual_symbols, num_classes = 2).reshape(b, 3 * 2).float()

            logits = model(state_img)
            loss = bce_loss(torch.sigmoid(logits), actual_symbols_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_acc += ((logits > 0) == (actual_symbols_onehot > 0.5)).float().mean().detach().cpu().numpy()
        
        if epoch_idx % 10 == 9:
            print("Pretain epoch {} - average train loss: {:.4f}; average acc: {:.4f}".format(
                epoch_idx + 1, total_loss / len(train_loader), total_acc / len(train_loader)
            ))

    if use_ckpt:
        torch.save(model.state_dict(), "./ckpts/seq_attn_pc.pt")
    return model

def main():
    batch_size = 64
    num_epochs = 30

    tr_dataset = HanoiSeqDataset(disk_size_candidates = train_size, seq_len = train_len, num_digits = 9, dataset_size = 1000, 
                                 sample_type = "full_img_state", train = True)
    train_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)

    ts_dataset = HanoiSeqDataset(disk_size_candidates = test_size, seq_len = test_len, num_digits = 9, dataset_size = 1280, 
                                 sample_type = "full_img_state", train = True)
    test_loader = DataLoader(ts_dataset, batch_size = 32, shuffle = True)

    torch.manual_seed(0)

    device = torch.device("cuda:0")
    
    # prepare NN
    model = BindingLayer(num_outs = 6).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    # prepare PC
    pc = JL.get_hanoi_aux_pc_fix(groundtruth_weights = False)

    # pretrain
    model = pretrain(model, optimizer, device, num_epochs = 200, use_ckpt = False)

    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    pg_bar = ProgressBar(num_epochs, len(train_loader), ["aveg LL", "aveg acc"])
    for _ in range(num_epochs):
        total_ll = 0.0
        total_acc = 0.0
        total_seq = 0.0
        pg_bar.new_epoch_begin()
        for state_imgs, last_actions, actions, raw_states, aux_vars, sample_idxs in train_loader:
            b, s, h, w = state_imgs.size(0), state_imgs.size(1), state_imgs.size(2), state_imgs.size(3)
            state_imgs = state_imgs.to(device)
            last_actions = last_actions.to(device)
            actions = actions.to(device)
            aux_vars = aux_vars.float().to(device)
            
            pred_symbols = torch.sigmoid(model(state_imgs.view(b * s, h, w))).view(b, s, 3 * 2) # (b, s, 3 * 2)
            last_actions_onehot = F.one_hot(last_actions, num_classes = 6) # (b, s, 6)
            actions_onehot = F.one_hot(actions, num_classes = 6) # (b, s, 6)
            aux_in = aux_vars
            aux_in[:,0,:] = last_actions_onehot[:,0,:]
            aux_out = aux_in.clone() * 0.0 + 1/6.0
            aux_out[:,:-1,:] = aux_in[:,1:,:]

            with torch.no_grad():
                pc_inputs = torch.cat((pred_symbols, aux_in, actions_onehot, aux_out), dim = 2).view(b * s, 24).clamp(1e-6, 1.0 - 1e-6)
                pc_inputs = pc_inputs.detach().cpu().numpy()

                bpc = JL.cu_bit_circuit(pc)
                pc_lls = JL.pc_values(bpc, pc_inputs)
                pc_grads1 = JL.pc_gradients(bpc, pc_inputs, log_grad = False, eps = 1e-6)
                pc_grads2 = JL.pc_gradients(bpc, pc_inputs, log_grad = False, eps = 1e-6, marginalized_vars = [i for i in range(13, 19)])
                pc_grads3 = JL.pc_gradients(bpc, pc_inputs, log_grad = False, eps = 1e-6, marginalized_vars = [i for i in range(13, 25)])
                pc_grads_nn = torch.from_numpy(pc_grads1[:,:6] - pc_grads3[:,:6]).to(device).reshape(b, s, 6)
                pc_grads_auxin = (pc_grads1[:,6:12] - pc_grads2[:,6:12]).reshape(b, s, 6)
                pc_grads_auxin[:,0,:] = 0.0

                # update PC
                JL.full_batch_em(pc, pc_inputs, step_size = 0.01)
                JL.fix_pc_input_weights2(pc)

            optimizer.zero_grad()

            pred_symbols.backward(gradient = -pc_grads_nn)

            optimizer.step()

            # update auxiliary variables
            tr_dataset.update_aux_vars(sample_idxs, pc_grads_auxin, lr = 0.01)

            ll = float(np.mean(pc_lls))
            total_ll += ll

            with torch.no_grad():
                pred_act = JL.pc_condprobs(pc, pc_inputs, [i for i in range(13, 25)])[:,:6].reshape(b, s, 6)
                pred_act = torch.from_numpy(pred_act).to(device)
                pred_act = torch.argmax(pred_act, dim = 2)
                
                acc = float(torch.mean((pred_act == actions).float()).detach().cpu().numpy())
                total_acc += acc

            pg_bar.new_batch_done([ll, acc])
        
        pg_bar.epoch_ends([total_ll / len(train_loader), total_acc / len(train_loader)])
        #for eval_mode in ["prob", "sample", "argmax", "groundtruth"]:
        for eval_mode in ["argmax"]:
            total_seq_acc = 0.0
            acc_flag = np.zeros([test_len, 32])
            for state_imgs, last_actions, actions, raw_states, aux_vars, sample_idxs in test_loader:
                b, s, h, w = state_imgs.size(0), state_imgs.size(1), state_imgs.size(2), state_imgs.size(3)
                state_imgs = state_imgs.to(device)
                last_actions = last_actions.to(device)
                actions = actions.to(device)

                last_actions_onehot = F.one_hot(last_actions, num_classes = 6) # (b, s, 6)
                actions_onehot = F.one_hot(actions, num_classes = 6) # (b, s, 6)
                aux_in = torch.zeros([b, s, 6], device = device)
                aux_in[:,0,:] = last_actions_onehot[:,0,:]
                aux_out = torch.ones([b, s, 6], device = device)
                
                pred_symbols = torch.sigmoid(model(state_imgs.view(b * s, h, w))).view(b, s, 3 * 2) # (b, s, 3 * 2)

                with torch.no_grad():
                    pc_inputs = torch.cat((pred_symbols, aux_in, actions_onehot, aux_out), dim = 2).clamp(1e-6, 1.0 - 1e-6)
                    pc_inputs = pc_inputs.detach().cpu().numpy()

                    curr_acc = np.zeros([s])
                    
                    for seq_idx in range(s):
                        preds = JL.pc_condprobs(pc, pc_inputs[:,seq_idx,:], [i for i in range(13, 25)]) # (b, 6 * 2)
                        pred_act, pred_aux = torch.from_numpy(preds[:,:6]).to(device), preds[:,6:]
                        acc = float(torch.mean((torch.argmax(pred_act, dim = 1) == actions[:,seq_idx]).float()).detach().cpu().numpy())
                        
                        acc_flag[seq_idx,:] = (torch.argmax(pred_act, dim = 1) == actions[:,seq_idx]).float().detach().cpu().numpy()
                        
                        curr_acc[seq_idx] = acc
                        if seq_idx < s - 1:
                            if eval_mode == "prob":
                                pass
                            elif eval_mode == "sample":
                                sampled_aux = np.argmax(np.log(pred_aux) + np.random.gumbel(size = pred_aux.shape), axis = 1) 
                                pred_aux = np.eye(6)[sampled_aux]
                            elif eval_mode == "argmax":
                                best_aux = np.argmax(pred_aux, axis = 1)
                                pred_aux = np.eye(6)[best_aux]
                            elif eval_mode == "groundtruth":
                                pred_aux = last_actions_onehot[:,seq_idx+1,:].float().detach().cpu().numpy()
                            else:
                                raise NotImplementedError()
                            pc_inputs[:,seq_idx+1,6:12] = pred_aux
                    full_seq = np.mean(np.prod(acc_flag, 0))
                total_seq_acc = total_seq_acc + curr_acc
                total_seq += full_seq

            print(" - [{}] aveg test acc: |".format(eval_mode), end = "")
            for seq_idx in range(s):
                print(" {:.4f} |".format(total_seq_acc[seq_idx] / len(test_loader)), end = "")
            print("")
            print(total_seq/len(test_loader))


if __name__ == "__main__":
    main()
