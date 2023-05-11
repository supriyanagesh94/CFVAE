import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CFVAE
from utils import train_vae, evaluate_vae

"""
Script to load an MLP model trained to predict if intervention is required in the next 24 hours and train a VAE to generate counterfactual examples

Variables:
    intervention: 'vaso','vent' (intervention that we are predicting)
    loss_wt: [VAE loss, CF loss] weighting the overall loss function of the system 
    pt_modelname: pretrained intervention prediction model name
Hyperparameters:
    emb_dim1: Size of linear layers in the VAE  

    lr: learning rate
    bs: batch size
    epochs 

    Note: mlp_inpemb,f_dim1, f_dim2 values should be equal to a pre-trained intervention prediction model hyperparameters
"""

emb_dim1 = 40  # size of the outer hidden layer of the VAE
lr = 1e-3  # learning rate parameter for training
bs = 32  # batch size parameter for training
opt = 'Adam'  # optimizer used
epochs = 100  # number of epochs
loss_wt = [1, 100]  # weight of VAE loss (0) and weight of counterfactual loss (1)
mlp_inpemb = 30  # dimension of the word embedding.
f_dim1 = 10  # hidden units in first layer of MLP
f_dim2 = 10  # hidden units in second layer of MLP

dataset_splits = 'train', 'val', 'test'
intervention = 'vaso'

# identify the device to run on, preferring cuda and defaulting to cpu
device = None
for d in ('cuda:1', 'cuda:0', "cpu"):
    try:
        device = torch.device(d)
        break
    except:
        pass

model_name = f'vae_epochs_{epochs}embed_{emb_dim1}lr_{lr}losswt_{loss_wt}'
pt_modelname = f'multitaskmlp_{mlp_inpemb}embed_{f_dim1}fc1_{f_dim2}fc2_epochs_32bs.lr.pt'

# Loading the data saved while predicting intervention
data_path = f'/data/data_chil/data_pred/paramvitals/raw/intervention_{intervention}/standard_scaled/'
pickle_paths = [os.path.join(data_path, i + '.pickle') for i in dataset_splits]

X2 = {}
Y2 = {}

for s in dataset_splits:
    pickle_path = os.path.join(data_path, f'{s}.pickle')

    # try loading the pickle file
    # if this fails, load directly from CSV and create the pickle file for future runs
    try:
        with open(pickle_path, 'rb') as f:
            X2[s], _, _, _, Y2[s] = pickle.load(f)
    except:
        X2[s] = pd.read_csv(os.path.join(data_path, f'X2_curated_{s}.csv'), index_col=0)
        Y2[s] = np.array(pd.read_csv(os.path.join(data_path, f'Y2_{s}.csv'), index_col=0))

        np.reshape(X2[s], (X2[s].shape[0], X2[s].shape[1], 1))
        np.reshape(Y2[s], (Y2[s].shape[0], 1))

        with open(pickle_path, 'wb') as f:
            pickle.dump((X2[s], None, None, None, Y2[s]), f)

feat_dim = X2['train'].shape[1]

num_1 = len(np.where(Y2['train'][:, 0] == 1)[0])
num_0 = len(np.where(Y2['train'][:, 0] == 0)[0])
num = max(num_0, num_1)

criterion_cf = nn.CrossEntropyLoss()

model = CFVAE(feat_dim, emb_dim1, 9, 9, 9, mlp_inpemb, f_dim1, f_dim2)

opt_fn = {'adam': optim.Adam, 'sgd': optim.SGD}[opt.lower()]
optimizer = opt_fn(model.parameters(), lr)

criterion = nn.MSELoss()
criterion_x = nn.L1Loss()

paths = {i: os.path.join(i, f'VAE_CF/multitask_rank/intervention_{intervention}/{opt}/')
         for i in ('logs', 'output', 'model')}

writer = SummaryWriter(paths['logs'] + model_name)

best_val_loss = float("inf")
best_val_cf_auc = 0

bb_modelpath = f'model/rank_multitask/paramvitals/raw/intervention_{intervention}/mlp/adam/scale_standard/{pt_modelname}'

bb_model = torch.load(bb_modelpath)
bb_model.eval()
bb_model.to(device)

# setting the weights of the intervention prediction MLP to be equal to the pre-trained weights
model_dict = model.state_dict()
bb_dict = bb_model.state_dict()

params1 = bb_model.named_parameters()
params2 = model.named_parameters()

dict_params2 = dict(params2)

for name1, param1 in params1:  #
    if name1 in dict_params2:
        dict_params2[name1].data.copy_(param1.data)

model.load_state_dict(dict_params2)

# freeze weights by setting required grad to False
mlp_names = list(bb_dict.keys())
for name, param in model.named_parameters():
    if name in mlp_names:
        param.requires_grad = False

model.to(device)

epoch_start_time = 0
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    tot_log_loss, train_loss_vae, train_loss_cf = train_vae(device, model, optimizer, criterion, criterion_cf,
                                                            criterion_x, loss_wt, bs, lr, epoch, epochs,
                                                            X2['train'], Y2['train'])
    writer.add_scalar('Loss/train_vae', train_loss_vae, epoch)
    writer.add_scalar('Loss/train_cf', train_loss_cf, epoch)
    val_loss, val_loss_vae, val_loss_cf, acc_cf, auc_cf, _ = evaluate_vae(device, model, optimizer, criterion,
                                                                          criterion_cf, criterion_x, loss_wt, bs, lr,
                                                                          epoch, epochs, X2['val'], Y2['val'])
    writer.add_scalar('Loss/val_vae', val_loss_vae, epoch)
    writer.add_scalar('Loss/val_cf', val_loss_cf, epoch)
    print('-' * 95)
    print('|end of epoch {:3d}| time: {:5.2f}s| valid loss {:5.2f} | valid acc {:5.2f} | valid auc {:5.2f} | '.format(
        epoch, (time.time() - epoch_start_time), val_loss, acc_cf, auc_cf))
    print('-' * 95)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

test_loss, _, _, acc_cf, auc_cf, conf_cf = evaluate_vae(device, best_model, optimizer, criterion, criterion_cf,
                                                        criterion_x, loss_wt, bs, lr, epochs, epochs, X2['test'],
                                                        Y2['test'])
val_loss, _, _, val_acc_cf, val_auc_cf, val_conf_cf = evaluate_vae(device, best_model, optimizer, criterion,
                                                                   criterion_cf, criterion_x, loss_wt, bs, lr, epochs,
                                                                   epochs, X2['val'], Y2['val'])
print('=' * 95)
print(
    '|end of training {:3d}| time: {:5.2f}s| test loss {:5.2f} | test acc {:5.2f} | test auc {:5.2f} | '.format(epochs,
                                                                                                                time.time() - epoch_start_time,
                                                                                                                test_loss,
                                                                                                                acc_cf,
                                                                                                                auc_cf))

outfile = paths['output'] + 'output_' + model_name + '.txt'

if not os.path.exists(paths['output']):
    os.makedirs(paths['output'])

if not os.path.exists(paths['model']):
    os.makedirs(paths['model'])

torch.save(best_model, paths['model'] + model_name + '.pt')

with open(outfile, 'w') as f:
    f.write(model_name + '\n')
    f.write('Test accuracy cf:' + str(acc_cf) + '\n')
    f.write('Test AUC cf:' + str(auc_cf) + '\n')
    f.write('Conf cf:' + str(conf_cf) + '\n')

    f.write('Best val accuracy CF: ' + str(val_acc_cf) + '\n')
    f.write('Best val AUC CF:' + str(val_auc_cf) + '\n')
    f.write('Best val conf CF:' + str(val_conf_cf) + '\n')
