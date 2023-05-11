import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from get_features import PrepareData
from model import MultiTaskMLPModel
from utils import train_multitask, evaluate_multitask

window = 48
pred_window = 24

dataset_splits = 'train', 'val', 'test'
intervention = 'vaso'

ROOT_DATA_FOLDER = '/data/data_chil/'
DATA_INPUT_PATH = f'{ROOT_DATA_FOLDER}data_rank/paramvitals/raw/intervention_{intervention}/predwindow_{pred_window}/standard_scaled/'
DATA_OUTPUT_PATH = DATA_INPUT_PATH

feature_list = ['gender_F', 'gender_M', 'age_1', 'age_2', 'age_3', 'age_4']
vital_list = ['heart rate', 'heart rate.1', 'systolic blood pressure', 'systolic blood pressure.1',
              'diastolic blood pressure', 'diastolic blood pressure.1', 'oxygen saturation', 'oxygen saturation.1',
              'respiratory rate', 'respiratory rate.1', 'glascow coma scale total', 'glascow coma scale total.1',
              'temperature', 'temperature.1']

intervention_ablation = True

net_type = 'mlp'  # type of NN to use -- currently only affects output filenames
opt = 'adam'  # optimizer to use -- allowed are adam and sgd
scale_type = 'standard'

input_dim = 30  # size of the input to embedding (linear function of input).
embed_dim1 = 10  # size of the first hidden embedding layer.
embed_dim2 = 10  # size of the second hidden embedding layer.

epochs = 50  # number of epochs
lr = 1e-5  # learning rate parameter for training
bs = 32  # batch size parameter for training
loss_wts = [1, 1]  # relative weight of ranking loss [first val] vs prediction loss [second val]

# identify the device to run on, preferring cuda and defaulting to cpu
device = None
for d in ('cuda:1', 'cuda:0', "cpu"):
    try:
        device = torch.device(d)
        break
    except:
        pass

dataproc = PrepareData(ROOT_DATA_FOLDER, feature_list, vital_list, intervention, window, pred_window)

# load data and split into train, val, test
data = dataproc.load_data()

X, X2 = defaultdict(defaultdict), defaultdict(defaultdict)
Y, Y2 = defaultdict(defaultdict), defaultdict(defaultdict)

sub = {}
for idx, s in enumerate(dataset_splits):
    for offset, struct in {0: X, 3: X['vital'], 6: Y, 9: Y['others'], 12: sub}.items():
        struct[s] = data[idx + offset]

# sliding windows to segment the data
##for each patient get vector of when intervention is required in the future: Y2['inttime']['train']
for s in dataset_splits:
    X2[s], Y2[s], Y2['curr']['vital'][s], Y2['curr']['others'][s], Y2['inttime'][s] = \
        dataproc.get_slidingwindowXY(X[s], Y[s], X['vital'][s], Y['others'][s], sub[s])

    Y2['curr']['vtraj'][s], Y2['curr']['keep'][s] = dataproc.fitline_vitals(Y2['curr']['vital'][s])
    X2['curated'][s] = dataproc.include_slope_int_X(X2[s], Y2['curr']['vtraj'][s], Y2['curr']['others'][s])

    X2['unscaled'][s] = X2['curated'][s].copy()

X2['curated']['train'], X2['curated']['val'], X2['curated']['test'] = \
    dataproc.scaledata(X2['curated']['train'], X2['curated']['val'], X2['curated']['test'], scale_type)

for s in dataset_splits:
    X2['rank'][s], Y2['rank'][s], Y2['count'][s], Y2['intv'][s] = \
        dataproc.get_pairs(X2['curated'][s], Y2['inttime'][s], Y2[s], 50000, neq=True)

    X2['rank']['bal'][s], Y2['rank']['bal'][s], Y2['count']['bal'][s], Y2['intv']['bal'][s] = \
        dataproc.bal_XY(X2['rank'][s], Y2['rank'][s], Y2['count'][s], Y2['intv'][s])

    X2['rank']['rand'][s], Y2['rank']['rand'][s], Y2['count']['rand'][s], Y2['intv']['rand'][s] = \
        dataproc.randomize(X2['rank']['bal'][s], Y2['rank']['bal'][s], Y2['count']['bal'][s], Y2['intv']['bal'][s])

os.makedirs(DATA_OUTPUT_PATH)
if not os.path.exists(DATA_OUTPUT_PATH):
    os.makedirs(DATA_OUTPUT_PATH)

for s in dataset_splits:
    pickle_save_path = os.path.join(DATA_OUTPUT_PATH, f'{s}_rank.pickle')
    with open(pickle_save_path, 'wb') as f:
        pickle.dump((X2['rank']['rand'][s],
                     X2['unscaled'][s], Y2['rank']['rand'][s],
                     Y2['count']['rand'][s], Y2['intv']['rand'][s]), f)

# load saved data
for s in dataset_splits:
    pickle_load_path = os.path.join(DATA_INPUT_PATH, f'{s}_rank.pickle')

    with open(pickle_load_path, 'rb') as f:
        X2['rank']['train'], _, Y2['rank']['train'], Y2['count']['train'], Y2['intv']['train'] = pickle.load(f)

if intervention_ablation:
    all_features = ['gender_F', 'gender_M', 'age_1', 'age_2', 'age_3', 'age_4', 'slope_heart rate.1',
                    'slope_systolic blood pressure.1', 'slope_diastolic blood pressure.1', 'slope_oxygen saturation.1',
                    'slope_respiratory rate.1', 'slope_glascow coma scale total.1',
                    'slope_temperature.1', 'heart rate.1', 'systolic blood pressure.1', 'diastolic blood pressure.1',
                    'oxygen saturation.1', 'respiratory rate.1', 'glascow coma scale total.1', 'temperature.1', 'vent',
                    'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                    'norepinephrine',
                    'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']

    select_feat_ind = []
    for f in range(len(all_features)):
        if all_features[f] != intervention:
            select_feat_ind.append(f)

    select_feat_ind = np.array(select_feat_ind)
    for s in dataset_splits:
        X2['rank'][s] = X2['rank'][s][:, select_feat_ind, :]

paths = {i: f'{i}/rank_multitask/paramvitals/raw/intervention_{intervention}/predwindow_{pred_window}/{net_type}/'
            f'{opt}/scale_{scale_type}/loss_wts{loss_wts}/{"intervention_ablation/" if intervention_ablation else ""}'
         for i in ('logs', 'output', 'model')}

feat_dim = X2['rank']['train'].shape[1]

model = MultiTaskMLPModel(feat_dim=feat_dim, inp_emb=input_dim, emb_dim1=embed_dim1, emb_dim2=embed_dim2)
model_name = f'multitaskmlp_{input_dim}embed_{embed_dim1}fc1_{embed_dim2}fc2_{epochs}epochs_{bs}bs_{lr}lr'

model = model.to(device)

opt_fn = {'adam': optim.Adam, 'sgd': optim.SGD}[opt.lower()]
optimizer = opt_fn(model.parameters(), lr)

num_0 = len(np.where(Y2['intv']['train'][:, 0] == 0)[0])
num_1 = len(np.where(Y2['intv']['train'][:, 0] == 1)[0])

num = max(num_0, num_1)

wt = [num / num_0, num / num_1]
wt = torch.FloatTensor(wt).cuda()

criterion_pred = nn.CrossEntropyLoss(weight=wt)
criterion_rank = nn.BCELoss()
writer = SummaryWriter(paths['logs'] + model_name)
best_val_loss = float("inf")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    total_rank_loss, total_pred_loss = train_multitask(device, model, optimizer, loss_wts, criterion_rank,
                                                       criterion_pred, bs, lr, epoch, epochs, X2['rank']['train'],
                                                       Y2['rank']['train'], Y2['intv']['train'])
    writer.add_scalar('Loss/train_rank', total_rank_loss, epoch)
    writer.add_scalar('Loss/train_pred', total_pred_loss, epoch)

    val_loss_rank, val_loss_pred, acc_intv, auc_intv, _, acc_rank, auc_rank, _ = evaluate_multitask(device, model,
                                                                                                    optimizer, loss_wts,
                                                                                                    criterion_rank,
                                                                                                    criterion_pred, bs,
                                                                                                    lr, epoch, epochs,
                                                                                                    X2['rank']['val'],
                                                                                                    Y2['rank']['val'],
                                                                                                    Y2['intv']['val'])

    writer.add_scalar('Loss/val_rank', val_loss_rank, epoch)
    writer.add_scalar('Loss/val_pred', val_loss_pred, epoch)
    print('-' * 95)
    print(
        '|end of epoch {:3d}| time: {:5.2f}s| valid rank loss {:5.2f} | valid pred loss {:5.2f} | valid rank auc {:5.2f} | valid pred auc {:5.2f} |'.format(
            epoch, (time.time() - epoch_start_time), val_loss_rank, val_loss_pred, auc_rank, auc_intv))
    print('-' * 95)

    if ((loss_wts[0] * val_loss_rank + loss_wts[1] * val_loss_pred) < best_val_loss):
        best_val_loss = loss_wts[0] * val_loss_rank + loss_wts[1] * val_loss_pred
        best_model = model

test_loss_rank, test_loss_pred, test_acc_intv, test_auc_intv, test_conf_intv, test_acc_rank, test_auc_rank, test_conf_rank = evaluate_multitask(
    device, model, optimizer, loss_wts, criterion_rank, criterion_pred, bs, lr, epoch, epochs, X2['rank']['test'],
    Y2['rank']['test'], Y2['intv']['test'])

val_loss_rank, val_loss_pred, val_acc_intv, val_auc_intv, val_conf_intv, val_acc_rank, val_auc_rank, val_conf_rank = evaluate_multitask(
    device, best_model, optimizer, loss_wts, criterion_rank, criterion_pred, bs, lr, epoch, epochs, X2['rank']['val'],
    Y2['rank']['val'], Y2['intv']['val'])

print('=' * 95)
print(
    '|end of training {:3d}| time: {:5.2f}s| test rank loss {:5.2f} | test pred loss {:5.2f} | test rank auc {:5.2f} | test pred auc {:5.2f} |'.format(
        epoch, (time.time() - epoch_start_time), test_loss_rank, test_loss_pred, test_auc_intv, test_auc_rank))

outfile = f'{paths["output"]}output_{model_name}.txt'

os.makedirs(paths['output'], exist_ok=True)
os.makedirs(paths['model'], exist_ok=True)

torch.save(best_model, f'{paths["model"]}{model_name}.pt')

with open(outfile, 'w') as f:
    f.write(model_name + '\n')
    f.write('Ranking' + '\n')
    f.write('Test accuracy:' + str(test_acc_rank) + '\n')
    f.write('Test AUC:' + str(test_auc_rank) + '\n')
    f.write('Conf:' + str(test_conf_rank) + '\n')

    f.write('Intv pred' + '\n')
    f.write('Test accuracy:' + str(test_acc_intv) + '\n')
    f.write('Test AUC:' + str(test_auc_intv) + '\n')
    f.write('Conf:' + str(test_conf_intv) + '\n')

    f.write('Ranking' + '\n')
    f.write('Best val accuracy: ' + str(val_acc_rank) + '\n')
    f.write('Best val AUC:' + str(val_auc_rank) + '\n')
    f.write('Best val conf:' + str(val_conf_rank) + '\n')

    f.write('Intv pred' + '\n')
    f.write('Best val accuracy:' + str(val_acc_intv) + '\n')
    f.write('Best val AUC:' + str(val_auc_intv) + '\n')
    f.write('Best val conf:' + str(val_conf_intv) + '\n')
