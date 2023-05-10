import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def get_batch(X, Y, i, bs):
    batch_len = min(bs, X.shape[0] - 1 - i)
    data = X[i:i + batch_len, :]
    target = Y[i:i + batch_len]
    return data, target


def get_batch_ts(X, Y, i, bs):
    batch_len = min(bs, X.shape[0] - 1 - i)
    data = X[i:i + batch_len, :, :]
    target = Y[i:i + batch_len]
    return data, target


def train(device, model, optimizer, criterion, batch_size, learning_rate, epoch, epochs, X_TRAIN, Y_TRAIN):
    """
    - method to train MLP for intervention prediction 
    - this function iterates over all batches (one epoch)
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    num_correct = 0
    num_total = 0
    for batch, i in enumerate(range(0, X_TRAIN.shape[0] - 1, batch_size)):
        data, target = get_batch(X_TRAIN, Y_TRAIN, i, batch_size)
        data = torch.from_numpy(data)
        data = data.to(device)
        target = torch.from_numpy(target)
        target = target.to(device)
        data = data.float()
        target = target.long()
        optimizer.zero_grad()
        out_pred = model(data)  # output from the model of size (batch_size,2)
        m = nn.Softmax(dim=1)  # softmax across the classes
        out_prob = m(out_pred)  # applying softmax activation for probability of each class
        out_loss = out_pred.to(device)

        loss = criterion(out_loss, target)  # cross entropy loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200

        _, predicted = torch.max(out_prob, 1)  # predicted class is the one with maximum softmax probability
        num_correct += (predicted == target).sum().item()
        num_total += target.shape[0]

        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            acc = num_correct / num_total
            logging.info(
                '| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | acc{:5.2f}'.format(
                    epoch, epochs, batch, X_TRAIN.shape[0] // batch_size, learning_rate, elapsed * 1000 / log_interval,
                    curr_loss, acc))
            total_loss = 0
            num_correct = 0
            num_total = 0
            start_time = time.time()

    return total_loss


def train_multitask(device, model, optimizer, loss_wts, criterion_rank, criterion_pred, batch_size, learning_rate,
                    epoch, epochs, X_rank_train, Y_rank_train, Y_intv_train):
    model.train()
    total_rank_loss = 0
    total_pred_loss = 0
    total_loss = 0
    start_time = time.time()
    num_correct = 0
    num_total = 0
    for batch, i in enumerate(range(0, X_rank_train.shape[0] - 1, batch_size)):
        data, target_intv = get_batch_ts(X_rank_train, Y_intv_train, i, batch_size)
        _, target_rank = get_batch(X_rank_train, Y_rank_train, i, batch_size)
        data = torch.from_numpy(data)
        data = data.to(device)
        target_intv = torch.from_numpy(target_intv)
        target_intv = target_intv.to(device)
        data = data.float()
        target_intv = target_intv.long()
        target_rank = torch.from_numpy(target_rank)
        target_rank = target_rank.to(device)
        target_rank = target_rank.float()
        optimizer.zero_grad()
        rank_score, x1_pred, x2_pred = model(data[:, :, 0],
                                             data[:, :, 1])  # output from the model of size (batch_size,2)
        rank_score = rank_score[:, 0]
        m = nn.Softmax(dim=1)  # softmax across the classes
        x1_prob = m(x1_pred)  # applying softmax activation for probability of each class
        x2_prob = m(x2_pred)

        loss_pred1 = criterion_pred(x1_pred, target_intv[:, 0])  # cross entropy loss
        loss_pred2 = criterion_pred(x2_pred, target_intv[:, 1])
        loss_rank = criterion_rank(rank_score, target_rank)

        loss = 0.5 * loss_wts[1] * loss_pred1 + 0.5 * loss_wts[1] * loss_pred2 + loss_wts[0] * loss_rank
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_rank_loss += loss_rank.item()
        total_pred_loss += (loss_pred1.item() + loss_pred2.item())
        total_loss += loss.item()
        log_interval = 200

        _, predicted = torch.max(x1_prob, 1)  # predicted class is the one with maximum softmax probability
        num_correct += (predicted == target_intv[:, 0]).sum().item()
        num_total += target_intv.shape[0]

        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            acc = num_correct / num_total
            logging.info(
                '| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | acc{:5.2f}'.format(
                    epoch, epochs, batch, X_rank_train.shape[0] // batch_size, learning_rate,
                    elapsed * 1000 / log_interval, curr_loss, acc))
            total_loss = 0
            total_rank_loss = 0
            total_pred_loss = 0
            num_correct = 0
            num_total = 0
            start_time = time.time()

    return total_rank_loss, total_pred_loss


def train_ts(device, model, optimizer, criterion, batch_size, learning_rate, epoch, epochs, X_TRAIN, Y_TRAIN,
             batch_first, net_type):
    """
    - method to train TransformerModel for intervention prediction 
    - this function iterates over all batches (one epoch)
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    num_correct = 0
    num_total = 0
    for batch, i in enumerate(range(0, X_TRAIN.shape[0] - 1, batch_size)):
        data, target = get_batch_ts(X_TRAIN, Y_TRAIN, i, batch_size)
        if (not batch_first):
            data = np.reshape(data, (data.shape[1], data.shape[0], data.shape[2]))  # t,bs,f

        data = torch.from_numpy(data)
        data = data.to(device)
        target = torch.from_numpy(target)
        target = target.to(device)
        data = data.float()
        target = target.long()
        optimizer.zero_grad()
        delta_T = np.zeros((data.shape[0], data.shape[1], 1))
        if (net_type == 'transformer'):
            out_pred = model(data, delta_T, 'pos_sin')  # output from the model of size (batch_size,2)
        elif (net_type == 'lstm'):
            out_pred = model(data)

        m = nn.Softmax(dim=1)  # softmax across the classes
        out_prob = m(out_pred)  # applying softmax activation for probability of each class
        out_loss = out_pred.to(device)

        loss = criterion(out_loss, target)  # cross entropy loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200

        _, predicted = torch.max(out_prob, 1)  # predicted class is the one with maximum softmax probability
        num_correct += (predicted == target).sum().item()
        num_total += target.shape[0]

        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            acc = num_correct / num_total
            logging.info(
                '| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | acc{:5.2f}'.format(
                    epoch, epochs, batch, X_TRAIN.shape[0] // batch_size, learning_rate, elapsed * 1000 / log_interval,
                    curr_loss, acc))
            total_loss = 0
            num_correct = 0
            num_total = 0
            start_time = time.time()

        return total_loss


def evaluate(device, eval_model, optimizer, criterion, bs, lr, epoch, epochs, X, Y):
    """
    - method to evaluate a trained model for intervention prediction
    - this function runs over all batches (one epoch)
    - computes the accuracy, auc and confusion matrix 
    """
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, X.shape[0] - 1, bs)):
            data, target = get_batch(X, Y, i, bs)
            data = torch.from_numpy(data)
            target = torch.from_numpy(target)
            data = data.to(device)
            target = target.to(device)
            data = data.float()
            target = target.long()
            out_pred = eval_model(data)
            m = nn.Softmax(dim=1)
            out_prob = m(out_pred)

            out_loss = out_pred.to(device)

            total_loss += data.shape[0] * criterion(out_loss, target).item()

            _, predicted = torch.max(out_prob, 1)
            # predicted = torch.round(out_pred)
            pred_prob = out_pred[:, 1]  # prob of positive class

            if (batch == 0):
                all_targets = target
                all_pred = predicted
                all_prob = pred_prob
            else:
                all_targets = torch.cat((all_targets, target), 0)
                all_pred = torch.cat((all_pred, predicted), 0)
                all_prob = torch.cat((all_prob, pred_prob), 0)

            np_targets = all_targets.cpu().numpy()
            np_prob = all_prob.cpu().numpy()
            np_class = all_pred.cpu().numpy()
            auc = roc_auc_score(np_targets, np_prob)
            acc = accuracy_score(np_targets, np_class)
            conf = confusion_matrix(np_targets, np_class)

    return (total_loss / (X.shape[0] - 1), acc, auc, conf)


def evaluate_multitask(device, eval_model, optimizer, loss_wts, criterion_rank, criterion_pred, bs, learning_rate,
                       epoch, epochs, X, Y_rank, Y_intv):
    eval_model.eval()
    total_loss_rank = 0
    total_loss_pred = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, X.shape[0] - 1, bs)):
            data, target_intv = get_batch_ts(X, Y_intv, i, bs)
            _, target_rank = get_batch(X, Y_rank, i, bs)
            data = torch.from_numpy(data)
            target_intv = torch.from_numpy(target_intv)
            data = data.to(device)
            target_intv = target_intv.to(device)
            data = data.float()
            target_intv = target_intv.long()
            target_rank = torch.from_numpy(target_rank)
            target_rank = target_rank.to(device)
            target_rank = target_rank.float()
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model_encoder = model_encoder.to(device)

            rank_score, x1_pred, x2_pred = eval_model(data[:, :, 0], data[:, :, 1])
            rank_score = rank_score[:, 0]
            m = nn.Softmax(dim=1)
            x1_prob = m(x1_pred)
            x2_prob = m(x2_pred)

            loss_rank = criterion_rank(rank_score, target_rank).item()
            loss_pred1 = criterion_pred(x1_pred, target_intv[:, 0]).item()
            loss_pred2 = criterion_pred(x2_pred, target_intv[:, 1]).item()
            # total_loss += data.shape[0]*(loss_rank+loss_pred)
            total_loss_rank += loss_rank
            total_loss_pred += (loss_pred1 + loss_pred2)

            _, predicted1 = torch.max(x1_prob, 1)
            pred_prob1 = x1_prob[:, 1]  # prob of positive class

            _, predicted2 = torch.max(x2_prob, 1)
            pred_prob2 = x2_prob[:, 1]

            rank_pred = torch.zeros(rank_score.shape).to(device)
            rank_pred[rank_score >= 0.5] = 1
            rank_pred[rank_score < 0.5] = 0

            if (batch == 0):
                all_pred_targets = target_intv[:, 0]
                all_pred = predicted1
                all_prob = pred_prob1

                all_pred_targets = torch.cat((all_pred_targets, target_intv[:, 1]), 0)
                all_pred = torch.cat((all_pred, predicted2), 0)
                all_prob = torch.cat((all_prob, pred_prob2), 0)

                all_rank_pred = rank_pred
                all_rank_prob = rank_score
                all_rank_targets = target_rank
            else:
                all_pred_targets = torch.cat((all_pred_targets, target_intv[:, 0]), 0)
                all_pred = torch.cat((all_pred, predicted1), 0)
                all_prob = torch.cat((all_prob, pred_prob1), 0)

                all_pred_targets = torch.cat((all_pred_targets, target_intv[:, 1]), 0)
                all_pred = torch.cat((all_pred, predicted2), 0)
                all_prob = torch.cat((all_prob, pred_prob2), 0)

                all_rank_pred = torch.cat((all_rank_pred, rank_pred), 0)
                all_rank_prob = torch.cat((all_rank_prob, rank_score), 0)
                all_rank_targets = torch.cat((all_rank_targets, target_rank), 0)

        np_intv_targets = all_pred_targets.cpu().numpy()
        np_intv_prob = all_prob.cpu().numpy()
        np_intv_class = all_pred.cpu().numpy()
        auc_intv = roc_auc_score(np_intv_targets, np_intv_prob)
        acc_intv = accuracy_score(np_intv_targets, np_intv_class)
        conf_intv = confusion_matrix(np_intv_targets, np_intv_class)

        np_rank_targets = all_rank_targets.cpu().numpy()
        np_rank_prob = all_rank_prob.cpu().numpy()
        np_rank_class = all_rank_pred.cpu().numpy()
        auc_rank = roc_auc_score(np_rank_targets, np_rank_prob)
        acc_rank = accuracy_score(np_rank_targets, np_rank_class)
        conf_rank = confusion_matrix(np_rank_targets, np_rank_class)
    return total_loss_rank, total_loss_pred, acc_intv, auc_intv, conf_intv, acc_rank, auc_rank, conf_rank


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae(device, model, optimizer, criterion, criterion_cf, criterion_x, loss_wt, batch_size, learning_rate, epoch,
              epochs, X_TRAIN, Y_TRAIN):
    """
    - method to train VAE to generate counterfactual reconstructions 
    - total loss = w_1*VAE loss + w_2*Counterfactual loss (Cross entropy)
    - loss_wt = [w_1,w_2]
    - this function iterates over all batches (one epoch)
    """

    model.train()
    total_loss = 0
    total_loss_cf = 0
    total_loss_vae = 0
    start_time = time.time()
    num_correct = 0
    num_total = 0
    for batch, i in enumerate(range(0, X_TRAIN.shape[0] - 1, batch_size)):
        data, target = get_batch(X_TRAIN[:, :, 0], Y_TRAIN[:, 0], i, batch_size)

        target_cf = np.zeros(target.shape)
        for tt in range(len(target)):
            target_cf[tt] = 1 - target[tt]

        data = torch.from_numpy(data)
        data = data.to(device)
        data = data.float()
        target = torch.from_numpy(target)
        target = target.to(device)
        target = target.long()
        target_cf = torch.from_numpy(target_cf)
        target_cf = target_cf.to(device)
        target_cf = target_cf.long()

        optimizer.zero_grad()
        reconstruction, mu, logvar, pred_s1 = model(
            data)  # pred_s1 = MLP model prediction for the generated reconstruction

        bce_loss = criterion(reconstruction, data)
        cf_loss = criterion_cf(pred_s1, target_cf)
        # cf_loss = criterion_cf(pred_s1,target)
        vae_loss = final_loss(bce_loss, mu, logvar)

        total_loss_vae += vae_loss.item()
        total_loss_cf += cf_loss.item()

        loss = loss_wt[0] * vae_loss + loss_wt[1] * cf_loss
        # loss = loss_wt[0]*loss - loss_wt[1]*cf_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 1000

        """
        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss/log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | '.format(epoch,epochs,batch,X_TRAIN.shape[0]//batch_size,learning_rate,elapsed*1000/log_interval,curr_loss))
            total_loss = 0
            total_loss_vae = 0
            total_loss_cf = 0
            num_correct = 0
            num_total = 0
            start_time = time.time()
        """

    return total_loss, total_loss_vae, total_loss_cf


def train_vanilla_vae(device, model_encoder, model_decoder, optimizer, criterion, batch_size, learning_rate, epoch,
                      epochs, X_TRAIN, Y_TRAIN):
    model_encoder.train()
    model_decoder.train()
    total_loss = 0
    start_time = time.time()
    num_correct = 0
    num_total = 0
    for batch, i in enumerate(range(0, X_TRAIN.shape[0] - 1, batch_size)):
        data, target = get_batch(X_TRAIN, Y_TRAIN, i, batch_size)

        data = torch.from_numpy(data)
        data = data.to(device)
        data = data.float()
        target = torch.from_numpy(target)
        target = target.to(device)
        target = target.long()

        optimizer.zero_grad()
        mu, logvar, z = model_encoder(data)  # pred_s1 = MLP model prediction for the generated reconstruction
        reconstruction = model_decoder(z)

        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model_encoder.parameters()) + list(model_decoder.parameters()), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 1000

        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            logging.info(
                '| epoch {:3d} /{:3d}| {:5d}/{:5d} batches | lr {:02.2f}]ms/batch {:5.2f} | loss {:5.2f} | '.format(
                    epoch, epochs, batch, X_TRAIN.shape[0] // batch_size, learning_rate, elapsed * 1000 / log_interval,
                    curr_loss))
            total_loss = 0
            start_time = time.time()
        return total_loss


def evaluate_vae(device, eval_model, optimizer, criterion, criterion_cf, criterion_x, loss_wt, batch_size,
                 learning_rate, epoch, epochs, X, Y):
    """
    - method to evaluate a trained LinearVAE model 
    - total_loss = w_1*VAE_loss + w_2*Counterfactual loss  
    - loss_wt = [w_1,w_2]
    - computes the accuracy, auc and confusion matrix for the counterfactuals generated
    """
    eval_model.eval()
    total_loss = 0
    total_loss_vae = 0
    total_loss_cf = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, X.shape[0] - 1, batch_size)):
            data, target = get_batch(X[:, :, 0], Y[:, 0], i, batch_size)

            target_cf = np.zeros(target.shape)
            for tt in range(len(target)):
                target_cf[tt] = 1 - target[tt]

            data = torch.from_numpy(data)
            target = torch.from_numpy(target)
            data = data.to(device)
            target = target.to(device)
            data = data.float()
            target = target.long()
            target_cf = torch.from_numpy(target_cf)
            target_cf = target_cf.to(device)
            target_cf = target_cf.long()

            reconstruction, mu, logvar, pred_s1 = eval_model(data)

            bce_loss = criterion(reconstruction, data)
            cf_loss = criterion_cf(pred_s1, target_cf)
            # cf_loss = criterion_cf(pred_s1,target)
            vae_loss = final_loss(bce_loss, mu, logvar)
            # x_loss = criterion_x(reconstruction,data)

            total_loss_vae += vae_loss.item()
            total_loss_cf += cf_loss.item()

            loss = loss_wt[0] * vae_loss + loss_wt[1] * cf_loss
            # loss = loss_wt[0]*loss - loss_wt[1]*cf_loss
            total_loss += loss.item()

            m = nn.Softmax(1)
            prob_cf = m(pred_s1)
            _, predicted_cf = torch.max(prob_cf, 1)
            pred_prob_cf = prob_cf[:, 1]  # prob of positive class

            if (batch == 0):
                all_targets_cf = target_cf
                all_pred_cf = predicted_cf
                all_prob_cf = pred_prob_cf
            else:
                all_targets_cf = torch.cat((all_targets_cf, target_cf), 0)
                all_pred_cf = torch.cat((all_pred_cf, predicted_cf), 0)
                all_prob_cf = torch.cat((all_prob_cf, pred_prob_cf), 0)

        np_targets_cf = all_targets_cf.cpu().numpy()
        np_prob_cf = all_prob_cf.cpu().numpy()
        np_class_cf = all_pred_cf.cpu().numpy()
        auc_cf = roc_auc_score(np_targets_cf, np_prob_cf)
        acc_cf = accuracy_score(np_targets_cf, np_class_cf)
        conf_cf = confusion_matrix(np_targets_cf, np_class_cf)

    return (total_loss, total_loss_vae, total_loss_cf, acc_cf, auc_cf, conf_cf)


def evaluate_vanilla_vae(device, eval_model_encoder, eval_model_decoder, optimizer, criterion, batch_size,
                         learning_rate, epoch, epochs, X, Y):
    """
    - method to evaluate a trained LinearVAE model 
    - total_loss = w_1*VAE_loss + w_2*Counterfactual loss  
    - loss_wt = [w_1,w_2]
    - computes the accuracy, auc and confusion matrix for the counterfactuals generated
    """
    eval_model_encoder.eval()
    eval_model_decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, X.shape[0] - 1, batch_size)):
            data, target = get_batch(X, Y, i, batch_size)

            data = torch.from_numpy(data)
            target = torch.from_numpy(target)
            data = data.to(device)
            target = target.to(device)
            data = data.float()
            target = target.long()

            mu, logvar, z = eval_model_encoder(data)
            reconstruction = eval_model_decoder(z)

            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)

            total_loss += loss.item()

    return total_loss
