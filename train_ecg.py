import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix
import models
import process_data
import analysis


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()


def accuracy_per_class(mcm, clsnum):
    return (mcm[clsnum][0][0] + mcm[clsnum][1][1]) / (mcm[clsnum][0][0] + mcm[clsnum][0][1] + mcm[clsnum][1][0] + mcm[clsnum][1][1])

def sensitivity_per_class(mcm, clsnum):
    return mcm[clsnum][0][0] / (mcm[clsnum][0][0] + mcm[clsnum][1][0])

def specificity_per_class(mcm, clsnum):
    return mcm[clsnum][1][1] / (mcm[clsnum][1][1] + mcm[clsnum][0][1])

def pos_pred_val_per_class(mcm, clsnum):
    return mcm[clsnum][0][0] / (mcm[clsnum][0][0] + mcm[clsnum][0][1])

def neg_pred_val_per_class(mcm, clsnum):
    return mcm[clsnum][1][1] / (mcm[clsnum][1][1] + mcm[clsnum][1][0])

def stat_per_class(mcm):
    out = []
    for clsnum in range(5):
         out.append((accuracy_per_class(mcm, clsnum), sensitivity_per_class(mcm, clsnum), specificity_per_class(mcm, clsnum), \
            pos_pred_val_per_class(mcm, clsnum), neg_pred_val_per_class(mcm, clsnum)))
    return out

def avg_stats(mcm):
    avg_acc = 0
    avg_sens = 0
    avg_spec = 0
    avg_pos_pred = 0
    avg_neg_pred = 0
    for clsnum in range(5):
        avg_acc += accuracy_per_class(mcm, clsnum)/5
        avg_sens += sensitivity_per_class(mcm, clsnum)/5
        avg_spec += specificity_per_class(mcm, clsnum)/5
        avg_pos_pred += pos_pred_val_per_class(mcm, clsnum)/5
        avg_neg_pred += neg_pred_val_per_class(mcm, clsnum)/5
    return avg_acc, avg_sens, avg_spec, avg_pos_pred, avg_neg_pred

def eval(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        y_pred = []
        y_true = []
        for x, y in val_loader:
            x=x.to(device).float()
            y = y.to(device).long()
            y_hat = model(x)
            val_loss += criterion(y_hat, y).item()
            y_hat = F.softmax(y_hat, dim=1)
            y_hat = torch.argmax(y_hat, dim=1)
            y_pred.extend(y_hat.cpu().numpy())
            y_true.extend(y.long().cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        f = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        cm = confusion_matrix(y_true, y_pred)
        mcm = multilabel_confusion_matrix(y_pred, y_true) # input is inverted to get expected output
    return f, acc, cm, mcm, stat_per_class(mcm), avg_stats(mcm)


def train(model, train_loader, val_loader, n_epochs, criterion, optimizer):
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        i=0
        for x, y in train_loader:
            x=x.to(device).float()
            y = y.to(device).long()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        f, acc, cm, mcm, stats, avgstats = eval(model, val_loader, criterion)
        print('Epoch: %d \t Validation f: %.2f, acc: %.2f'%(epoch+1, f, acc))
#         print('Confusion matrix')
#         print(cm)
#         print(mcm)
#         print('Accuracy, Sensitivity, Specificity, Positive Pred Value, Negative Pred Value')
#         print('class 1 :', stats[0])
#         print('class 2 :', stats[1])
#         print('class 3 :', stats[2])
#         print('class 4 :', stats[3])
#         print('class 5 :', stats[4])
        print('avg metrics :', avgstats)

    
def run_ecg(model_name, somte=False,batch_size=200, learning_rate=0.0005, num_epochs=25, save_model=False, save_path=None):
    
    train_loader, val_loader = process_data.load_data(batch_size=batch_size, smote=smote)
    model = models.get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, val_loader, num_epochs, criterion, optimizer)
    if save_model:
        torch.save(model.state_dict(), save_path)
    return model