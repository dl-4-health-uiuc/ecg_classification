import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import models
import process_data
import analysis


#setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()

# not used now
def transfer_model(pretrained):
    if freeze_params:
        for param in pretrained.parameters():
            param.requires_grad = False
    pretrained.fc1 = nn.Linear(64,32)
    pretrained.fc2 = nn.Linear(32, 1)
    if device_count > 1:
        pretrained = nn.DataParallel(pretrained)
    pretrained = pretrained.to(device)
    return pretrained

## evaluate model with validation dataset
def eval(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        y_pred = []
        y_true = []
        for x, y in val_loader:
            x=x.to(device).reshape(-1, 1, 187).float()
            y = y.to(device).long()
            y_hat = model(x)
            y_hat = torch.sigmoid(y_hat).reshape(-1)
            y = y.float()
            val_loss += criterion(y_hat, y).item()
            
            y_hat = y_hat > 0.5
            y_pred.extend(y_hat.cpu().numpy())
            y_true.extend(y.long().cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        f = f1_score(y_pred=y_pred, y_true=y_true)
        p = precision_score(y_pred=y_pred, y_true=y_true)
        r = recall_score(y_pred=y_pred, y_true=y_true)
        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        cm = confusion_matrix(y_true, y_pred)
        return p,r,f, acc, cm

## train the model with the configs given
def train(model, train_loader, val_loader, n_epochs, criterion, optimizer):
    max_f1 = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        i=0
        for x, y in train_loader:
            x=x.to(device).reshape(-1, 1, 187).float()
            y = y.to(device).long()
            y_hat = model(x)
            y_hat = torch.sigmoid(y_hat).reshape(-1)
            y = y.float()
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        p,r,f, acc, cm = eval(model, val_loader, criterion)
        if epoch > 10 and f > max_f1:
            max_f1=f
            print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
            print('Epoch: %d \t Validation acc: %.4f'%(epoch+1, acc))
            print('P,r,f')
            print(p,r,f)
            print(cm)

## function to train the data and print the results based on the configurations passed.
## the transfer path should have the path to the model state dict of the pretrained model (if empty the model will run from scratch)
## the saved_loader string is just the suffix after loader.
## a save_path for the model state dict can also be given, so that it can be used as a pretrained model in mi prediction task.       
def run_mi(model_name, smote=True,batch_size=256, learning_rate=0.001, num_epochs=25, transfer='', saved_loader='', save_path=None):
    if smote and saved_loader:
        train_loader = torch.load("train_loader"+saved_loader)
        val_loader = torch.load("val_loader"+saved_loader)
    else:
        train_loader, val_loader = process_data.load_data_mi(batch_size=batch_size, smote=smote)
    model = models.get_model(model_name, mi=True, transfer=transfer)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, val_loader, num_epochs, criterion, optimizer)
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model