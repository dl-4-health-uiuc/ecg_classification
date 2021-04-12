import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import models
import process_data
import analysis


#setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()

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
        f = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        cm = confusion_matrix(y_true, y_pred)
        return f, acc, cm
    
def train(model, train_loader, val_loader, n_epochs, criterion, optimizer):
    model.train()
    for epoch in range(n_epochs):
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
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        f, acc, cm = eval(model, val_loader, criterion)
        print('Epoch: %d \t Validation f: %.4f, acc: %.4f'%(epoch+1, f, acc))
#         print(cm)
