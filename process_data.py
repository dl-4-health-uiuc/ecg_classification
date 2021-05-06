import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])


def load_data(batch_size = 128, smote=False, num_samples=-1):

    df_train = pd.read_csv("input/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("input/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(int)
    X = np.array(df_train[list(range(187))].values)

    Y_test = np.array(df_test[187].values).astype(int)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    #Smote for data augmentation
    if smote:
        sm = SMOTETomek()
        X, Y = sm.fit_resample(X,Y)  
        X = X[..., np.newaxis]

    train_dataset = CustomDataset(X, Y)
    val_dataset = CustomDataset(X_test, Y_test)
    if num_samples > 0:
        train_dataset = train_dataset[:num_samples]
        val_dataset = val_dataset[:num_samples]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def load_data_mi(batch_size = 128, smote=False):

    df_mi1 = pd.read_csv("input/ptbdb_abnormal.csv", header=None)
    df_mi2 = pd.read_csv("input/ptbdb_normal.csv", header=None)
    df_mi = pd.concat([df_mi1,df_mi2], ignore_index=True)
#     df_mi = df_mi.sample(frac=1, random_state=1)
#     train_ratio = 0.8
#     train_index = round(0.8*len(df_mi))
    df_train, df_test = train_test_split(df_mi, test_size=0.2, random_state=1, stratify=df_mi[187])

#     df_train = df_mi.iloc[:train_index]
#     df_test = df_mi.iloc[train_index:]

    Y = np.array(df_train[187].values).astype(int)
    X = np.array(df_train[list(range(187))].values)

    Y_test = np.array(df_test[187].values).astype(int)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    if smote:
        sm = SMOTETomek()
        X, Y = sm.fit_resample(X,Y)  
        X = X[..., np.newaxis]

    train_dataset = CustomDataset(X, Y)
    val_dataset = CustomDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

