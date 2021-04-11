import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imblearn.combine import SMOTETomek

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])


def load_data(batch_size = 128):

    df_train = pd.read_csv("input/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("input/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(int)
    X = np.array(df_train[list(range(187))].values)

    Y_test = np.array(df_test[187].values)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
	
	#Smote for data augmentation
	sm = SMOTETomek()
	X_sm, y_sm = sm.fit_resample(X,y)  
    X_sm = X_sm[..., np.newaxis]

    train_dataset = CustomDataset(X_sm, Y_sm)
    val_dataset = CustomDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

