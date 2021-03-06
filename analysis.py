import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

def get_embeddings(model, loader, device):
    model.eval()
    model.fc2 = nn.Sequential()
    x_pred = []
    x_true = []
    i = 0
    for x, y in loader:
        x = x.to(device).reshape(-1, 1, 187).float()
        x_hat = model(x)
        x_pred = x_pred + x_hat.detach().to('cpu').numpy().tolist()
        x_true.extend(y.long().detach().to('cpu').numpy())
    return np.array(x_pred), np.array(x_true)

def get_tsne(embeds):
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeds)
    return tsne_proj

def plot_tsne(coordinates, labels, num_categories = 5):
    cmap = cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,8))
    for lab in range(num_categories):
        indices = labels==lab
        ax.scatter(coordinates[indices,0],coordinates[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
