import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()
    
## The model from the paper ECG Heartbeat Classification: A Deep Transferable Representation
class CNResidual(nn.Module):
    def __init__(self, size):
        super().__init__()
        # layers
        self.c1 = nn.Conv1d(size, size, kernel_size=5, padding=2)
        self.c2 = nn.Conv1d(size, size, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)
        
    def forward(self, x):
        # call layers
        x1 = F.relu(self.c1(x))
        x1 = self.c2(x1)
        return self.pool(F.relu(x+x1))
    
class CNet(nn.Module):

    def __init__(self, num_resids=5):
        super().__init__()
        # layers
        self.c1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.cresids = nn.ModuleList()
        for i in range(num_resids):
            self.cresids.append(CNResidual(32))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        
    def forward(self, x):
        # call layers
        x = x.reshape(-1, 1, 187)
        x = self.c1(x)
        for cresid in self.cresids:
            x = cresid(x)
        x  = x.reshape(-1, 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class BiRNN(nn.Module):
    def __init__(self, is_lstm=True):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.LSTM(input_size=187, hidden_size=128, bidirectional=True, dropout=0.1, num_layers=2)
        if not is_lstm:
            self.encoder = nn.GRU(input_size=187, hidden_size=128, bidirectional=True, dropout=0.1, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = x.reshape(-1, 1, 187)
        output, _ = self.encoder(x)
        biout = output[:,:,:128] + output[:,:,128:]
        biout = biout.reshape(-1, 128)
        o = F.relu(self.fc1(biout))
        o = self.fc2(o)
        return o



def get_model(model_name):
    if model_name == "cnet":
        model = CNet()
    elif model_name == 'bilstm':
        model = BiRNN()
    elif model_name == 'bigru':
        model = BiRNN(is_lstm=False)
        
    if not model:
        raise Exception("Model with name " + str(model_name) + " not found")
    if device_count > 1:
        model = nn.DataParallel(model)
    return model.to(device)


def transfer_model(pretrained, freeze_params=True):
    if freeze_params:
        for param in pretrained.parameters():
            param.requires_grad = False
        pretrained.fc1 = nn.Linear(64,32)
        pretrained.fc2 = nn.Linear(32, 1)
    if device_count > 1:
        pretrained = nn.DataParallel(pretrained)
    pretrained = pretrained.to(device)
    return pretrained
