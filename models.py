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
    def mimodel(self):
        self.fc2 = nn.Linear(32, 1) 
    
class SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        self.c1 = nn.Conv1d(11, 32, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.c2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.c3 = nn.Conv1d(64, 128, kernel_size=2, stride=1)
        self.rnn = nn.LSTM(256, 100, 1, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        # call layers
        x = x.reshape(-1, 11, 17)
        x = F.relu(self.c1(x))
        x = self.pool1(x)
        x = self.pool2(F.relu(self.c2(x)))
        x = F.relu(self.c3(x))
        s = x.size()
#         print(x.size())
        x = x.reshape(-1, 1, s[1]*s[2])
        output, (hidden, cell) = self.rnn(x)
#         print(hidden.size())
        return hidden, cell

class SeqDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        self.rnn = nn.LSTM(256, 100, 1, batch_first=True, bidirectional=True)
        
    def forward(self, x, hidden, cell):
        # call layers
        out, (hn, cn) = self.rnn(x, (hidden, cell))
#         print(out.size())
        out = out.squeeze(1)
        out = out[:, :100] + out[:, 100:]
        return out

class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SeqEncoder()
        self.decoder = SeqDecoder()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 5)
    def forward(self, x):
        batch_size = x.size()[0]
        h, c = self.encoder(x)
        x_decode = torch.zeros(batch_size, 1, 256).to(device)
        out = self.decoder(x_decode, h, c)
        return self.fc2(F.relu(self.fc1(out)))
    def mimodel(self):
        self.fc2 = nn.Linear(50, 1) 
    
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
    def mimodel(self):
        self.fc2 = nn.Linear(64, 1) 



def get_model(model_name, mi=False, transfer=''):
    if model_name == "cnet":
        model = CNet()
    elif model_name == "seq":
        model = SeqModel()
    elif model_name == 'bilstm':
        model = BiRNN()
    elif model_name == 'bigru':
        model = BiRNN(is_lstm=False)
    
    if transfer:
        model.load_state_dict(torch.load(transfer))
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad=False
    if mi:
        model.mimodel()
    
    if not model:
        raise Exception("Model with name " + str(model_name) + " not found")
    if device_count > 1:
        model = nn.DataParallel(model)
    return model.to(device)


def transfer_model(pretrained, freeze_params=True, fc2_input=32, fc1_input=64):
    if freeze_params:
        for param in pretrained.parameters():
            param.requires_grad = False
    pretrained.fc1 = nn.Linear(fc1_input, fc2_input)
    pretrained.fc2 = nn.Linear(fc2_input, 1)
    if device_count > 1:
        pretrained = nn.DataParallel(pretrained)
    pretrained = pretrained.to(device)
    return pretrained
