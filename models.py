import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self):
        super().__init__()
        # layers
        self.c1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.cresid1 = CNResidual(32)
        self.cresid2 = CNResidual(32)
        self.cresid3 = CNResidual(32)
        self.cresid4 = CNResidual(32)
        self.cresid5 = CNResidual(32)
#         self.cresids = []
#         for i in range(NUM_RESIDUAL_BLOCKS):
#             self.cresids.append(CNResidual(32).to(device))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        
    def forward(self, x):
        # call layers
        x = self.c1(x)
        x = self.cresid1(x)
        x = self.cresid2(x)
        x = self.cresid3(x)
        x = self.cresid4(x)
        x = self.cresid5(x)
#         for cresid in self.cresids:
#             x = cresid(x)
        x  = x.reshape(-1, 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_model(model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count()

    if model_name == "cnet":
        model = CNet()
        
    if not model:
        raise Exception("Model with name " + str(model_name) + " not found")
    if device_count > 1:
        model = nn.DataParallel(model)
    return model.to(device)

