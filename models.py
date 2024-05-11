import torch.nn as nn
import torch
torch.manual_seed(1)

class CrnnModel(nn.Module):
  #initialize model
  def __init__(self):
    #call superclass
    super(CrnnModel, self).__init__()
    #add dropout for avoiding overfitting
    self.dropout = nn.Dropout(0.5)
    #convolution block
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=4)
    )
    #fully-connected layer
    self.GruLayer = nn.GRU(input_size=2048,
                        hidden_size=256,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False)

    self.GruLayerF = nn.Sequential(nn.BatchNorm1d(2048),
                                       nn.Dropout(0.6))

    self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
                                  nn.ReLU(),
                                  nn.Dropout(0.5))

    self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                  nn.ReLU(),
                                  nn.Dropout(0.5))

    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                nn.Softmax(dim=1))

  #forward
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    #flatten Tensor
    x = x.contiguous().view(x.size()[0],x.size()[2] , -1)
    x, _ = self.GruLayer(x)
    x = x.contiguous().view(x.size()[0],  -1)
    # out_features=4096
    x = self.GruLayerF(x)
    x = self.fcBlock1(x)
    x = self.fcBlock2(x)
    x = self.output(x)
    return x