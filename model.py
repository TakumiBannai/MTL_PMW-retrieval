import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTwoTask(nn.Module):
    def __init__(self):
        super(CNNTwoTask, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=13,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, 2, 1),
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 2, 2, 0),
                                         torch.nn.BatchNorm2d(128),
                                         torch.nn.ReLU())

        self.mlp1 = torch.nn.Linear(2 * 2 * 128, 100)
        self.dropout = nn.Dropout(0.5)
        self.mlp2_1 = torch.nn.Linear(100, 2)
        self.mlp2_2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.dropout(x)
        y = self.mlp2_1(x)
        z = self.mlp2_2(x)
        return y, F.relu(z)

    def __repr__(self):
        return "CNNTwoTask"


# Single task learning
class CNNSingleTask(nn.Module):
    def __init__(self, task="Clsf") :
        self.task = task
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, stride=2, padding=1), 
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, 2, 1),
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 2, 2, 0),
                                         torch.nn.BatchNorm2d(128),
                                         torch.nn.ReLU())

        self.fc1 = torch.nn.Linear(2 * 2 * 128, 100)
        self.fc_clsf = torch.nn.Linear(100, 2)
        self.fc_reg = torch.nn.Linear(100, 1)
        self.dropout_fc1 = nn.Dropout(0.5)

    def forward(self, x):
        # Conv.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layer
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        if self.task=="Clsf":
            clsf = self.fc_clsf(x)
            return clsf
        elif self.task=="Reg":
            reg = self.fc_reg(x)
            return F.relu(reg)

    def __repr__(self):
        return "CNNSingleTask"


class CNNMultiTask(nn.Module):
    def __init__(self):
        super(CNNMultiTask, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=13,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, 2, 1),
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 2, 2, 0),
                                         torch.nn.BatchNorm2d(128),
                                         torch.nn.ReLU())

        self.mlp1 = torch.nn.Linear(2 * 2 * 128, 100)
        self.dropout = nn.Dropout(0.5)
        self.mlp2_1 = torch.nn.Linear(100, 4)
        self.mlp2_2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.dropout(x)
        y = self.mlp2_1(x)
        z = self.mlp2_2(x)
        return y, F.relu(z)


    def __repr__(self):
        return "CNNMultiTask"
