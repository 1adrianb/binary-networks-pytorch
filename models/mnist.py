import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import LearnableBias


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1)
        self.move1_c = LearnableBias(1)
        self.move1_a = LearnableBias(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.move2_c = LearnableBias(128)
        self.move2_a = LearnableBias(64)

        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x = self.move1_c(x)
        x = self.conv1(x)
        # x = self.move1_a(x)
        x = F.relu(x)
        # x = self.move2_c(x)
        x = self.conv2(x)
        # x = self.move2_a(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # print(self.move2_a.bias)
        output = F.log_softmax(x, dim=1)
        # print(self.conv1.weight)
        return {"preds": output}
