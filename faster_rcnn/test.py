import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

x = torch.randn((100, 10))
y = torch.randn((100, 1))


class TwoMLPNet(nn.Module):
    def __init__(self):
        super(TwoMLPNet, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        x = F.relu_(self.fc3(x))
        return x


model = TwoMLPNet()

epochs = 30
lr = 0.05

params = model.parameters()
optimizer = optimizer.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

for epoch in range(epochs):
    model.train()
    for data, label in zip(x, y):
        predict = model(data)
        loss = F.mse_loss(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch{}".format(epoch), "loss:{}".format(loss))

