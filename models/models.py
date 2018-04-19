import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    I. CNN model

    CNN(
      (layer1): Sequential(
        (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
      )
      (layer2): Sequential(
        (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): Dropout2d(p=0.15)
        (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        (3): ReLU()
        (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
      )
      (fc): Sequential(
        (0): Linear(in_features=2048, out_features=120, bias=True)
        (1): Dropout(p=0.15)
        (2): Linear(in_features=120, out_features=36, bias=True)
      )
    )"""
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self._num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.15),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 32, 120),
            nn.Dropout(p=0.15),
            nn.Linear(120, self._num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)