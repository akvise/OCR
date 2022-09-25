import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):

    def __init__(self, voc_len):
        super(CRNN, self).__init__()

        self.voc_len = voc_len

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d((1, 2), stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d((1, 2), stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)

        self.norm = nn.BatchNorm2d(512)

        self.rnn1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)

        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, self.voc_len)

    def forward(self, images):
        images = F.relu(self.conv1(images))
        images = self.pool1(images)
        images = F.relu(self.conv2(images))
        images = self.pool2(images)
        images = F.relu(self.conv3(images))
        images = F.relu(self.conv4(images))
        images = self.pool3(images)
        images = F.relu(self.conv5(images))

        images = self.norm(images)
        images = F.relu(self.conv6(images))
        images = self.norm(images)

        images = self.pool4(images)

        images = F.relu(self.conv7(images))

        batch, channel, height, width = images.size()

        images = images.view(batch, channel * height, width)
        images = images.permute(2, 0, 1)  # (width, batch, feature)

        images = self.linear1(images)

        # rnn
        rnn, _ = self.rnn1(images)
        rnn, _ = self.rnn2(rnn)

        out = self.linear2(rnn)

        return out
