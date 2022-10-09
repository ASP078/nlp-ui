import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Decoder(nn.Module):
    def __init__(self, hiddenSize:int, outputSize:int):
        super(Decoder, self).__init__()

        self.device = utils.DEVICE

        self.hiddenSize = hiddenSize

        self.embedding = nn.Embedding(outputSize, hiddenSize)

        self.gru = nn.GRU(hiddenSize, hiddenSize)

        self.out = nn.Linear(hiddenSize, outputSize)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize, device=self.device)