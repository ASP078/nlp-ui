import torch
import torch.nn as nn
import utils

class Encoder(nn.Module):
    def __init__(self, inputSize:int, hiddenSize:int):
        super(Encoder, self).__init__()

        self.device = utils.DEVICE

        self.hiddenSize = hiddenSize

        self.embedding = nn.Embedding(inputSize, hiddenSize)

        self.gru = nn.GRU(hiddenSize, hiddenSize)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)

        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hiddenSize, device = self.device)