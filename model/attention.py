import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Attention(nn.Module):
    def __init__(self, hiddenSize, outputSize, dropoutP=0.1, maxLength=utils.MAX_LENGTH):
        
        self.device = utils.DEVICE

        super(Attention, self).__init__()

        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.dropoutP = dropoutP
        self.maxLength = maxLength

        self.embedding = nn.Embedding(self.outputSize, self.hiddenSize)
        self.attention = nn.Linear(self.hiddenSize*2, self.maxLength)
        self.attentionCombine = nn.Linear(self.hiddenSize*2, self.hiddenSize)
        self.dropout = nn.Dropout(self.dropoutP)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.outputSize)
    
    def forward(self, input, hidden, encoderOutputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        attentionWeights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )

        attentionApplied = torch.bmm(
            attentionWeights.unsqueeze(0), encoderOutputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attentionApplied[0]), 1)
        output = self.attentionCombine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attentionWeights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize, device = self.device)

