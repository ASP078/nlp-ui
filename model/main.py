from encoder import Encoder
from attention import Attention
import utils
import data
from train import trainIters, evaluateRandomly
import torch

def main():
    torch.cuda.empty_cache()

    inputLang, outputLang, pairs = data.prepareData("eng", "hin", False)

    encoder = Encoder(inputLang.nWords, utils.HIDDEN_SIZE).to(utils.DEVICE)

    attentionDecoder = Attention(utils.HIDDEN_SIZE, outputLang.nWords, dropoutP=0.1).to(utils.DEVICE)

    trainIters(inputLang, outputLang, pairs, encoder, attentionDecoder, nIters=40000, printEvery=4000, plotEvery=100)

    # evaluateRandomly(pairs, encoder, attentionDecoder)

    torch.save(encoder.state_dict, 'models/encoder@40kepoch-512-0,0023err.pth')
    torch.save(attentionDecoder.state_dict, 'models/attentionDecoder@40kepoch-512-0,0023err.pth')

if __name__ == "__main__":
    main()