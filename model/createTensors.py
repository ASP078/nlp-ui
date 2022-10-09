import utils
import torch

def indexesFromSentence(lang, sentence):
    return [lang.wordToIndex[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)

    indexes.append(utils.EOS_TOKEN)

    return torch.tensor(indexes, dtype=torch.long, device=utils.DEVICE).view(-1,1)

def tensorFromPair(pair, inputLang, outputLang):
    inputTensor = tensorFromSentence(inputLang, pair[0])
    targetTensor = tensorFromSentence(outputLang, pair[1])

    return (inputTensor, targetTensor)