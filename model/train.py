import utils
import torch
import random
import time
from torch import optim
import createTensors
import torch.nn as nn

def train(inputTensor, targetTensor, encoder, decoder, encoderOptimzer, decoderOptimizer, criterion, maxLength=utils.MAX_LENGTH):

    encoderHidden = encoder.initHidden()

    encoderOptimzer.zero_grad()
    decoderOptimizer.zero_grad()

    inputLength = inputTensor.size(0)
    targetLength = targetTensor.size(0)

    encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize, device = utils.DEVICE)

    loss = 0

    for ei in range(inputLength):
        encoderOutput, encoderHidden = encoder(inputTensor[ei], encoderHidden)

        encoderOutputs[ei] = encoderOutput[0,0]
    
    decoderInput = torch.tensor([[utils.SOS_TOKEN]], device=utils.DEVICE)

    decoderHidden = encoderHidden

    useTeacherForcing = True if random.random() < utils.TEACHER_FORCING_RATIO else False

    if useTeacherForcing:
        for di in range(targetLength):
            decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

            loss += criterion(decoderOutput, targetTensor[di])

            decoderInput = targetTensor[di]
    
    else:
        for di in range(targetLength):
            decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

            topv, topi = decoderOutput.topk(1)

            decoderInput = topi.squeeze().detach()

            loss += criterion(decoderOutput, targetTensor[di])

            if decoderInput.item() == utils.EOS_TOKEN:
                break
    
    loss.backward()

    encoderOptimzer.step()
    decoderOptimizer.step()

    torch.cuda.empty_cache()

    return loss.item() / targetLength

def trainIters(inputLang, outputLang, pairs, encoder, decoder, nIters, printEvery = 1000, plotEvery = 100, learningRate=utils.LR):

    startTime = time.time()
    plotLosses = []
    printLossTotal = 0
    plotLossTotal = 0

    encoderOptimzer = optim.SGD(encoder.parameters(), lr = learningRate)

    decoderOptimizer = optim.SGD(decoder.parameters(), lr=learningRate)

    trainingPairs = [createTensors.tensorFromPair(random.choice(pairs), inputLang, outputLang) for i in range(nIters)]

    criterion = nn.NLLLoss()

    for i in range(1, nIters+1):
        trainingPair = trainingPairs[i-1]
        inputTensor = trainingPair[0]
        targetTensor = trainingPair[1]

        loss = train(inputTensor, targetTensor, encoder, decoder, encoderOptimzer, decoderOptimizer, criterion)

        printLossTotal+=loss
        plotLossTotal+=loss

        if i % printEvery == 0:
            printLossAvg = printLossTotal / printEvery

            printLossTotal = 0

            print('%s (%d %d) %.4f' % (utils.timeSince(startTime, i/nIters), i, i/nIters*100, printLossAvg))
        
        if i % plotEvery == 0:
            plotLossAvg = plotLossTotal/plotEvery

            plotLosses.append(plotLossAvg)
            plotLossTotal = 0
    
    torch.cuda.empty_cache()

    utils.showPlot(plotLosses)

def evaluate(inputLang, outputLang, encoder, decoder, sentence, maxLength = utils.MAX_LENGTH):
    with torch.no_grad():
        inputTensor = createTensors.tensorFromSentence(inputLang, sentence)
        inputLength = inputTensor.size()[0]

        encoderHidden = initHidden()
        encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize, device = utils.DEVICE)

        for ei in ranger(inputLength):
            encoderOutput, encoderHidden = encoder(inputTensor[ei],encoderHidden)

            encoderOutputs[ei] += encoderOutput[0,0]

        decoderInput = torch.tensor([[utils.SOS_TOKEN]], device =utils.DEVICE)

        decoderHidden = encoderHidden

        decoderWords = []

        decoderAttentions = torch.zeros(maxLength, maxLength)

        for di in range(maxLength):
            decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs)

            decoderAttentions[di] = decoderAttention.data

            topv, topi = decoderOutput.topk(1)

            if topi.item()==utils.EOS_TOKEN:
                decoderWords.append('<EOS>')
                break
            else:
                decoderWords.append(outputLang.indexToWord[topi.item()])
            
            decoder_input = topi.squeeze().detach()
        
        torch.cuda.empty_cache()

        return decoderWords, decoderAttentions[:di+1]

def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        outputWords, attentions = evaluate(encoder, decoder, pair[0])
        outputSentence = ' '.join(outputWords)
        print('<', outputSentence)
        print('')
    
    torch.cuda.empty_cache()