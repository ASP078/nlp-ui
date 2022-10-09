import data_preprocessing
import random

def prepareData(langOne, langTwo, reverse=False):
    inputLang, outputLang, pairs = data_preprocessing.readLanguages(langOne, langTwo, reverse)

    print(f"Read {len(pairs)} sentence pairs")

    pairs = data_preprocessing.filterPairs(pairs)

    print(f"Trimmed to {len(pairs)} sentence pairs")

    print("Counting words.....")

    for pair in pairs:
        inputLang.addSentence(pair[0])
        outputLang.addSentence(pair[1])
    
    print("Counted words:")

    print(inputLang.name, inputLang.nWords)

    print(outputLang.name, outputLang.nWords)

    return inputLang, outputLang, pairs

def main():
    inputLang, outputLang, pairs = prepareData("eng", "hin", False)

    print(random.choice(pairs))

if __name__ == "__main__":
    main()