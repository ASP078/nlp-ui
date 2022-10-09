class Language():
    """
        Class for datta storing and counting in the form of text/word
    """
    def __init__(self, name:str):
        self.name = name
        
        self.wordToIndex = dict()
        self.wordToCount = dict()

        self.indexToWord = {0:"SOS", 1:"EOS"}

        self.nWords = 2

    def addSentence(self, sentence:str):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.wordToIndex:
            self.wordToIndex[word] = self.nWords
            
            self.wordToCount[word] = 1

            self.indexToWord[self.nWords] = word

            self.nWords+=1
        else:
            self.wordToCount[word]+=1
    



