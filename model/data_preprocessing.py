import unicodedata
import re
import utils
from lang import Language

def unicodeToASCII(s:str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s:str) -> str:
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)

    return s

def readLanguages(langOne, langTwo, reverse=False):
    
    print("Reading lines .... ")

    langOneLines = open(utils.ENGLISH_TEXT_FILE, encoding='utf-8').read().strip().split('\n')

    langTwoLines = open(utils.HINDI_TEXT_FILE, encoding='utf-8'
    ).read().strip().split('\n')

    pairs = [
        [normalizeString(langOneLines[i]), normalizeString(langTwoLines[i])] for i in range(len(langOneLines))
        ]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

        inputLang = Language(langTwo)
        outputLang = Language(langOne)
    
    else:
        inputLang = Language(langOne)
        outputLang = Language(langTwo)
    
    return inputLang, outputLang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < utils.MAX_LENGTH and len(p[1].split(' ')) < utils.MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]