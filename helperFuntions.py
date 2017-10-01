import string
import nltk
import re
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def removePunctuation (sentence):
    exclude = set(string.punctuation)
    sentence_new = []
    for ch in sentence:
        if (ch == '-'):
            sentence_new.append(" ")
        else:
            sentence_new.append(ch)
    sentence = ''.join(sentence_new)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    return sentence

def removeNumbers(sentence):
    newSentence = []
    words = sentence.split()
    for w in words:
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(val is None):
            continue
        else:
            newSentence.append(w)
    return ' '.join(newSentence)

def removeStopWords(sentence):
    newSentence = []
    stopWords = set(nltk.corpus.stopwords.words('english'))
    words = sentence.split()
    for w in words:
        if (w in stopWords):
            continue
        else:
            newSentence.append(w)
    return ' '.join(newSentence)

def toLower(sentence):
    newSentence = []
    words = sentence.split()
    for w in words:
        if (w == "START" or w == "END"):
            newSentence.append(w)
        else:
            newSentence.append(w.lower())
    return ' '.join(newSentence)

# def preProcessSentence(sentence):
#     newSentence = []
#     stopWords = set(nltk.corpus.stopwords.words('english'))
#     # stopWords = []
#     sentence = sentence.lower()
#     # print sentence
#     sentence = removePunctuation(sentence)
#     words = sentence.split()
#     for w in words:
#          #check if the word stats with an alphabet
#         val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
#         #ignore if it is a stop word
#         if(w in stopWords or val is None):
#             continue
#         else:
#             newSentence.append(w)
#     return ' '.join(newSentence)

def bigramReturner (sentence):
    words = sentence.split()
    # print nltk.bigrams(words)
    bigrams = nltk.bigrams(words)
    bigramFeatureVector = []
    for item in bigrams:
        bigramFeatureVector.append(' '.join(item))

    return bigramFeatureVector

def getFeatureVector(sentence):
    featureVector = []
    words = sentence.split()
    for w in words:
        featureVector.append(w.lower())
    return featureVector

def best_word_feats(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])

def best_bigram_word_feats(words, n=500):
    score_fn = BigramAssocMeasures.chi_sq
    bigram_finder = BigramCollocationFinder.from_words(words, 2)
    # print bigram_finder
    bigrams = bigram_finder.nbest(score_fn, n)
    # print bigrams
    # d = dict([(bigram, True) for bigram in bigrams])
    # d.update(best_word_feats(words))
    return bigrams