import os
import nltk
import string
import helperFuntions
import sklearn
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
import numpy as np

parentDir = os.path.dirname(os.getcwd())
currentDir = parentDir + "/sentimentAnalyser/"
dataNeg = currentDir + "rt-polarity.neg"
dataPos = currentDir + "rt-polarity.pos"
dataNeg_new = currentDir + "rt-polarity_new.neg"
dataPos_new = currentDir + "rt-polarity_new.pos"
feature = currentDir + "training_features.txt"
feature_output = currentDir + "training_features_output.txt"
feature_without_bigram = currentDir + "training_features_Unigram.txt"
feature_output_without_bigram = currentDir + "training_features_output_unigram.txt"

linesNeg = open(dataNeg).read().splitlines()

linesPos = open(dataPos).read().splitlines()

print linesNeg

with open(dataNeg, 'r') as src:
    with open(dataNeg_new, 'w') as dest:
       for line in src:
           dest.write('%s %s %s\n' % ("START", line.rstrip('\n'), "END"))


with open(dataPos, 'r') as src:
    with open(dataPos_new, 'w') as dest:
       for line in src:
           dest.write('%s %s %s\n' % ("START", line.rstrip('\n'), "END"))

wordsNeg = open(dataNeg_new).read().splitlines()
wordsPos = open(dataPos_new).read().splitlines()

i = 0
wordsPos_new = []
for line in wordsPos:
    # if (i == 3700):
    #     break
    # else:
    wordsPos_new.append(line)
        # i += 1
wordsPos = '\n'.join(wordsPos_new)

i = 0
wordsNeg_new = []
for line in wordsNeg:
    # if (i == 3700):
    #     break
    # else:
    wordsNeg_new.append(line)
        # i += 1
wordsNeg = '\n'.join(wordsNeg_new)

# print wordsNeg
# print wordsPos
# print wordsNeg
# remove = dict.fromkeys(map(ord, '\n ' + string.punctuation))
# print remove
# with open(dataNeg) as inputfile:
#     # f = inputfile.read()
#     # print f
#     f = inputfile.read()
#     print type(f)
#     f_new = f.translate(remove)
#
# print f_new


wordsNeg = helperFuntions.removePunctuation(wordsNeg)
wordsNeg = helperFuntions.toLower(wordsNeg)
wordsNeg = helperFuntions.removeNumbers(wordsNeg)
wordsPos = helperFuntions.removePunctuation(wordsPos)
wordsPos = helperFuntions.toLower(wordsPos)
wordsPos = helperFuntions.removeNumbers(wordsPos)
# print wordsNeg
# print wordsNeg


dataNegTrain = []
dataPosTrain = []
for x in range(5331):
    dataNegTrain.append(linesNeg[x])
    dataPosTrain.append(linesPos[x])
# print data_pos_train
# dataNegCross = []
# dataPosCross = []
# for x in range(3700,5331):
#     dataNegCross.append(linesNeg[x])
#     dataPosCross.append(linesPos[x])
# # print data_pos_cross
dataNegTest = []
dataPosTest = []
# for x in range(3700,5331):
#     dataNegTest.append(linesNeg[x])
#     dataPosTest.append(linesPos[x])
# print data_pos_test

trainSamples = {}

for x in dataNegTrain:
    trainSamples[x] = "neg"
for x in dataPosTrain:
    trainSamples[x] = "pos"

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()


featureUnigramPos = []
featureBigramPos = []

featureUnigramNeg = []
featureBigramNeg = []

for x in dataNegTrain:
    # print x
    xNew = helperFuntions.removePunctuation(x)
    xNew = helperFuntions.toLower(xNew)
    xNew = helperFuntions.removeNumbers(xNew)
    xNew = helperFuntions.removeStopWords(xNew)
    # featureBigramNeg.append(helperFuntions.bigramReturner(xNew))
    featureUnigramNeg.append(helperFuntions.getFeatureVector(xNew))
    # break

for x in dataPosTrain:
    # print x
    xNew = helperFuntions.removePunctuation(x)
    xNew = helperFuntions.toLower(xNew)
    xNew = helperFuntions.removeNumbers(xNew)
    xNew = helperFuntions.removeStopWords(xNew)
    # featureBigramPos.append(helperFuntions.bigramReturner(xNew))
    featureUnigramPos.append(helperFuntions.getFeatureVector(xNew))
    # break

for word in featureUnigramPos:
    word_fd.update(word)
    label_word_fd['pos'].update(word)

for word in featureUnigramNeg:
    word_fd.update(word)
    label_word_fd['neg'].update(word)

# print featureBigramPos
# print featureUnigramPos
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count


word_scores = {}

for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                           (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                           (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
# print word_scores
best = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:10000]
# print best
bestwords = set([w for w, s in best])
# print bestwords

wordsNeg = wordsNeg.split()
# print wordsNeg
wordsPos = wordsPos.split()

bigramsPos = helperFuntions.best_bigram_word_feats(wordsPos)
bigramsNeg = helperFuntions.best_bigram_word_feats(wordsNeg)

# print featureBigramPos
# print featureBigramNeg
# print bigramsNeg
# print bigramsPos

# print label_word_fd
bigramFeatureVector = []
for item in bigramsPos:
    bigramFeatureVector.append(' '.join(item))

for item in bigramsNeg:
    bigramFeatureVector.append(' '.join(item))

featureVector = []
for x in bestwords:
    featureVector.append(x)
# print featureVector
# for x in bigramFeatureVector:
#     featureVector.append(x)

F = np.array(featureVector)
# print featureVector
X = []
i = 0
for x in linesNeg:
    print i
    train_example = []
    xNew = helperFuntions.removePunctuation(x)
    xNew = helperFuntions.toLower(xNew)
    xNew = helperFuntions.removeNumbers(xNew)
    bigrams = helperFuntions.bigramReturner(xNew)
    xNew = helperFuntions.removeStopWords(xNew)
    # print xNew
    # break
    for y in F:
        if (y in xNew or y in bigrams):
            train_example.append(1)
        else:
            train_example.append(0)
    # train_example_new = np.array(train_example)
    # print train_example
    i += 1
    X.append(train_example)
    # if (i == 0):
    #     X = np.array(train_example)
    #     i += 1
    # else:
    #     X = np.append(X, train_example, 0)
    #     i += 1



for x in linesPos:
    train_example = []
    i += 1
    print i
    xNew = helperFuntions.removePunctuation(x)
    xNew = helperFuntions.toLower(xNew)
    xNew = helperFuntions.removeNumbers(xNew)
    bigrams = helperFuntions.bigramReturner(xNew)
    xNew = helperFuntions.removeStopWords(xNew)
    # print xNew
    # break
    for y in F:
        if (y in xNew or y in bigrams):
            train_example.append(1)
        else:
            train_example.append(0)
    # train_example_new = np.array(train_example)
    X.append(train_example)
    # X = np.append(X, train_example, 0)
with open(feature_without_bigram, 'w') as f:
    for item in X:
        f.write("%s\n" % item)


# X_new = np.array(X)
# np.savetxt(feature, X_new, delimiter=",")
Y = []
for i in range(10662):
    if (i >= 5331):
        Y.append(1)
    else:
        Y.append(0)

with open(feature_output_without_bigram, 'w') as f:
    for item in Y:
        f.write("%s\n" % item)


# Y_new = np.array(Y)
# np.savetxt(feature, Y_new, delimiter=",")