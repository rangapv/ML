#!/usr/bin/env python
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "She was running and coding at the same and I thought this was the craziest things I had ever seen."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

print("{0:20}{1:20}".format("Word","Stemmed"))
for w in sentence_words:
    print("{0:20}{1:20}".format(w,ps.stem(w)))

print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word, pos="v")))
