#!/usr/bin/env python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sentence = """Clairson International Corp. said it expects to report a
net loss for its second quarter ended March 26 and doesn't expect to meet analysts' profit
estimates of $3.0 to $4 million, or
1,276 cents a share to 1,279 cents a share, for its year ending Sept. 24."""

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(sentence)

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
