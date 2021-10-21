#!/usr/bin/env python
#import spacy library
import spacy

#load core english library
nlp = spacy.load("en_core_web_sm")

#take unicode string
#here u stands for unicode
doc = nlp(u"Clairson International Corp. said it expects to report a net loss for its second quarter ended March 26 and doesn't expect to meet analysts' profit estimates of $3.0 to $4 million, or 1,276 cents a share to 1,279 cents a share, for its year ending Sept. 24. (From the Wall Street Journal (1988))")
#to print sentences
for sent in doc.sents:
  print(sent)
