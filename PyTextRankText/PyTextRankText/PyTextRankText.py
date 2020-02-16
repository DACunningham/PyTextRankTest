#import spacy
#import pytextrank

## example text
#text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

## load a spaCy model, depending on language, scale, etc.
#nlp = spacy.load("en_core_web_sm")

## add PyTextRank to the spaCy pipeline
#tr = pytextrank.TextRank()
#nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

#doc = nlp(text)

## examine the top-ranked phrases in the document
#for p in doc._.phrases:
#    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
#    print(p.chunks)

#!/usr/bin/env python
# encoding: utf-8

import logging
import pytextrank
import spacy
import sys

######################################################################
## sample usage
######################################################################

# load a spaCy model, depending on language, scale, etc.

nlp = spacy.load("en_core_web_sm")

# logging is optional: to debug, set the `logger` parameter
# when initializing the TextRank object

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("PyTR")

# add PyTextRank into the spaCy pipeline

tr = pytextrank.TextRank(logger=None)
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

# parse the document

with open("doc1.txt", "r") as f:
    text = f.read()

doc = nlp(text)

print("pipeline", nlp.pipe_names)
print("elapsed time: {} ms".format(tr.elapsed_time))


# examine the top-ranked phrases in the document

for phrase in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(phrase.rank, phrase.count, phrase.text))
    print(phrase.chunks)


# generate a GraphViz doc to visualize the lemma graph

tr.write_dot(path="lemma_graph.dot")

print("\n----\n")


# switch to a longer text document...

with open("doc2.txt", "r") as f:
    text = f.read()

doc = nlp(text)

for phrase in doc._.phrases[:20]:
    print(phrase)

print("\n----\n")

# summarize the document based on the top 15 phrases, 
# yielding the top 5 sentences...

for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=5):
    print(sent)

print("\n----\n")


# to show use of stopwords, first we output a baseline...

with open("doc3.txt", "r") as f:
    text = f.read()

doc = nlp(text)

for phrase in doc._.phrases[:10]:
    print(phrase)

print("\n----\n")

# now add `("gensim", "PROPN")` to the stop words list
# then see how the top-ranked phrases differ...

tr.load_stopwords(path="stopwords.txt")

doc = nlp(text)

for phrase in doc._.phrases[:10]:
    print(phrase)