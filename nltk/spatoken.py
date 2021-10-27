#!/usr/bin/env python
import spacy

#load core english library
nlp = spacy.load("en_core_web_sm")

text_english = """Imagine this: instead of sending a four-hundred-pound rover vehicle to Mars,
we merely shoot over to the planet a single sphere, one that can fit on the end of a pin.
Using energy from sources around it, the sphere divides itself into a diversified army of
similar spheres. The spheres hang on to each other and sprout features: wheels, lenses,
temperature sensors, and a full internal guidance system. You'd be gobsmacked to watch
such a system discharge itself."""

doc = nlp(text_english)

tokens = [token.text for token in doc]
print(tokens)
