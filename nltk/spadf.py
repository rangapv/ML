#!/usr/bin/env python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
texts = [
"""Imagine this: instead of sending a four-hundred-pound rover vehicle to Mars, we merely shoot over to the planet a single sphere, one that can fit on the end of a pin. Using energy from sources around it, the sphere divides itself into a diversified army of similar spheres. The spheres hang on to each other and sprout features: wheels, lenses, temperature sensors, and a full internal guidance system. You'd be gobsmacked to watch such a system discharge itself.""" ,

'The countries of Haiti and the Dominican Republic share the Caribbean island of Hispaniola. Consider what would happen if a tsunami were to slam into the Dominican Republic and make it uninhabitable. One possibility is that the Dominicans would be erased from the map and Haiti would continue business as usual. But there’s a second possibility: What if the Haitians shifted their nation several hundred miles to the west, bigheartedly accommodating the Dominicans by shrinking their own territory and sharing what remained? In this case, thanks to neighboring generosity, the two nations would be harmoniously compressed onto a smaller, remaining bit of real estate.'
]
df = pd.DataFrame({'Text': ['text1', 'text2'], 'text':texts})

# initialize
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
# create matrix
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['Text'].values, columns=cv.get_feature_names())
print(df_dtm)
