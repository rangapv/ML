import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df  = pd.read_csv('./miles.csv')
df.info
print("Painting the correlations")
#Once we load seaborn into the session, everytime a matplotlib plot is executed, seaborn's default customizations are added
sns.scatterplot(df['Kilometres'], df['Miles'])
plt.show()
