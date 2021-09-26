import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df  = pd.read_csv('./miles.csv')
df.info
print("Painting the correlations")
#Once we load seaborn into the session, everytime a matplotlib plot is executed, seaborn's default customizations are added
sns.scatterplot(df['Kilometres'], df['Miles'])
print("Define input(X) and output(Y) variables")
X_train = df['Kilometres']
y_train = df['Miles']

print("Creating the model")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

print("Compiling the model")
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

print ("Training the model")
epochs_hist = model.fit(X_train, y_train, epochs = 250)

print("Evaluating the model")
print(epochs_hist.history.keys())

#graph
plt.plot(epochs_hist.history['loss'])
plt.title('Evolution of the error associated with the model')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

#prediction
kilometers = 100
predictedMiles = model.predict([kilometers])
print("The conversion from Kilometres ( { kilometers } ) to Miles is as follows: ( { predictedMiles } ))

plt.show()
