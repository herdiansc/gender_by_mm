from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy
import os

seed = 7
numpy.random.seed(seed)

dataset = numpy.genfromtxt('training_data.csv', delimiter=',', skip_header=1)

X = dataset[:,0:31]
Y = dataset[:,31]

mask = ~numpy.any(numpy.isnan(X), axis=1)

X = X[mask]
Y = Y[mask]

model = Sequential()
model.add(Dense(12, input_dim=31, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=500, batch_size=20)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model structure to json
model_json = model.to_json()
with open("training_data.json", "w") as json_file:
    json_file.write(model_json)

# save model weights to HDF5
model.save_weights("training_data.h5")