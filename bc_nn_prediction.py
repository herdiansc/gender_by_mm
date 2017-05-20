from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy
import os

seed = 7
numpy.random.seed(seed)

dataset = numpy.array([[5,1,1,1,1,1,5,5,1,1,1,1,1,5,1,5,1,1,5,5,1,1,1,5,5,1,5,5,1,1,1]])

X = dataset[:,0:31]

# load json and create model
json_file = open('training_data.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("training_data.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

predictions = loaded_model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(predictions)
print(rounded)
if rounded[0] == 0:
    print('Female')
else:
    print('Male')