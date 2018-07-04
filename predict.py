from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

seed = 7
numpy.random.seed(seed)

# "Music","Slow songs or fast songs","Dance","Folk","Country","Classical music","Musical","Pop","Rock","Metal or Hardrock","Punk",
# "Hiphop, Rap","Reggae, Ska","Swing, Jazz","Rock n roll","Alternative","Latino","Techno, Trance","Opera","Movies","Horror",
# "Thriller","Comedy","Romantic","Sci-fi","War","Fantasy/Fairy tales","Animated","Documentary","Western","Action"
dataset = numpy.array([[5,4,1,1,4,1,5,5,5,5,3,5,1,5,1,5,1,1,5,5,1,1,1,5,5,1,5,5,1,1,1]])
# dataset = numpy.array([[3,2,1,1,1,1,3,3,5,3,1,1,1,1,1,2,4,1,1,1,4,2,3,4,2,5,5,3,2,3,4]])
# dataset = numpy.array([[5,3,2,3,2,3,3,2,5,5,3,4,3,4,4,5,3,1,3,5,5,5,5,2,3,3,4,3,3,2,4]])


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
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = loaded_model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(predictions)
print(rounded)
if rounded[0] == 0:
    print('Female')
else:
    print('Male')