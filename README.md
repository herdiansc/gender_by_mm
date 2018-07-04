# Gender Classification with Keras
This is an experimentation on gender classification using neural network with keras. The classification will be done by building neural network model based on music and movie preferences.

This is an experimentation of simple neural network in python.

- This algorithm takes dataset for training from file training_data.csv.
- Build model and save its structure and weights to json and h5py
- Predict output(Male|Female) based on new input

## Dataset description
- Extracted from kaggle dataset(https://www.kaggle.com/miroslavsabo/young-people-survey) by using only music and movie preferences
- Contains 32 columns(31 inputs and 1 output(0=Female|1=Male))
- The 31 inputs are consists of music and movie preferences
- Each preferences has integer option 1 to 5 where 1 is very negative and all the way to 5 is positive(Example: Pop. 1=Dont enjoy at all, 5=Enjoy very much)
- Last output is in last column consist of (0;Female|1;Male)

## Research Question
- Based on listed music and movie preferences, is he/she male or female?

## Requirement
- Python: 3.6.0 |Anaconda 4.3.1 (64-bit)| (default, Dec 23 2016, 12:22:00) [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
- scipy: 0.18.1
- numpy: 1.11.3
- sklearn: 0.18.1
- tensorflow: 1.0.0
- keras: 2.0.4
- h5py
