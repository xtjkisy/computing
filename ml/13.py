!pip install tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline  
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


#next row--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

data = pd.read_csv('diabetes.csv')

print(data.sample(10))

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

X = data.drop('Outcome', axis=1)
Y = data['Outcome']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=5)

model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


#next row --------------------------------------------------------------------------------------


from sklearn.model_selection import train_test_split
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs =  200, verbose = 1)
