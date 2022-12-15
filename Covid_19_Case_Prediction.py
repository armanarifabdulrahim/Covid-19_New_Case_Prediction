
#%%
# Import Module
import os
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard



#%%
# Data Loading

TRAIN_CSV = 'cases_malaysia_train.csv'
TEST_CSV = 'cases_malaysia_test.csv'
CSV_PATH = os.path.join(os.getcwd(), 'Dataset')

train_df = pd.read_csv(os.path.join(CSV_PATH,TRAIN_CSV))





#%%
# Data Inspection

train_df.head()
train_df.info()
train_df.describe()
train_df.isna().sum()

plt.figure()
plt.plot(train_df['cases_new'])
plt.title('Covid 19 New Case Original')
plt.show()





#%%
# Data Cleaning

#Convert cases_new to int/float
new_case = train_df['cases_new']
new_case = pd.to_numeric(new_case, errors='coerce')
new_case.info()
new_case.isna().sum()

#Solve NA
new_case = new_case.interpolate(method='polynomial', order=2)
new_case.isna().sum()

plt.figure()
plt.plot(new_case)
plt.title('Covid 19 New Case Cleaned')
plt.show()





#%%
# Feature Extraction


#%%
# Data Pre-processing

#Reshape data
data = new_case.values
data = data[::, None]

#Min Max Scaler
mm_scaler = MinMaxScaler()
data = mm_scaler.fit_transform(data)

#Select a range of data to train 
window = 30 #30 days
X_train = []
y_train = []

for i in range(window,len(data)):
    X_train.append(data[i-window:i])
    y_train.append(data[i])

#Convert to array
X_train = np.array(X_train)
y_train = np.array(y_train)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, random_state=123)





#%%
# Model Development

#Using Sequentila model
model = Sequential()
model.add(LSTM(64, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

#Compile model
model.compile(optimizer='adam',
    loss='mse',
    metrics=['mse', 'mape']
    )

#Callback - Early Stopping, TesorBoard
logdir = os.path.join(os.getcwd(), 
    'logs', 
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )

es = EarlyStopping(monitor='val_loss',
    patience=2
    )

tb = TensorBoard(log_dir=logdir)

#train model
history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=25, callbacks=[es,tb])





#%%
# Model Evaluation

#Load test dataset
test_df = pd.read_csv(os.path.join(CSV_PATH,TEST_CSV))
test_df.info()

#Clean test data
newcase_test = test_df['cases_new']
newcase_test = newcase_test.interpolate(method='polynomial', order=2)
newcase_test.columns = ['cases_new']
newcase_test.isna().sum()

#Combine train data and test data
df_plus = pd.concat((new_case, newcase_test), axis=0)
df_plus = df_plus[len(df_plus)-window-len(test_df):]

#Reshape new dataframe 
data = df_plus.values
data = data[::, None]

#Normalize Test Data
data_scaled = mm_scaler.transform(data)

#Select a range of data to test
data_scaled = data_scaled[0:len(newcase_test) + window]

X_test =[]
y_test =[]

for i in range(window,len(data_scaled)):
    X_test.append(data_scaled[i-window:i])
    y_test.append(data_scaled[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

#Prediction
predicted_price = model.predict(X_test)

#metrics to evaluate the performance
print('Mean Absolute Percentage Error : \n', mean_absolute_percentage_error(y_test,predicted_price))


#plot the graph
y_test = mm_scaler.inverse_transform(y_test)
predicted_price = mm_scaler.inverse_transform(predicted_price)

plt.figure()
plt.plot(predicted_price,color='green')
plt.plot(y_test, color='purple')
plt.legend(['Predicted','Actual'])
plt.xlabel('Time')
plt.ylabel('New Case')
plt.show()


#metrics to evaluate the performance

print('Mean Absolute Percentage Error : \n', mean_absolute_percentage_error(y_test,predicted_price))





#%%
# Save Model

with open('mms.pkl','wb') as f:
    pickle.dump(mm_scaler,f)

model.save('covid19_prediction_model.h5')


