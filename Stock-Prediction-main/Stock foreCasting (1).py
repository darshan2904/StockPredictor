#!/usr/bin/env python
# coding: utf-8

# In[24]:


import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler




api_key = '3CHVFUMIJNISRPKP'
symbol = input("Enter The Stock Code:")

#RELIANCE.BSE

start_date_str = '2016-01-01'
end_date_str = input("Enter Todays Date(YYYY-MM-DD):")


start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}'

# Send the request and store the response
response = requests.get(url)

# Convert the response data to a JSON object
data = json.loads(response.text)

# Extract the daily time series data from the JSON object
daily_data = data['Time Series (Daily)']


dates = []
closing_prices = []

# Loop through the daily data and append dates and closing prices to the lists for the desired time period
for date, values in daily_data.items():
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    if start_date <= date_obj <= end_date:
        dates.append(date)
        closing_prices.append(float(values['4. close']))

# Reverse the lists to get the data in chronological order
dates.reverse()
closing_prices.reverse()

data = pd.DataFrame({'Close' : closing_prices})
temp = pd.DataFrame({'Date' : dates})


def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime(year=year, month=month, day=day)


temp['Date'] = temp['Date'].apply(str_to_datetime)
# Plot the closing prices on a graph
plt.plot(temp['Date'], data['Close'])

# Set the title and axis labels
plt.title(f'{symbol} Closing Prices ({start_date_str} - {end_date_str})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
# Display the graph
plt.show()



# In[25]:


scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(data).reshape(-1,1))


# In[26]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[27]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[28]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[29]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[30]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[31]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[32]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[33]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[34]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[35]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[36]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[37]:


len(test_predict)
test_data[100:]
size = len(test_data) - 100
x_input=test_data[size:].reshape(1,-1)
x_input.shape


# In[38]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[39]:


# demonstrate prediction for next 300 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[40]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[41]:


import matplotlib.pyplot as plt
len(df1)


# In[42]:


size1  = len(df1) -100
plt.plot(day_new,scaler.inverse_transform(df1[size1:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[43]:


size1 = size1 + 25
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[size1:])


# In[ ]:




