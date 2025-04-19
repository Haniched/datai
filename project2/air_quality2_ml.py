import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle #save encoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("Air_Quality.csv")
         
df = df.loc[df['Name'] == 'Fine particles (PM 2.5)']
df['Start_Date'] = pd.to_datetime(df['Start_Date'])

# Data handling
X = df.loc[:, ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Join ID', 'Geo Place Name', 'Time Period', 'Start_Date']]
y = df.loc[:, ['Data Value']]

X = X.fillna(0)
Xorg = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), 
                                     ['Name','Measure', 'Measure Info', 'Geo Type Name', 'Geo Join ID', 'Geo Place Name', 'Time Period', 'Start_Date']
                                     )], 
                                    remainder = 'passthrough')
X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

# Feature Scaling
scaler_x = StandardScaler(with_mean=False)
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)

y_test = scaler_y.transform(y_test)
model = Sequential()    
model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history=model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test,y_test))

# Visualisoidaan mallin oppiminen
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.grid(True)
plt.show()

y_pred = scaler_y.inverse_transform(model.predict(X_test))

y_test = scaler_y.inverse_transform(y_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print ('\nann:')
print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}\n')



model.save('air_quality2-model.keras')

# save encoder to disk
with open('air_quality2-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
    
# save scalers to disk
with open('air_quality2-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)
    
with open('air_quality2-scaler_y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)

