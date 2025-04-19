import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Air_Quality.csv')

df = df.loc[df['Name'] == 'Fine particles (PM 2.5)']
df['Start_Date'] = pd.to_datetime(df['Start_Date'])

# Data handling
X = df.loc[:, ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Join ID', 'Geo Place Name', 'Time Period', 'Start_Date']]
y = df.loc[:, ['Data Value']]

X = X.fillna(0)
Xorg = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), 
                                     ['Name','Measure', 'Measure Info', 'Geo Type Name', 'Geo Join ID', 'Geo Place Name', 'Time Period','Start_Date']
                                     )], 
                                    remainder = 'passthrough')
X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# data scaling
scaler_x = StandardScaler(with_mean=False)
X_train = scaler_x.fit_transform(X_train) # first time fit_transform then transform
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler()
y_train  = scaler_y.fit_transform(y_train)

def randomForest(n_estimators):
    # Training the Random Forest Regression model on the Training set
    model_rf = RandomForestRegressor(n_estimators=n_estimators)
    model_rf.fit(X_train, y_train.ravel())
    
    # Predicting the Test set results
    y_pred_rf = scaler_y.inverse_transform(model_rf.predict(X_test).reshape(-1,1))
    
    # Regression metrics
    mae=mean_absolute_error(y_test, y_pred_rf) 
    r2=r2_score(y_test, y_pred_rf)
    mea = mean_squared_error(y_test, y_pred_rf)
    rmse = np.sqrt(mea)
    
    return r2, mae, rmse, y_pred_rf

r2, mae, rmse, y_pred_rf = randomForest(100)
print (f'\nRandom Forest:\nr2 {r2}')
print (f'mae {mae}')
print (f'rmse {rmse}')