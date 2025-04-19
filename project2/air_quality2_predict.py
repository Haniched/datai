import pandas as pd
from keras.models import load_model #load model
import pickle #load encoder

# load model
model = load_model('air_quality2-model.keras')

# load encoder
with open('air_quality2-ct.pickle', 'rb') as f:
    ct = pickle.load(f)
    
# load scalers
with open('air_quality2-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)

with open('air_quality2-scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

# ennusta datalla
Xnew = pd.read_csv('Air_Quality_new.csv')
Xnew_org = Xnew
Xnew['Start_Date'] = pd.to_datetime(Xnew['Start_Date'])
Xnew = ct.transform(Xnew)
Xnew = scaler_x.transform(Xnew)
ynew = model.predict(Xnew) 
ynew = scaler_y.inverse_transform(ynew)

# get scaled value back to unscaled
Xnew = scaler_x.inverse_transform(Xnew)

ynew = pd.DataFrame(ynew).reindex()
ynew.columns = ['Predicted Data Value']
df_results = Xnew_org.join(ynew)

# tallennetaan ennusteet csv-tiedostoon
df_results.to_csv('air_quality2-predicted.csv', index=False)
