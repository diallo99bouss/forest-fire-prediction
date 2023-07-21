import xgboost as xgb
import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
from mongodb import mongodbconnection
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Data retrieved from DB using mongoconnection module
dbcon = mongodbconnection(username='mongodb', password='12345')
list_cursor = dbcon.getdata(dbName='FireDataML', collectionName='ml_task')
log.info('Connected to Mongodb and data retrieved')

# Data From MongoDB is used for Standardization
df = pd.DataFrame(list_cursor)
df.drop('_id', axis=1, inplace=True)
log.info('DataFrame created')
scaler = StandardScaler()
X = df.drop(['FWI', 'Classes'], axis=1)
# Standardize
X_reg_scaled = scaler.fit_transform(X)
log.info('Standardization done')

# Réentraînez le modèle XGBoost avec de nouvelles données
# Remplacez ces lignes par le code pour réentraîner vos modèles
model_C = xgb.XGBClassifier()  # Remplacez par le modèle que vous souhaitez entraîner
model_C.fit(X_reg_scaled, df['Classes'])

model_R = xgb.XGBRegressor()  # Remplacez par le modèle que vous souhaitez entraîner
model_R.fit(X_reg_scaled, df['FWI'])

# Sauvegardez le modèle avec la méthode spécifique de XGBoost
model_C.save_model('model/classification.json')
model_R.save_model('model/regression.json')

# Reste du code de votre application Flask continue ici...

# Le reste de votre code pour l'application Flask continue ici
# ...
# Data retrieved from DB using mongoconnection module
dbcon = mongodbconnection(username='mongodb', password='12345')
list_cursor = dbcon.getdata(dbName='FireDataML', collectionName='ml_task')
log.info('Connected to Mongodb and data retrieved')

# Data From MongoDB is used for Standardization
df = pd.DataFrame(list_cursor)
df.drop('_id', axis=1, inplace=True)
log.info('DataFrame created')
scaler = StandardScaler()
X = df.drop(['FWI', 'Classes'], axis=1)
# Standardize
X_reg_scaled = scaler.fit_transform(X)
log.info('Standardization done')


# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        log.info('Input from Api testing', data)
        new_data = [list(data.values())]
        final_data = scaler.transform(new_data)
        output = int(model_C.predict(final_data)[0])
        if output == 1:
            text = 'The Forest in Danger'
        else:
            text = 'Forest is Safe'
        return jsonify(text, output)
    except Exception as e:
        output = 'Check the in input again!'
        log.error('error in input from Postman', e)
        return jsonify(output)


# Route for Classification Model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        final_features = scaler.transform(final_features)
        output = model_C.predict(final_features)[0]
        log.info('Prediction done for Classification model')
        if output == 0:
            text = 'Forest is Safe!'
        else:
            text = 'Forest is in Danger!'
        return render_template('index.html', prediction_text1="{} --- Chance of Fire is {}".format(text, output))
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('index.html', prediction_text1="Check the Input again!!!")


# Route for Regression Model
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        data = [float(x) for x in request.form.values()]
        data = [np.array(data)]
        data = scaler.transform(data)
        output = model_R.predict(data)[0]
        log.info('Prediction done for Regression model')
        if output > 15:
            return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(output))
        else:
            return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(output))
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('index.html', prediction_text2="Check the Input again!!!")


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=False)
