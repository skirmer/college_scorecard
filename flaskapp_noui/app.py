from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import pandas as pd
import numpy as np
import s3fs
import crosswalks
import numpy as np
import json

def load_data():
    s3 = s3fs.S3FileSystem(anon=True)
    s3fpath = 's3://saturn-public-data/college-scorecard/cleaned_merged.csv'

    major = pd.read_csv(
        s3fpath,
        storage_options={'anon': True},
        dtype = 'object',
        na_values = ['PrivacySuppressed']
    )
    return major

def split_data(df):
    
    X = df[['SAT_AVG_ALL','CREDDESC', 'CIPDESC_new',
            'CONTROL', 'REGION', 'tuition', 'LOCALE', 'ADM_RATE_ALL']]
    y = df[['EARN_MDN_HI_2YR']]
    
    return [X, y]

def trainmodel(X, y):
    enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
    imp = IterativeImputer(max_iter=10, random_state=0, initial_strategy='mean', add_indicator = True)
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    ct = ColumnTransformer(
        [('onehot', enc, ['CONTROL','CREDDESC', 'CIPDESC_new','REGION', 'LOCALE']),
        ('impute', imp, ['SAT_AVG_ALL', 'ADM_RATE_ALL'])], 
        remainder='passthrough' 
    )

    pipe = Pipeline(steps=[('coltrans', ct), ('linear', linear_model.LinearRegression())])
    pipe = pipe.fit(X_train, y_train) 
    
    return pipe, pipe.score(X_test,y_test)


def train_model():
    df = load_data()
    X, y = split_data(df)

    modobj, modscore = trainmodel(X, y)
    return modobj, modscore, df

app = Flask(__name__)
modobj, modscore, df = train_model()
creds = sorted(list(df['CREDDESC'].unique()), reverse = True)
cips = sorted(list(df['CIPDESC_new'].unique()), reverse = True)
crosswalks.coltype = sorted(crosswalks.coltype)
crosswalks.reg = sorted(crosswalks.reg, reverse = True)
crosswalks.locs = sorted(crosswalks.locs, reverse = True)


@app.route("/")
def index():
    # Main page
    return render_template('index.html', 
    modscore = round(modscore,3), 
    col_list = crosswalks.coltype, 
    reg_dict = crosswalks.reg,
    locale_dict = crosswalks.locs, 
    cip_list = cips,
    cred_list = creds,
    tuit_min = min(round(df['tuition'].astype(float), 1)),
    tuit_max = max(round(df['tuition'].astype(float), 1)),
    adm_min = min(df['ADM_RATE_ALL'].astype(float)),
    adm_max = max(df['ADM_RATE_ALL'].astype(float)))

@app.route('/api/v1/')
def result():

    variables = {'SAT_AVG_ALL':'sat',
                'CREDDESC':'cred',
                'CIPDESC_new':'cip', 
                'CONTROL':'coltype', 
                'REGION':'region',
                'tuition':'tuit',
                'LOCALE':'locale', 
                'ADM_RATE_ALL':'adm', 
        }
    for i in variables:
        if variables[i] in request.args:
            if variables[i]  == 'adm':
                variables[i] = request.args.get(variables[i], '', type= float)
            elif (variables[i] == 'tuit') | (variables[i] == 'sat'):
                variables[i] = request.args.get(variables[i], '', type= int)
            else:
                variables[i] = request.args.get(variables[i], '')
        else:
            return f"Error: No {variables[i]} field provided. Please specify {variables[i]}."

    newdf = pd.DataFrame([variables])

    [[prediction]] = modobj.predict(newdf)
    pred_final = prediction if prediction > 0 else np.nan

    variables['predicted_earn'] = pred_final
    return  jsonify(variables)

if __name__ == "__main__":
  app.run()