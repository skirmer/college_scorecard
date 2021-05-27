from flask import Flask, redirect, url_for, render_template, request, flash
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
import plotly
import plotly.graph_objects as go

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

def plotly_hist(df, prediction=1, degreetype='All', majorfield = "All Fields"):

    plot_series1 = df[df.CREDDESC == degreetype]['EARN_MDN_HI_2YR'].astype(int)
    plot_series2 = df[(df.CIPDESC_new == majorfield) & (df.CREDDESC == degreetype)]['EARN_MDN_HI_2YR'].astype(int)
               
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=plot_series1, name = "All Fields",histnorm='percent', xbins=dict(size = 1000)))
    fig.add_trace(go.Histogram(x=plot_series2, name = majorfield, histnorm='percent', xbins=dict(size = 1000)))
    
    fig.update_layout(
        barmode='overlay',
        title_text=f'Median Earnings, {degreetype}', # title of plot
        xaxis_title_text='USD ($)', # xaxis label
        yaxis_title_text='Percent of Total', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1, # gap between bars of the same location coordinates
        template = "plotly_white",
        font=dict(
            family="Helvetica",
            size=12
        ),
        legend=dict(yanchor = 'top', y=1, 
                    xanchor = 'right', x = 1)
    )
    fig.add_vline(x=prediction, line_dash="dash", annotation_text=f"Predicted: ${round(prediction, 2):,}")

    fig.update_traces(opacity=0.55)
    return fig
    

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

@app.route('/result')
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

    p3 = plotly_hist(df=df, degreetype = variables['CREDDESC'], prediction=pred_final, majorfield=variables['CIPDESC_new'])
    graphJSON = json.dumps(p3, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'result.html', 
        locale = variables['LOCALE'], 
        collegetype = variables['CONTROL'], 
        cip = variables['CIPDESC_new'],  
        cred = variables['CREDDESC'], 
        pred_final = round(pred_final, 2), 
        region=variables['REGION'], 
        sat=variables['SAT_AVG_ALL'], 
        tuit = variables['tuition'], 
        adm = variables['ADM_RATE_ALL'], 
        graphJSON=graphJSON
   )

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8000)