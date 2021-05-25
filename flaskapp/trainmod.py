from flask import Flask, redirect, url_for, render_template, request, flash
from numpy.core.numeric import cross
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
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import Span
from bokeh.models import NumeralTickFormatter



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

def plot_hist(df, prediction=1):
    hist, edges = np.histogram(df['EARN_MDN_HI_2YR'].astype(int), bins = 40)

    p = figure(
        title="Histogram of Earnings, All Fields",
        y_axis_label="Count",
        x_axis_label="Earnings",
        width=750,
        height=300,
    )

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    vline = Span(location=prediction, dimension='height', line_color='red', line_width=3)
    p.renderers.extend([vline])
    p.xaxis[0].formatter = NumeralTickFormatter(format="$%.0f")

    return p


def plot_groupdist(df, prediction=1, majorfield = 'Accounting and Related Services.'):  
    hist, edges = np.histogram(df[df.CIPDESC_new == majorfield]['EARN_MDN_HI_2YR'].astype(int), bins = 40)

    p = figure(
        title=f"Histogram of Earnings, {majorfield}",
        y_axis_label="Count",
        x_axis_label="Earnings",
        width=750,
        height=300,
    )

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    vline = Span(location=prediction, dimension='height', line_color='red', line_width=3)
    p.renderers.extend([vline])
    p.xaxis[0].formatter = NumeralTickFormatter(format="$%.0f")

    return p
    

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
def result(modobj=modobj, df=df):

    loc_val = request.args.get('locale', '')
    cip_val = request.args.get('cip', '')
    cred_val = request.args.get('cred', '')
    reg_val = request.args.get('region', '')
    adm_val = request.args.get('adm', '', type= float)
    tuit_val = request.args.get('tuit', '', type= int)
    collegetype_val = request.args.get('coltype', '')
    sat_val = request.args.get('sat_number', '', type= int)

    sat= sat_val
    cred=cred_val
    cip=cip_val
    col=collegetype_val
    reg=reg_val
    tuit=tuit_val
    loc=loc_val
    adm=adm_val

    newdf = pd.DataFrame([[sat, cred, cip, col, reg, tuit, loc, adm]], 
        columns = ['SAT_AVG_ALL','CREDDESC', 'CIPDESC_new','CONTROL', 'REGION', 'tuition', 'LOCALE', 'ADM_RATE_ALL'])

    [[prediction]] = modobj.predict(newdf)
    pred_final = prediction if prediction > 0 else np.nan

    p1 = plot_hist(df=df, prediction=pred_final)
    p2 = plot_groupdist(df=df, prediction=pred_final, majorfield=cip)
    script, div = components(p1)
    script2, div2 = components(p2)

    return render_template(
        'result.html', 
        locale = loc_val, 
        collegetype = collegetype_val, 
        cip = cip_val, 
        cred = cred_val, 
        pred_final = round(pred_final, 2), 
        region=reg_val,
        sat=sat_val, 
        tuit = tuit_val,
        adm = adm_val,
        script=script, div=div,
        script2=script2, div2=div2
   )