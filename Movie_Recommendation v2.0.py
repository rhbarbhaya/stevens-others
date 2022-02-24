# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 04:48:47 2018

@author: rhbar
"""
#importing dataset and setting defaults for plots
from IPython.display import Image, HTML
import json
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
try:
    from xgboost import XGBClassifier, XGBRegressor
    from wordcloud import WordCloud, STOPWORDS
    import sklearn
    import plotly
except ImportError:
    os.system("python -m pip install xgboost")
    os.system("python -m pip install wordcloud")
    os.system("python -m pip install sklearn")
    os.system("python -m pip install plotly")
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from wordcloud import WordCloud, STOPWORDS
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
plotly.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')

sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)

#importing dataset
data = pd.read_csv("movies_metadata.csv")
data = data.drop(["imdb_id"], axis = 1)
data["revenue"] = data["revenue"].replace(0, np.nan)
data['year'] = pd.to_datetime(data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
base_poster_url = 'http://image.tmdb.org/t/p/w185/'
data['poster_path'] = "<img src='" + base_poster_url + data['poster_path'] + "' style='height:100px;'>"
gross_top = data[['poster_path', 'title', 'budget', 'revenue', 'year']].sort_values('revenue', ascending=False).head(10)
pd.set_option('display.max_colwidth', 100)
HTML(gross_top.to_html(escape=False))
print (gross_top)