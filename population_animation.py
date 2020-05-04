#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:53:03 2020

@author: nathan
"""

from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.offline import plot

with urlopen(('https://raw.githubusercontent.com/plotly/datasets/master/geojso'
              'n-counties-fips.json')) as response:
    counties = json.load(response)

df = pd.read_csv(('https://www2.census.gov/programs-surveys/popest/datasets/20'
    '10-2018/counties/asrh/cc-est2018-alldata.csv'), encoding='latin')
df = df[df['YEAR'] == 11]; df = df[df['AGEGRP'] == 0]
df['FIPS'] = [f'{s:02}{c:03}' for s, c in zip(df.STATE, df.COUNTY)]
df = df[['FIPS', 'TOT_POP']]

df.sort_values(by='TOT_POP', inplace=True)
df.reset_index(inplace=True)

df['COLOR'] = [int(x) for x in np.arange(0, 256, 256/len(df))][:len(df)]

fig = px.choropleth(df, geojson=counties, locations='FIPS', color='COLOR',
                    scope="usa", labels={'TOT_POP':'population'})
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
plot(fig)