#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:15:53 2020

@author: nathan
"""

import pandas as pd
import random
from sklearn.utils import shuffle
from plotly.offline import plot

# rate of transmission
rt = 2

# state codes ...
trans = {'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR',
         'California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE',
         'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID',
         'Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
         'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
         'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
         'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE',
         'Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
         'New Mexico':'NM', 'New York':'NY','North Carolina':'NC',
         'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR',
         'Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
         'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT',
         'Vermont':'VT','Virginia':'VA','Washington':'WA',
         'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}

# data processing for populations
df = pd.read_csv('cc-est2018-alldata.csv', encoding='latin')
df = df[df['YEAR'] == 11]; df = df[df['AGEGRP'] == 0];

df = df[df['STNAME'].apply(lambda state: state in trans)]
df = df[['STNAME', 'TOT_POP']]
df['STNAME'] = df['STNAME'].apply(lambda x: trans[x])
df = df.groupby('STNAME', as_index=True).sum()
df //= 100
# sir model
populations = df['TOT_POP']
df.columns = ['s']
df['i'] = 0; df['r'] = 0

# add patient 0
state2 = random.choice(df.index)
df.loc[state2]['i'] = 1
df.loc[state2]['s'] -= 1

# simulation
periods = [df]
travel_chance = 1 / 8
while periods[-1]['i'].sum():
    print(periods[-1]['i'].sum())
    # current sir statistics
    df = periods[-1].copy()
    # shuffle to randomize travel
    shuffle(df)
    
    # spread infection
    for state1, sir in df.iterrows():
        # percent able to be infected
        ratio = sir.s / populations[state1]
        # add previously infected people to recovered
        sir['r'] += sir['i']
        # new infections
        i = 0
        for _ in range(int(sir['i'] * rt)):
            r = random.random()
            # if they infect someone in another state
            if r < travel_chance:
                state2 = random.choice(df.index) # this should be weighted
                if r < df.loc[state2]['s'] / populations[state2]:
                    df.loc[state2]['i'] += 1
                    df.loc[state2]['s'] -= 1
            # if they infect someone in their own state
            elif r < ratio:
                i += 1
                ratio = sir.s / populations[state1]
        
        # update sir statistics 
        sir['i'] = i
        sir['s'] -= i
        df.loc[state1] = sir
    
    periods.append(df)
    
zmax = max((df['i'] / df.sum(axis=1)).max() for df in periods)
data = [dict(type='choropleth',
             locations = df.index,
             z = (df['i'] + df['r']) / df.sum(axis=1),
             zmin=0.0, zmax=zmax,
             locationmode='USA-states') for df in periods]

# sliders
steps = []
for i in range(len(data)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data)],
                label=f't{i}')
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
layout = dict(geo=dict(scope='usa', projection={'type': 'albers usa'}), sliders=sliders)

fig = dict(data=data, layout=layout)
plot(fig)

# infections over time
df = pd.DataFrame({x:y for x, y in zip(periods[0].index, zip(*(tmp['i'].tolist() for tmp in periods)))})

# total
df = df.sum(axis=1)

# cumulative
df.cumsum()

# rt over time
df.pct_change().add(1)

# infections per population over time
df = pd.DataFrame([x['i'].tolist() / populations for x in periods], columns=periods[0].index)
df.reset_index(inplace=True)

