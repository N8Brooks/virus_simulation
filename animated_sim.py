#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:49:33 2020

@author: nathan
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from random import choices
from celluloid import Camera
from scipy.spatial import distance_matrix

# initial rate of infections
r0 = 2
# inversely relatied to chance of travel
travel_chance = 8
# starting counties for patient 0
county_zero = [36047, 6037, 17031, 48201, 4013, 53033]
# processes for parallelization
processes = 20
# whether choropeth should use cumulative or current infections
cumulative = False

# In[] shapes of counties

# https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
df = gpd.read_file('cb_2018_us_county_20m/cb_2018_us_county_20m.shp')
df = df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')] # HI, AL
df = df[df['STATEFP'] != '72'] # Puerto Rico
df.index = df.STATEFP + df.COUNTYFP
df.drop('11001', inplace=True) # district of columbia
df = df[['geometry']]

# In[] populations of counties

tmp = pd.read_csv(('https://www2.census.gov/programs-surveys/popest/datasets/2'
    '010-2019/counties/totals/co-est2019-alldata.csv'), encoding='latin')
tmp.drop(tmp[tmp.STNAME == tmp.CTYNAME].index, inplace=True)
tmp.index = [f'{s:02}{c:03}' for s, c in zip(tmp.STATE, tmp.COUNTY)]
tmp.rename(columns={'POPESTIMATE2019':'population'}, inplace=True)
df = df.join(tmp['population'])
df.index = df.index.astype(int)

# In[] travel weights

# distance matrix
c = np.array([x.coords[0] for x in df['geometry'].centroid])
dist = pd.DataFrame(distance_matrix(c,c), index=df.index, columns=df.index)
np.fill_diagonal(dist.values, 0)

# population matrix
pops=pd.DataFrame([df['population']]*len(df),index=df.index,columns=df.index).T

# increase weight based on distance and inverse population
dist = pops / dist

# can't travel to self
np.fill_diagonal(dist.values, 0)

# In[] susceptible, infected, recovered

# time series
data_len = 0
data = pd.DataFrame()

# sir model
sir = df[['population']]
sir['sus'] = df['population']
sir['inf'] = 0
sir['rec'] = 0

# infecting patient 0
for cs in county_zero:
    sir.loc[cs, 'sus'] -= 1
    sir.loc[cs, 'inf'] += 1

# In[] running the simulation

@njit
def update_county(pop, sus0, inf):
    """updates row based on in county infections"""
    sus1 = sus0
    for _ in range(round(inf * r0)):
        if np.random.randint(pop) < sus1:
            sus1 -= 1
    return sus1, sus0 - sus1

def update_travel(k, dist):
    """swaps infected people within counties"""
    output = np.zeros_like(dist, dtype=np.uint32)
    for fips_to in choices(range(len(dist)), dist, k=k):
        output[fips_to] += 1
    return output

# progress bar with estimated cummulative infections and process pool
pbar = tqdm(total=int((1 - 1/r0) * df['population'].sum() + 0.5),\
    desc='Running Simulation')
pool = mp.Pool(processes)

# continue simulation until no one is infected
while sir['inf'].any():
    count = sir['inf'].sum()
    pbar.update(count)  # cummulative infections
    
    # adding population with >0 infections
    data=data.append(sir.set_index([[data_len]*len(sir),sir.index]),sort=False)
    data_len += 1
    
    # update recoveries
    sir['rec'] += sir['inf']
    
    # travel out of county
    travel = sir['inf'] // travel_chance
    sir['inf'] -= travel
    
    # traveling infections
    sir['inf'] += sum(pool.starmap(update_travel,\
       zip(travel.values, dist.values)))
    
    # in-county infections
    sir['sus'], sir['inf'] = zip(*pool.starmap(update_county,\
       sir[['population', 'sus', 'inf']].values))

# close progress bar and pool
pbar.close()
pool.close()

# adding population with 0 infections
data = data.append(sir.set_index([[data_len] * len(sir),sir.index]),sort=False)

# In[] infections per time period

"""
looks somewhat normal
peak is where it trasitions from epidemic to endemic
"""
title = 'Infections per Unit Time'
tmp = data['inf'].unstack().T.sum()
tmp.index = tmp.index * 4
ax = tmp.plot(title=title)
ax.get_figure().savefig(title.lower().replace(' ', '_'))

# In[] cumulative infections over time

"""
sort of s shaped
peak is about 77% of population for a r0 of 2.0
the formula for herd immunity would say (1 - 1/2.0) = 75%
"""
plt.clf()
title = 'Cumulative Infections over Time'
ax = tmp.cumsum().plot(title=title)
ax.get_figure().savefig(title.lower().replace(' ', '_'))

# In[] average infections created per infected individual (rt)

"""
starts at 2 (this is r0) and then works its way down to 0
this shows how herd immunity comes into play
"""

plt.clf()
title = 'Rate of Transmission over Time'
ax = tmp.pct_change().add(1).plot(title=title)
ax.get_figure().savefig(title.lower().replace(' ', '_'))

# In[] animating map

# current
data2 = (data['inf'] / data['population']).unstack()
data_len = data2.index.max()
if cumulative:
    data2 = data2.cumsum()
    for col in data2:
        data2[col] /= data2[col].loc[data_len - 1]
    data2.loc[data_len-1].min()
    vmax = 1.
else:
    vmax = data2.max(axis=1).describe()['75%']
cases = 0

fig = plt.figure(figsize=(16, 10))
ax = fig.gca()
ax.axis('off')
plt.tight_layout()
camera = Camera(fig)

"""
for i in trange(data_len, desc='Creating mp4'):
    cases += data['inf'].unstack().loc[i].sum()
    df['inf'] = data2.loc[i]
    df.plot(ax=ax, column='inf', vmin=0, vmax=vmax)
    ax.text(0.5, 1.01, f'Week {i/2:4.1f}: {cases:9,d} total cases', 
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=16)
    camera.snap()
"""

# create half cols
update = [data2.loc[0]]
for i in range(data_len):
    pre_row = data2.loc[i]
    pos_row = data2.loc[i + 1]
    update.extend([(3*pre_row + pos_row)/4, (pre_row + pos_row)/2, (pre_row + 3*pos_row)/4, pos_row])
data2 = pd.DataFrame(update).reset_index(drop=True)

for i, row in tqdm(data2.iterrows(), total=len(data2), desc='Creating mp4'):
    df['inf'] = row
    df.plot(ax=ax, column='inf', vmin=0, vmax=vmax)
    ax.text(0.5, 1.01, f'Day {i:3d}', 
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=16)
    camera.snap()

animation = camera.animate()
animation.save('pop_dist.mp4', fps=30)























