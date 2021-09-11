#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

ds_PF_ZT = pd.read_csv('TE/csv/PF_ZT.csv')
X = ds_PF_ZT.iloc[:, 0:3]
y = ds_PF_ZT.iloc[:, -1]
ds_PF_ZT


## pip install pymatgen
## pip install matminer

from pymatgen.core.composition import Composition
ds_PF_ZT['Formula']
Comp = []
for value in ds_PF_ZT['Formula']:
  Comp.append(Composition(value))
Comp

ds_PF_ZT['Composition'] = Comp
ds_PF_ZT

from matminer.featurizers.composition import ElementFraction
ef = ElementFraction()

ds_PF_ZT = ef.featurize_dataframe(ds_PF_ZT,'Composition')
ds_PF_ZT

ds_PF_ZT = ds_PF_ZT.loc[:, (ds_PF_ZT != 0).any(axis=0)]
ds_PF_ZT

ds_PF_ZT.columns
ds_PF_ZT = ds_PF_ZT.reindex(columns = ['Formula', 'Composition', 'temperature', 'power_factor',
       'Li', 'B', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 'Cr',
       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Se', 'Sr', 'Y', 'Zr',
       'Nb', 'Mo', 'Ru', 'Rh', 'Ag', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba',
       'La', 'Ce', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Yb', 'Hf', 'W', 'Au',
       'Tl', 'Pb', 'Bi', 'ZT'])

ds_PF_ZT

X_ML = ds_PF_ZT.iloc[:, 2: 55]
y_ML = ds_PF_ZT.iloc[:, -1]

corrMatrix = ds_PF_ZT.corr()
corrMatrix

import seaborn as sns
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(ds_PF_ZT.corr()[['ZT']].sort_values(by='ZT', ascending=False), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
heatmap.set_title('Features Correlating with ZT', fontdict={'fontsize':18}, pad=16)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ML, y_ML, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 50, random_state = 0)
rf.fit(X_train, y_train)

rf.feature_importances_

feature_importances = pd.DataFrame(X_ML.columns, rf.feature_importances_, columns=['importance']).sort_values('importance', ascending=True)
print(feature_importances)

figure(figsize=(10,10), dpi=100)
plt.barh(X_ML.columns,rf.feature_importances_)

import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
print(shap_values)

shap.summary_plot(shap_values, X_test, plot_type="bar")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
print(shap_values)

shap.summary_plot(shap_values, X_train, plot_type="bar")

