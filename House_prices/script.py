#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:39:44 2020

@author: Pedro_Martínez
"""
# Load  plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.metrics import mean_squared_error as MSE

import sys
sys.path.append('/Users/macbookpro/Desktop/Python_excercises/')

from Package_container.Utilities import CSV
from Package_container.Modelo import Regressor

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


House = Regressor()

train = CSV('train.csv')
#test = CSV('test.csv')
df_raw = train._read_csv('train.csv')
df_raw_test = train._read_csv('test.csv')

#cols_nan = sorted(train.find_nan(df_raw))
#col_ninstances = {feature: Counter(df_raw[feature]) for feature in df_raw}



#
#def lot_area(df_raw):
#    lot_dict = dict(zip(df_raw['LotConfig'].drop_duplicates(), [0.8, 1.5, 1, 1.2, 1.7]))
#    lot_col = df_raw['LotConfig'].map(lot_dict)
#
#    alley_dict = dict(zip(df_raw['Alley'].drop_duplicates(), [1, 1.1, 1.2]))
#    alley_col = df_raw['Alley'].map(alley_dict)
#    
#    LotShapedict =dict(zip(df_raw['LotShape'].drop_duplicates(), [1, 0.9, 0.8, 0.7]))
#    LotCol = df_raw['LotShape'].map(LotShapedict)
#
#
#    tot_area = (df_raw['1stFlrSF']+0.9*df_raw['2ndFlrSF']+1.1*df_raw['3SsnPorch']+1.5*df_raw['PoolArea']+0.7*df_raw['LotArea']*lot_col + 1.2*df_raw['OpenPorchSF']+ 1.2*df_raw['EnclosedPorch']+ 1.2*df_raw['ScreenPorch']+ 1.2*df_raw['LotArea']+0.5*df_raw['LotFrontage'])*alley_col*LotCol
#
#    df_new = df_raw.drop(['LotConfig', '1stFlrSF', '2ndFlrSF', '3SsnPorch', 'PoolArea', 'Alley','OpenPorchSF', 'LotArea', 'LotFrontage', 'LotShape','EnclosedPorch','ScreenPorch'], axis=1)
#    df_new['Tot_area'] = tot_area
#    return df_new
#
#
#df_lot = lot_area(df_raw)
#df_lot_test = lot_area(df_raw_test)



def matcher(df,pattern):
    cols = [f for f in df if (pattern in f.lower())]
    
    df1 = df.drop(cols, axis=1)
    df_new = df[cols]

#    df_gar_cat = House.order_category(df_gar)
    df_new_cat = pd.get_dummies(df_new, drop_first=True)    
#    df_new_cat = df_new_cat.fillna(0)

    
    scaler = Normalizer()
    X = df_new_cat

    # fit the classifier on the training dataset
    X = scaler.fit_transform(X)
    pca = PCA(1)
    pca.fit(X)
#    print(pca.n_components_)
    X = pd.DataFrame(pca.transform(X),columns=['Trans_'+pattern])
    X = pd.concat([df1, X], axis=1)
    return X



lista = [ 'roof', 'exter', 'overa', 'bath',  'kitchen', 'fire', 'heat', 'pool','condition', 'msz','year','bsmt', 'elect','land','misc','type','pave', 'hood','funct','fence','house', 'found',]

considerarluego=['utilities','bed','class','street','air','alley','garage','area','lot']
#garage,area'lot'
dropear=[]

for col in df_raw.columns:
    for elem in considerarluego:
        if elem in col.lower():
            dropear.append(col)
        


df = df_raw.drop(dropear,axis=1)
df_test = df_raw_test.drop(dropear,axis=1)

for pattern in lista:
    df=matcher(df,pattern)
    df_test=matcher(df_test,pattern)

#

def model(X, xtest):
#    X1 = X.drop(["Id"],axis=1)
#    xtest = xtest.drop("Id", axis=1)
    
#    X1 = House.standarize(X1)
#    xtest = House.standarize(xtest)
#    
    X_train, X_test, y_train, y_test = House.train_test( X, 'SalePrice')
#    X_train, X_test = House.select_inputs(X_train, X_test, y_train, 3)
    columnsSpecial = X_train.columns
    eval_set = [(X_test, y_test)]


    House.fit(X_train, y_train, 'rmse', eval_set)
    y_pred_test = House.predict(df_test[columnsSpecial])  
    
#    House.fit_grid(X_train, y_train, 'rmse', eval_set)
#    y_pred_test_grid = House.predict_grid(df_test[columnsSpecial])
#    
    
#    Evaluate Metrics
    y_pred_test_grid=[]
    y_pred_train = House.predict(X_train)
    y_pred_tt = House.predict(X_test)

#    y_pred_test=[]
#    y_pred_train = House.predict_grid(X_train)
#    y_pred_tt = House.predict_grid(X_test)


    House._error('train', y_train, y_pred_train)
    House._error('test', y_test, y_pred_tt)

    
    return y_pred_test,y_pred_test_grid

#colnan=train.find_nan(df)
    
list_drop=[f for f in df.columns if 'trans' in f.lower()]
list_drop.append('Id') 

idd=df_test['Id']
target=df['SalePrice']

df_trans=[df[f] for f in df.columns if 'trans' in f.lower()]
df_trans=pd.DataFrame(df_trans).T

l1=list_drop.copy()
l1.append('SalePrice')

df=House.intervals(df.drop(l1,axis=1),6)

df=df.fillna(0)
df=pd.concat([df,df_trans],axis=1)


df_test_trans=[df_test[f] for f in df.columns if 'trans' in f.lower()]
df_test_trans=pd.DataFrame(df_test_trans).T


df_test=House.intervals(df_test.drop(list_drop,axis=1),6)
df_test=df_test.fillna(0)
df_test=pd.concat([df_test,df_test_trans],axis=1)


predictions, pred_grid = model(df, df_test)

final_df = pd.concat([df_test['Id'].astype(int), pd.DataFrame(pred_grid,columns=['SalePrice'])],axis=1)
final_df=final_df.set_index('Id')
final_df.to_csv('HouseSubmission')





#df_clean_train = calculate_df_clean('train.csv')
#df_clean_test = calculate_df_clean('test.csv')

#muertos = model(df_clean_train, df_clean_test)
#final_df = pd.DataFrame(df_clean_test['PassengerId'])
#final_df['Survived'] = muertos
#final_df.to_csv('House Submission', index=None)
