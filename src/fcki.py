# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:13:30 2020

@author: Mahmoud M. Ismail
"""

from __future__ import division
from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
from pytesmo.metrics import rmsd, nrmsd, aad
from math import sqrt
import numpy as np
import random
import time
from fancyimpute import IterativeImputer
from openpyxl import load_workbook
from joblib import Parallel, delayed

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    return np.linalg.norm(row1 - row2)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = np.linalg.norm(train - test_row, axis=1)
    indices = np.argsort(distances)
    return train[indices[:num_neighbors]]

# Function to impute missing values
def impute_missing_values(data):
    imputer = IterativeImputer(max_iter=10, tol=1e-2, random_state=0)
    return imputer.fit_transform(data)

# FCKI function
def FCKI(df_incomplete_data, generator):
    Xmis = df_incomplete_data[df_incomplete_data.isnull().any(axis=1)]
    while not Xmis.empty:
        xi = Xmis.iloc[0]
        index_of_xi = Xmis.index[0]
        col_name_removed = df_incomplete_data.columns[pd.isnull(xi)]
        obsData_without_missing_cols_in_xi = df_incomplete_data.dropna(subset=col_name_removed)
        Pt = pd.concat([obsData_without_missing_cols_in_xi, xi.to_frame().T])
        St = Pt.copy()
        St_Complete_Temp = St.copy()
        random_missing_col = generator.choice([col for col in range(len(St.columns)) if not pd.isnull(St.iloc[-1, col])])
        AV = St_Complete_Temp.iloc[-1, random_missing_col]
        St.iloc[-1, random_missing_col] = np.NaN
        
        mp = len(Pt.index)
        K_List, RMSE_List = [], []
        for k_count in range(2, min(mp, 20)):
            K_List.append(k_count)
            neighbors_for_xi_in_St = get_neighbors(St.iloc[:-1].values, St.iloc[-1].values, k_count)
            df_neighbors_for_xi_in_St = pd.DataFrame(neighbors_for_xi_in_St, columns=St.columns)
            imputed_value = df_neighbors_for_xi_in_St.iloc[:, random_missing_col].mean()
            actual_value = St_Complete_Temp.iloc[-1, random_missing_col]
            average_RMSE_All_Cols_Final = np.sqrt((imputed_value - actual_value) ** 2)
            RMSE_List.append(average_RMSE_All_Cols_Final)
            if len(RMSE_List) > 1 and RMSE_List[-1] >= RMSE_List[-2]:
                break
        
        k_of_min_RMSE = K_List[RMSE_List.index(min(RMSE_List))]
        neighbors_αi = get_neighbors(Pt.iloc[:-1].values, xi.values, k_of_min_RMSE)
        df_neighbors_αi = pd.DataFrame(data=neighbors_αi, columns=Pt.columns)
        Θi = pd.concat([df_neighbors_αi, xi.to_frame().T])
        Θi_filled_EM = impute_missing_values(Θi.values)
        xi_imputed = Θi_filled_EM[-1]
        xi_imputed_with_index = pd.Series(xi_imputed, index=Pt.columns).rename(index_of_xi)
        
        df_incomplete_data.loc[index_of_xi] = xi_imputed_with_index
        Xmis = Xmis.iloc[1:]

    return df_incomplete_data



def MultiIndexFCKI(df_incomplete_data, generator):
    Xmis = df_incomplete_data[df_incomplete_data.isnull().any(axis=1)]
    while not Xmis.empty:
        xi = Xmis.iloc[0]
        index_of_xi = Xmis.index[0]
        col_name_removed = df_incomplete_data.columns[pd.isnull(xi)]
        obsData_without_missing_cols_in_xi = df_incomplete_data.dropna(subset=col_name_removed)
        Pt = obsData_without_missing_cols_in_xi.append(xi)
        St = Pt.copy()
        St_Complete_Temp = St.copy()
        
        random_missing_col_idx = generator.choice([col for col in range(len(St.columns)) if not pd.isnull(St.iloc[-1, col])])
        AV = St_Complete_Temp.iloc[-1, random_missing_col_idx]
        St.iloc[-1, random_missing_col_idx] = np.NaN
        
        mp = len(Pt.index)
        K_List, RMSE_List = [], []
        for k_count in range(2, min(mp, 20)):
            K_List.append(k_count)
            neighbors_for_xi_in_St = get_neighbors(St.iloc[:-1].values, St.iloc[-1].values, k_count)
            df_neighbors_for_xi_in_St = pd.DataFrame(neighbors_for_xi_in_St, columns=St.columns)
            imputed_value = df_neighbors_for_xi_in_St.iloc[:, random_missing_col_idx].mean()
            actual_value = St_Complete_Temp.iloc[-1, random_missing_col_idx]
            average_RMSE_All_Cols_Final = np.sqrt((imputed_value - actual_value) ** 2)
            RMSE_List.append(average_RMSE_All_Cols_Final)
            if len(RMSE_List) > 1 and RMSE_List[-1] >= RMSE_List[-2]:
                break
        
        k_of_min_RMSE = K_List[RMSE_List.index(min(RMSE_List))]
        neighbors_αi = get_neighbors(Pt.iloc[:-1].values, xi.values, k_of_min_RMSE)
        df_neighbors_αi = pd.DataFrame(data=neighbors_αi, columns=Pt.columns)
        Θi = df_neighbors_αi.append(xi)
        Θi_filled_EM = impute_missing_values(Θi.values)
        xi_imputed = Θi_filled_EM[-1]
        xi_imputed_with_index = pd.Series(xi_imputed, index=Pt.columns).rename(index_of_xi)
        
        df_incomplete_data.loc[index_of_xi] = xi_imputed_with_index
        Xmis = Xmis.iloc[1:]

    return df_incomplete_data