from __future__ import division
import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
resolved = proj_path.resolve()
if resolved not in sys.path: sys.path.append(str(resolved))

import numpy as np
import pandas as pd
import tempfile
import joblib
import os
import time
from joblib import Parallel, delayed
from fcmeans import FCM
from sklearn.decomposition import PCA
from pytesmo.metrics import rmsd, nrmsd, aad
from fancyimpute import IterativeImputer, KNN

from src.utils import get_commons
from src.datasets import split_df
from src.fcki import FCKI, impute_missing_values

print("STARTING")
tmp_folder = str(proj_path / 'tmp')
os.environ['JOBLIB_TEMP_FOLDER'] = tmp_folder
joblib_temp_dir = tempfile.mkdtemp(dir=tmp_folder)
joblib.Parallel(temp_folder=joblib_temp_dir)

def main():
    start_time = time.time()

    print("MAIN")
    paths, constants, config, logger, device = get_commons()
    rng = np.random.default_rng(seed=config["seed"])

    logger.info('START')

    bigfive_s_cols = [('TARGET', col) for col in constants['bigfive_s_columns']]
    mbti_cols = [('TARGET', col) for col in constants['mbti_columns']]
    stats_cols = [('STATS', col) for col in constants['stats_columns']]
    columns = bigfive_s_cols + mbti_cols + stats_cols
    logger.info(columns)

    data = pd.read_csv(paths["new"]["w-emoji"], index_col=0, header=[0, 1])[columns]

    chosen_indices = rng.choice(data.index, size=10000, replace=False)
    logger.info(f'Data before random: {data.shape}')
    # Select rows based on these indices
    data = data.loc[chosen_indices]
    logger.info(f'Data shape after random: {data.shape}')
    
    df_incomplete_data = data.dropna(subset=columns, how='all')
    df_complete_data = data.dropna(subset=columns, how='any')
    logger.info(f'Shape all vals: {df_complete_data.shape}')

    num_of_clusters = config['imputation']['num_of_clusters']

    logger.debug(columns)
    logger.debug(df_incomplete_data.columns)

    df_incomplete_data_imputed = pd.DataFrame(impute_missing_values(df_incomplete_data), columns=df_incomplete_data.columns, index=df_incomplete_data.index)
    logger.info('Imputed first values')
    logger.info('Performing PCA..')
    
    pca = PCA(n_components=config['imputation']['n_components'])
    transformed = pd.DataFrame(pca.fit_transform(df_incomplete_data_imputed), index=df_incomplete_data.index)

    logger.info('Performing FCA..')
    fcm = FCM(n_clusters=num_of_clusters)
    fcm.fit(transformed.values)

    fcm_labels = fcm.u.argmax(axis=1)
    logger.debug(f'FCM labels length: {len(fcm_labels)}, Transformed shape: {transformed.shape}')

    df_incomplete_data['cluster'] = fcm_labels
    common_indices = df_complete_data.index.intersection(df_incomplete_data.index)
    df_complete_data = df_complete_data.loc[common_indices]
    df_complete_data['cluster'] = df_incomplete_data.loc[common_indices, 'cluster']

    clusters_complete = [df_complete_data[df_complete_data['cluster'] == i].drop(columns='cluster') for i in range(num_of_clusters)]
    clusters_incomplete = [df_incomplete_data[df_incomplete_data['cluster'] == i].drop(columns='cluster') for i in range(num_of_clusters)]

    logger.info('Performing FCKI..')
    results = Parallel(n_jobs=-1)(delayed(FCKI)(clusters_incomplete[i], rng) for i in range(num_of_clusters))
    all_clusters = pd.concat(results)

    all_clusters = all_clusters.loc[~all_clusters.index.duplicated(keep='last')]
    all_clusters.sort_index(inplace=True)
    all_dataset_imputed = all_clusters

    logger.info('FCKI completed')
    timeFinal = (time.time() - start_time)
    print(f"--- {timeFinal} seconds ---")

    all_dataset_imputed.to_csv(paths['filled']['imputed'], index=False)

    logger.info('FINISHED')


def multiindex_org(data:pd.DataFrame, constants:dict) -> pd.DataFrame:
    stat_list = [('STATS', stat) for stat in constants['stats_columns']]
    target_list = [('ORG_TARGET', target) for target in constants["target_columns"]]
    tuples =  stat_list + target_list
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['GROUP', 'FEATURE'])
    data_multiindexed = pd.DataFrame(columns=multiindex)
    data_multiindexed['STATS'] = data[constants['stats_columns']]
    data_multiindexed['ORG_TARGET'] = data[constants['target_columns']]
    return data_multiindexed

def multiindex_imp(data:pd.DataFrame, constants:dict) -> pd.DataFrame:
    target_list = [('IMP_TARGET', target) for target in constants["target_columns"]]
    multiindex = pd.MultiIndex.from_tuples(target_list, names=['GROUP', 'FEATURE'])
    data_multiindexed = pd.DataFrame(columns=multiindex)
    data_multiindexed['IMP_TARGET'] = data[constants['target_columns']]
    return data_multiindexed

def main2():
    start_time = time.time()

    print("MAIN")
    paths, constants, config, logger, device = get_commons()
    rng = np.random.default_rng(seed=config["seed"])

    logger.info('START')

    bigfive_s_cols = [('TARGET', col) for col in constants['bigfive_s_columns']]
    mbti_cols = [('TARGET', col) for col in constants['mbti_columns']]
    stats_cols = [('STATS', col) for col in constants['stats_columns']]
    columns = bigfive_s_cols + mbti_cols + stats_cols
    logger.info(columns)

    data = pd.read_csv(paths["new"]["w-emoji"], index_col=0, header=[0, 1])
    org_columns = data.columns
    data.columns = data.columns.droplevel(0)
    data = data.drop('TEXT', axis='columns').groupby('AUTHOR').mean()
    # target_cols = constants['mbti_columns'] + constants['bigfive_s_columns']
    # target_cols = constants["target_columns"]
    new_cols = data.columns
    new_idx = data.index

    imputer = IterativeImputer(random_state=config["seed"]) # KNN()
    imputed = imputer.fit_transform(data)
    imputed = pd.DataFrame(data=imputed, index=new_idx, columns=new_cols)
    logger.info(imputed.head(10))

    # chosen_indices = rng.choice(data.index, size=10000, replace=False)
    # logger.info(f'Data before random: {data.shape}')
    # data = data.loc[chosen_indices]
    # logger.info(f'Data shape after random: {data.shape}')
    
    #logger.debug(columns)
    # logger.debug(df_incomplete_data.columns)

    for c_col in constants['bigfive_c_columns']:
        print(f'c_col: {c_col}')
        s_col = 's' + c_col[1:]
        print(f's_col: {s_col}')
        imputed[c_col] = np.where(imputed[s_col] >= 50, 1, 0)
        print(imputed[c_col])

    for mbti_col in constants['mbti_columns']:
        imputed[mbti_col] = np.where(imputed[mbti_col] >= 0.5, 1, 0)
        print(imputed[mbti_col])
    
    binary_cols = constants['bigfive_c_columns'] + constants['mbti_columns']
    imputed[binary_cols] = imputed[binary_cols].clip(lower=0, upper=1, axis=1).round().astype(int)
    # data[binary_cols] = data[binary_cols].astype(int)

    logger.info('Setting multiindex..')
    imputed = multiindex_imp(imputed, constants)
    data = multiindex_org(data, constants)
    data =  data.join(imputed)

    logger.info('Imputation completed')
    timeFinal = (time.time() - start_time)
    print(f"--- {timeFinal} seconds ---")

    assert imputed["IMP_TARGET"].isnull().sum(axis=1).sum() == 0

    data.to_csv(paths['new']['imputed-all'])

    for task in constants["tasks"]:
        others = [x for x in constants['tasks'] if x!= task]
        others = [y for x in others for y in constants['columns'][x]]
        others_col = [(over, x) for x in others for over in ('IMP_TARGET', 'ORG_TARGET')]
        task_df = data.drop(others_col, axis=1)
        task_df.to_csv(paths['new']['split'][task])

    logger.info('FINISHED')
    print('FINISHED')
    print(data.head())


if __name__ == '__main__':
    main2()
