import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path: sys.path.append(str(proj_path))

import os
import argparse
import logging

import json
import optuna
import torch
import torchmetrics
import torcheval.metrics
from torch import nn
import random
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import LibMTL as mtl

import src.metrics
import src.loss
from src.utils import get_commons, get_paths
from src.datasets import SingleInputDataset, get_stl_dataloaders, get_mtl_dataloaders, split_df
from src.ffn import Encoder, Decoder
from src.trainer import MTLTrainer, train, get_optimizer, get_scheduler

import torchvision
import matplotlib.pyplot as plt
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

os.environ['TQDM_DEFAULT_DISABLE'] = '1'
os.environ['WDM_PROGRESS_BAR'] = '0'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    mtl.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

finals = { # final layer act func, size
  'mbti': ('sigmoid', [4]), 
  'bigfive_c': ('sigmoid', [5]),
  'bigfive_s': ('none', [5])
}

def normalize(df:pd.DataFrame, cols:str):
  scaler = StandardScaler()
  df[cols] = scaler.fit_transform(df[cols])
  return df
  #df[cols] = (df[cols]-df[cols].mean()) / df[cols].std()
  #return df


def unified(trial:optuna.trial.Trial, learning_method:str, subset:str):
  assert learning_method.lower() in ('stl', 'mtl')
  assert subset.lower() in ('org', 'imp', 'mix')
  paths, constants, config, logger, device = get_commons(config_name='opt')

  set_seed(config['seed'])
  rng = np.random.default_rng(config['seed'])
  tsb = lambda x: trial.suggest_categorical(name=x, choices=[True, False])
  tsc = trial.suggest_categorical
  add_norm = True # To prevent exploding gradients 
  defeat = (0, 0, 100)
  tasks = constants['tasks']

  # DataFrame 
  org_cols = [('ORG_TARGET', col) for task in constants["tasks"] for col in constants["columns"][task]]
  tar_cols = [('IMP_TARGET', col) for task in constants["tasks"] for col in constants["columns"][task]]
  mbti_cols = [(f'{tar}_TARGET', col) for col in constants['bigfive_s_columns'] for tar in ('ORG', 'IMP')]

  dataframe = pd.read_csv(paths['new']['imputed-all'], header=[0, 1], index_col=0)
  if tsb('normalize_stats'): dataframe = normalize(dataframe, 'STATS')
#  if tsc('normalize_targets', [True, False]): dataframe = normalize(dataframe, mbti_cols)
  
  # Embeddings
  selected_embs = {}  
  for model_name, emb_size in constants['embedding_sizes'].items():
    if tsb('use ' + model_name): 
      selected_embs[model_name] = emb_size
  if not selected_embs: return defeat

  # Drop stats
  for stat in constants['stats_columns']:
    if not tsb('use ' + stat): 
      dataframe = dataframe.drop(('STATS', stat), axis=1)
  if 'STATS' not in dataframe.columns.get_level_values(0): return defeat

  stats_size = len(dataframe['STATS'].columns)
  embeddings_size = sum(selected_embs.values())

  embedding_list = []
  for model_name in selected_embs.keys():
    emb = pd.read_csv(paths['new']['old_embeddings'][model_name], index_col=0)
    new_columns = pd.MultiIndex.from_product([['CLS'], emb.columns])
    new_emb = pd.DataFrame(emb.values, columns=new_columns, index=emb.index)
    embedding_list.append(new_emb)
  
  dataframe = pd.concat([dataframe] + embedding_list, axis=1, copy=False)
  print(dataframe.shape)

  finals = { # final layer act func, size
    'mbti': ('sigmoid', [4]), 
    'bigfive_c': ('sigmoid', [5]),
    'bigfive_s': ('none', [5])
  }


  # Dataloaders
  dl_args = {
    'df': dataframe,
    'train_size': config['split']['train'], 
    'test_size': config['split']['test'], 
    'config': config['dataloaders'], 
    'constants': constants, 
    'generator': rng, 
  }

  dataloaders = split_df(learning=learning_method, subset=subset, **dl_args)


  logger.info(f'STARTING {learning_method}')

  ### STL
  if learning_method == 'stl':

    stl_metrics = {
      'mbti': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True), # Metric object, higher_is_better
      'bigfive_c': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True),
      'bigfive_s': (torchmetrics.regression.MeanAbsoluteError(), False)
    }

    stl_loss_fns = {
      'mbti': nn.BCELoss(),
      'bigfive_c': nn.BCELoss(),
      'bigfive_s': nn.MSELoss()
    }

    stl_n_hidden = {task: trial.suggest_int(f'stl_{task}_layers', 1, 4) for task in tasks}
    stl_hidden = {task: [pow(2, trial.suggest_int(f'stl_{task}_power_layer_{i}', 4, 12)) for i in range(stl_n_hidden[task])] for task in tasks}
    stl_dropout = {task: trial.suggest_float(f'stl_{task}_dropout', 0.01, 0.80) for task in tasks}
    stl_input_nn = [embeddings_size + stats_size]

    stl_decoders = nn.ModuleDict({k: Decoder(stl_input_nn + stl_hidden[k] + v[1], dropout=stl_dropout[k], final=v[0], norm=add_norm).to(device) for k, v in finals.items()})

    stl_optimizers = {}
    stl_schedulers = {}
    for task in tasks:
      optim_param = {
        'optim': 'adamp',
        'lr': trial.suggest_float(f'stl_{task}_lr', 0.00001, 0.01),
        'weight_decay': trial.suggest_float(f'stl_{task}_weight_decay', 0.0001, 0.01), # Different for adam than adamw. AdamP recommends 0.01
        'betas': config['optim_param']['betas']
      }
      
      scheduler_param = {
        'scheduler': config['scheduler_param']['scheduler'],
        'step_size': trial.suggest_int(f'stl_{task}_step_size', 1, 25), # How many steps before decay. Default 1
        'gamma': trial.suggest_float(f'stl_{task}_gamma', 0.0001, 0.01) # Rate of lr decay. Default 0.1
      }

      stl_alg, stl_optim_arg = get_optimizer(optim_param)
      stl_optimizers[task] = stl_alg(stl_decoders[task].parameters(), **stl_optim_arg)
      stl_schedulers[task] = get_scheduler(stl_optimizers[task], scheduler_param)

    def train_stl(task:str, threshold) -> dict:
      result = train(
        stl_decoders[task], 
        dataloaders[task], 
        stl_optimizers[task], 
        stl_loss_fns[task], 
        metric_fn=stl_metrics[task][0],
        n_epochs=config['training']['epochs'], 
        checkpoint_name=config['training']['checkpoint_name'], 
        patience=config['training']['patience'],
        device=device,
        logger=logger,
        higher_is_better=stl_metrics[task][1],
        scheduler=stl_schedulers[task],
        threshold=threshold
        )
      return result

    results = {}

    logger.info('RUNNING STL')
    for task in constants["tasks"]:
      print(stl_decoders[task])
      threshold = None if task == 'bigfive_s' else config['metrics']['class-args']['threshold']
      res = train_stl(task, threshold)
      print(task, res)
      results[task] = res['avg_tmetric'] # best_val_metric 

    return [results['mbti'], results['bigfive_c'], results['bigfive_s']]


  ### MTL
  elif learning_method == 'mtl':
    input_size = [embeddings_size + stats_size]
    enc_n_hidden = trial.suggest_int('enc_layers', 1, 5)
    suggested_powers = [trial.suggest_int(f'enc_power_layer_{i}', 4, 12) for i in range(enc_n_hidden)]
    enc_hidden = input_size + [pow(2, exp) for exp in suggested_powers]
    enc_dropout = trial.suggest_float(f'enc_dropout', 0.01, 0.80)

    # FFN
    class CustomEncoder(Encoder):
      def __init__(self, hidden=enc_hidden, dropout=enc_dropout, norm=add_norm):
          super(CustomEncoder, self).__init__(hidden, dropout, norm)

    dec_n_hidden = {task: trial.suggest_int(f'mtl_{task}_layers', 0, 4) for task in tasks}
    dec_hidden = {task: [pow(2, trial.suggest_int(f'mtl_{task}_power_layer_{i}', 4, 12)) for i in range(dec_n_hidden[task])] for task in tasks}
    dec_dropout = {task: trial.suggest_float(f'mtl_{task}_dropout', 0.01, 0.80) for task in tasks}
    dec_input_nn = [enc_hidden[-1]]
    print(enc_hidden)
    print(dec_hidden)
    print(dec_input_nn)
    decoders = nn.ModuleDict({k: Decoder(dec_input_nn + dec_hidden[k] + v[1], dec_dropout[k], final=v[0], norm=add_norm).to(device) for k, v in finals.items()})
    
    # Metrics
    mtl_task_dict = {
      'mbti': {
        'metrics': ['Acc'],
        'metrics_fn': src.metrics.BinaryMultilabelAccuracy(**config['metrics']['class-args']),
        'loss_fn': src.loss.BCELoss(),
        'weight': [1]
      },
      'bigfive_c': {
        'metrics': ['Acc'],
        'metrics_fn': src.metrics.BinaryMultilabelAccuracy(**config['metrics']['class-args']),
        'loss_fn': src.loss.BCELoss(),
        'weight': [1]
      },
      'bigfive_s': {
        'metrics': ['MAE'],
        'metrics_fn': src.metrics.MAEMetric(),
        'loss_fn': mtl.loss.MSELoss(),
        'weight': [0] # 0 means high loss is bad
      },
    }

    weighting = tsc('weighting', config['experiment']['weighting'].keys())
    architecture = tsc('architecture', config['experiment']['architecture'].keys())
    weight_args = config['experiment']['weighting'][weighting]
    arch_args = config['experiment']['architecture'][architecture]
    print(architecture)
    print(arch_args)

    if 'img_size' in arch_args.keys():
      arch_args['img_size'] = [input_size[0], 1]

    optim_param = {
      'optim': config['optim_param']['optim'],
      'lr': trial.suggest_float('mtl_lr', 0.00001, 0.01),
      'weight_decay': trial.suggest_float('mtl_weight_decay', 0.0001, 0.01), # Different for adam than adamw. AdamP recommends 0.01
      'betas': config['optim_param']['betas']
    }

    scheduler_param = {
      'scheduler': config['scheduler_param']['scheduler'],
      'step_size': trial.suggest_int('mtl_step_size', 1, 25), # How many steps before decay. Default 1
      'gamma': trial.suggest_float('mtl_gamma', 0.0001, 0.01) # Rate of lr decay. Default 0.1
    }

    # Trainer

    trainer = MTLTrainer(
        encoder_class=CustomEncoder,
        decoders=decoders,
        task_dict=mtl_task_dict,
        architecture=architecture,
        weighting=weighting,
        arch_args=arch_args,
        weight_args=weight_args,
        rep_grad=True,
        multi_input=True,
        optim_param=optim_param,
        scheduler_param=scheduler_param,
        device=device,
        bs=config['dataloaders']['test']['batch_size'],
        seed=config['seed'],
        save_path=paths["training"]["opt"],
    )

    # Training
    logger.info('RUNNING MTL')
    print('RUNNING MTL')
    try:
      res = trainer.train(
        train_dataloaders=dataloaders["train"],
        test_dataloaders=dataloaders["test"],
        val_dataloaders=dataloaders["val"],
        epochs=config["training"]["epochs"],
        patience=config['training']['patience']
      )
    except torch._C._LinAlgError as e:
        print(e)
        return defeat
    best_res = {task: res['best_result']['result'][task][0] for task in tasks}

    return [best_res['mbti'], best_res['bigfive_c'], best_res['bigfive_s']]

  else:
    raise ValueError(f'Learning method {learning_method} must be one of: stl, mtl')





if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optimize learning method')
  parser.add_argument('-l', type=str, help='stl / mtl')
  parser.add_argument('-s', type=str, help='org / imp / mix')
  parser.add_argument('-n', type=int, default=2, help='n trials')
  args = parser.parse_args()

  assert args.l in ('stl', 'mtl')
  assert args.s in ('org', 'imp', 'mix')

  paths = get_paths()
  identifier = f'{args.l}_{args.s}_{args.n}'
  db = paths['db']
  print("Database file will be created at:", os.path.abspath(db))
  journal = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(str(paths['outputs']['journal'] / f'{identifier}.log')),  # NFS path for distributed optimization
  )

  objective = lambda trial: unified(trial, learning_method=args.l, subset=args.s) 
  print(f'Args l: {args.l}')
  print(f'Args s: {args.s}')
  print(f'Args n: {args.n}')
  
  optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(paths['outputs']['study'] / f'{identifier}.out', mode='a'))
  study = optuna.create_study(
    study_name=identifier, 
    storage=journal, 
    directions=["maximize", "maximize", "minimize"], 
    load_if_exists=True
    ) # storage = journal/db
  study.set_user_attr("l", args.l)
  study.set_user_attr("s", args.s)
  study.set_user_attr("n", args.n)
  study.optimize(objective, n_trials=args.n, timeout=None)

  print("Number of finished trials: ", len(study.trials))

  print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

  best_mbti = max(study.best_trials, key=lambda t: t.values[0])
  best_bigfive_c = max(study.best_trials, key=lambda t: t.values[1])
  best_bigfive_s = min(study.best_trials, key=lambda t: t.values[2])
  besties = {
      'mbti': best_mbti, 
      'bigfive_c': best_bigfive_c, 
      'bigfive_s': best_bigfive_s
    }

  for name, best in besties.items():
    print(f"Trial with best performance for {name}: ")
    print(f"\tnumber: {best.number}")
    print(f"\tparams: {best.params}")
    print(f"\tvalues: {best.values}")

  with open(paths['results']['opt'] / f'{identifier}.json', "a") as f:
    for met in range(3):
      best = max(study.best_trials, key=lambda t: t.values[2])
      json.dump(best.params, f)
      f.write(str(best.values))

  def save_plot(fig, num, idf, name):
    dir_num = paths['img']['opt'] / num
    dir_idf = dir_num / idf
    os.makedirs(str(dir_num), exist_ok=True)
    os.makedirs(str(dir_idf), exist_ok=True)
    fig.write_image(str(dir_idf / f'{name}.png'))

  

  none_plots = [
    (plot_intermediate_values, 'intermediate_values')
  ]

  single_plots = [
    (plot_edf, 'edf'),
    (plot_param_importances, 'param_importances'),
    (plot_optimization_history, 'optimization_history'),
    (plot_parallel_coordinate, 'parallel_coordinate'),
    (plot_slice, 'slice'),
    (plot_contour, 'contour'),
  ]

  double_plots = [
    (optuna.visualization.plot_pareto_front, 'pareto_front'),
  ]

  single_args = [
    { 
      'target': lambda t: t.values[0],
      'target_name': "MBTI Acc",
    },
    { 
      'target': lambda t: t.values[1],
      'target_name': "Big Five Acc",
    },
    { 
      'target': lambda t: t.values[2],
      'target_name': "Big Five MAE",
    },
  ]

  double_args = [
    { 
      'targets': lambda t: (t.values[0], t.values[1]),
      'target_names': ["MBTI Acc", "Big Five Acc"],
    },
    { 
      'targets': lambda t: (t.values[1], t.values[2]),
      'target_names': ["Big Five Acc", "Big Five MAE"],
    },
    { 
      'targets': lambda t: (t.values[2], t.values[0]),
      'target_names': ["Big Five MAE", "MBTI Acc"],
    },
  ]

  for pl, name in none_plots:
    fig = pl(study)
    save_plot(fig, f'{args.n}', f'{args.l}_{args.s}', f'{name}')
  
  for pl, name in single_plots:
    for sargs in single_args:
      fig = pl(study, **sargs)
      save_plot(fig, f'{args.n}', f'{args.l}_{args.s}', f'{name}_{sargs["target_name"]}')
  
  for pl, name in double_plots:
    for dargs in double_args:
      fig = pl(study, **dargs)
      save_plot(fig, f'{args.n}', f'{args.l}_{args.s}', f'{name}_{dargs["target_names"][0]}_{dargs["target_names"][1]}')
  
  fig = plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")
  save_plot(fig, f'{args.n}', f'{args.l}_{args.s}', f'duration')

  logger.info(f'{identifier} FINSIHED')