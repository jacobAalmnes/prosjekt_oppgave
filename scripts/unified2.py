import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path: sys.path.append(str(proj_path))

import gc
import os
import itertools
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

import tracemalloc
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

from optuna.samplers import GridSampler, TPESampler, BaseSampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from optuna.distributions import CategoricalDistribution
import warnings
from optuna.exceptions import ExperimentalWarning
import matplotlib.pyplot as plt
import torchvision
import src
from src.trainer import MTLTrainer, train, get_optimizer, get_scheduler
from src.ffn import Encoder, Decoder
from src.datasets import SingleInputDataset, get_stl_dataloaders, get_mtl_dataloaders, split_df
from src.utils import get_commons, get_paths
import src.loss
import src.metrics
import LibMTL as mtl
import pandas as pd
import numpy as np
import tracemalloc
from sklearn.preprocessing import StandardScaler
import random
from torch import nn
import torcheval.metrics
import torchmetrics
import torch
import optuna
import json
import logging
import argparse
import itertools
import os
import gc
import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'stefandt' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path:
    sys.path.append(str(proj_path))


proj_path = Path('/cluster') / 'work' / 'stefandt' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path:
    sys.path.append(str(proj_path))


# from optuna.integration import ShapleyImportanceEvaluator


warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


finals = {  # final layer act func, size
    'mbti': ('sigmoid', [4]),
    'bigfive_c': ('sigmoid', [5]),
    'bigfive_s': ('none', [5])
}


def normalize(df: pd.DataFrame, cols: str):
  scaler = StandardScaler()
  df[cols] = scaler.fit_transform(df[cols])
  return df


def get_fixed_params(constants, config):
    fixed_params = {
        'normalize_stats': True,
        'stats': constants["stats_columns"],
        **{f'use {emb}': False for emb in constants['embedding_sizes'].keys()},
        **{f'use {stat}': True for stat in constants['stats_columns']},

        # STL parameters
        **{f'stl_{task}_layers': 1 for task in constants['tasks']},
        **{f'stl_{task}_dropout': 0.3 for task in constants['tasks']},
        **{f'stl_{task}_lr': 1e-3 for task in constants['tasks']},
        **{f'stl_{task}_weight_decay': 1e-3 for task in constants['tasks']},
        **{f'stl_{task}_step_size': 10 for task in constants['tasks']},
        **{f'stl_{task}_gamma': 0.001 for task in constants['tasks']},

        # MTL parameters
        'enc_layers': 2,
        'enc_dropout': 0.5,
        **{f'mtl_{task}_layers': 1 for task in constants['tasks']},
        **{f'mtl_{task}_dropout': 0.3 for task in constants['tasks']},
        'mtl_lr': 1e-3,
        'mtl_weight_decay': 1e-3,
        'mtl_step_size': 10,
        'mtl_gamma': 0.5,

        # Architecture and weighting (for MTL)
        'architecture': list(config['experiment']['mtl_all']['architecture'].keys())[0],
        'weighting': list(config['experiment']['mtl_all']['weighting'].keys())[0],
    }

    # Add layer sizes
    for task in constants['tasks']:
        fixed_params[f'stl_{task}_power_layer_0'] = 8

    fixed_params['enc_power_layer_0'] = 8

    for task in constants['tasks']:
        fixed_params[f'mtl_{task}_power_layer_0'] = 8

    return fixed_params

def run_experiment_with_fixed_params(embedding_type):
    # Define fixed parameters
    fixed_params = {
        'normalize_stats': True,
        'stats': constants["stats_columns"],
        **{f'use {emb}': False for emb in constants['embedding_sizes'].keys()},
        **{f'use {stat}': True for stat in constants['stats_columns']},

        # STL parameters
        **{f'stl_{task}_layers': 1 for task in constants['tasks']},
        **{f'stl_{task}_dropout': 0.3 for task in constants['tasks']},
        **{f'stl_{task}_lr': 1e-3 for task in constants['tasks']},
        **{f'stl_{task}_weight_decay': 1e-3 for task in constants['tasks']},
        **{f'stl_{task}_step_size': 10 for task in constants['tasks']},
        **{f'stl_{task}_gamma': 0.001 for task in constants['tasks']},

        # MTL parameters
        'enc_layers': 2,
        'enc_dropout': 0.5,
        **{f'mtl_{task}_layers': 1 for task in constants['tasks']},
        **{f'mtl_{task}_dropout': 0.3 for task in constants['tasks']},
        'mtl_lr': 1e-3,
        'mtl_weight_decay': 1e-3,
        'mtl_step_size': 10,
        'mtl_gamma': 0.5,

        # Architecture and weighting (for MTL)
        'architecture': list(config['experiment']['mtl_all']['architecture'].keys())[0],
        'weighting': list(config['experiment']['mtl_all']['weighting'].keys())[0],
    }

    # Add layer sizes
    for task in constants['tasks']:
        fixed_params[f'stl_{task}_power_layer_0'] = 8

    fixed_params['enc_power_layer_0'] = 8

    for task in constants['tasks']:
        fixed_params[f'mtl_{task}_power_layer_0'] = 8

    
    # Create a FixedTrial with the fixed parameters
    fixed_trial = optuna.trial.FixedTrial(fixed_params)
    
    return objective(fixed_trial)



def unified(trial: optuna.trial.FixedTrial, learning_method: str, subset: str, use_embeddings: bool = True):
  print(trial.params)
  assert learning_method.lower() in ('stl', 'mtl')
  assert subset.lower() in ('org', 'imp', 'mix')
  gc.collect()
  paths, constants, config, logger, device = get_commons(config_name='opt')

  set_seed(config['seed'])
  rng = np.random.default_rng(config['seed'])
  add_norm = True  # To prevent exploding gradients
  defeat = (0, 0, 100)
  tasks = constants['tasks']

  # DataFrame
  org_cols = [('ORG_TARGET', col) for task in constants["tasks"]
              for col in constants["columns"][task]]
  tar_cols = [('IMP_TARGET', col) for task in constants["tasks"]
              for col in constants["columns"][task]]
  mbti_cols = [(f'{tar}_TARGET', col)
               for col in constants['bigfive_s_columns'] for tar in ('ORG', 'IMP')]

  dataframe = pd.read_csv(
      paths['new']['imputed-all'], header=[0, 1], index_col=0)
  if trial.params['normalize_stats']:
      dataframe = normalize(dataframe, 'STATS')

  if use_embeddings is True:
    selected_embs = {}
    for model_name in constants['embedding_sizes'].keys():
      if trial.params[f'use {model_name}']:
        selected_embs[model_name] = constants['embedding_sizes'][model_name]

    if not selected_embs:
      return defeat

    embedding_list = []
    for model_name in selected_embs.keys():
        emb = pd.read_csv(paths['new']['old_embeddings']
                          [model_name], index_col=0)
        new_columns = pd.MultiIndex.from_product([['CLS'], emb.columns])
        new_emb = pd.DataFrame(
            emb.values, columns=new_columns, index=emb.index)
        embedding_list.append(new_emb)

    embeddings_size = sum(selected_embs.values())
  else:
    embeddings_size = 0

  selected_cols = []
  for col in constants['stats_columns']:
    if trial.params[f'use {col}']:
      selected_cols.append(col)

  if not selected_cols:
    return defeat

  for stat in selected_cols:
    if not stat:
      dataframe = dataframe.drop(('STATS', stat), axis=1)

  if 'STATS' not in dataframe.columns.get_level_values(0):
      return defeat

  stats_size = len(dataframe['STATS'].columns)

  if use_embeddings:
    dataframe = pd.concat([dataframe] + embedding_list, axis=1, copy=False)

  print(dataframe.shape)

  finals = {  # final layer act func, size
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
      'use_embeddings': use_embeddings,
  }

  dataloaders = split_df(learning=learning_method, subset=subset, **dl_args)

  logger.info(f'STARTING {learning_method}')

  # STL
  if learning_method == 'stl':

    stl_metrics = {
        # Metric object, higher_is_better
        'mbti': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True),
        'bigfive_c': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True),
        'bigfive_s': (torchmetrics.regression.MeanAbsoluteError(), False)
    }

    stl_loss_fns = {
        'mbti': nn.BCELoss(),
        'bigfive_c': nn.BCELoss(),
        'bigfive_s': nn.MSELoss()
    }

    stl_n_hidden = {task: trial.params[f'stl_{task}_layers'] for task in tasks}
    stl_hidden = {task: [pow(2, trial.params[f'stl_{task}_power_layer_{i}'])
                         for i in range(stl_n_hidden[task])] for task in tasks}
    stl_dropout = {task: trial.params[f'stl_{task}_dropout'] for task in tasks}
    stl_input_nn = [embeddings_size + stats_size]

    stl_decoders = nn.ModuleDict({k: Decoder(
        stl_input_nn + stl_hidden[k] + v[1], dropout=stl_dropout[k], final=v[0], norm=add_norm).to(device) for k, v in finals.items()})

    stl_optimizers = {}
    stl_schedulers = {}
    for task in tasks:
      optim_param = {
          'optim': 'adamp',
          'lr': trial.params[f'stl_{task}_lr'],
          # Different for adam than adamw. AdamP recommends 0.01
          'weight_decay': trial.params[f'stl_{task}_weight_decay'],
          'betas': config['optim_param']['betas']
      }

      scheduler_param = {
          'scheduler': config['scheduler_param']['scheduler'],
          # How many steps before decay. Default 1
          'step_size': trial.params[f'stl_{task}_step_size'],
          # Rate of lr decay. Default 0.1
          'gamma': trial.params[f'stl_{task}_gamma']
      }

      stl_alg, stl_optim_arg = get_optimizer(optim_param)
      stl_optimizers[task] = stl_alg(
          stl_decoders[task].parameters(), **stl_optim_arg)
      stl_schedulers[task] = get_scheduler(
          stl_optimizers[task], scheduler_param)

    def train_stl(task: str, threshold) -> dict:
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
      results[task] = res['avg_tmetric']  # best_val_metric

    return [results['mbti'], results['bigfive_c'], results['bigfive_s']]

  # MTL
  elif learning_method == 'mtl':
    input_size = [embeddings_size + stats_size]
    enc_n_hidden = trial.params['enc_layers']
    suggested_powers = [
        trial.params[f'enc_power_layer_{i}'] for i in range(enc_n_hidden)]
    enc_hidden = input_size + [pow(2, exp) for exp in suggested_powers]
    enc_dropout = trial.params[f'enc_dropout']

    # FFN
    class CustomEncoder(Encoder):
      def __init__(self, hidden=enc_hidden, dropout=enc_dropout, norm=add_norm):
          super(CustomEncoder, self).__init__(hidden, dropout, norm)

    dec_n_hidden = {task: trial.params[f'mtl_{task}_layers'] for task in tasks}
    dec_hidden = {task: [pow(2, trial.params[f'mtl_{task}_power_layer_{i}'])
                         for i in range(dec_n_hidden[task])] for task in tasks}
    dec_dropout = {task: trial.params[f'mtl_{task}_dropout'] for task in tasks}
    dec_input_nn = [enc_hidden[-1]]
    print(enc_hidden)
    print(dec_hidden)
    print(dec_input_nn)
    decoders = nn.ModuleDict({k: Decoder(
        dec_input_nn + dec_hidden[k] + v[1], dec_dropout[k], final=v[0], norm=add_norm).to(device) for k, v in finals.items()})

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
            'weight': [0]  # 0 means high loss is bad
        },
    }

    weighting = trial.params['weighting']
    architecture = trial.params['architecture']
    weight_args = config['experiment']['weighting'][weighting]
    arch_args = config['experiment']['architecture'][architecture]
    print(architecture)
    print(arch_args)

    if 'img_size' in arch_args.keys():
      arch_args['img_size'] = [input_size[0], 1]

    optim_param = {
        'optim': config['optim_param']['optim'],
        'lr': trial.params['mtl_lr'],
        # Different for adam than adamw. AdamP recommends 0.01
        'weight_decay': trial.params['mtl_weight_decay'],
        'betas': config['optim_param']['betas']
    }

    scheduler_param = {
        'scheduler': config['scheduler_param']['scheduler'],
        # How many steps before decay. Default 1
        'step_size': trial.params['mtl_step_size'],
        'gamma': trial.params['mtl_gamma']  # Rate of lr decay. Default 0.1
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
    raise ValueError(
        f'Learning method {learning_method} must be one of: stl, mtl'
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optimize learning method')
  parser.add_argument('-l', type=str, help='stl / mtl')
  parser.add_argument('-s', type=str, help='org / imp / mix')
  parser.add_argument('-n', type=int, default=2, help='n trials')
  parser.add_argument('-e', type=str, default=None,
                      help='experiment: None / emb / stats / mtl_all')
  args = parser.parse_args()

  assert args.l in ('stl', 'mtl')
  assert args.s in ('org', 'imp', 'mix')

  paths, constants, config, logger, device = get_commons(config_name='unified')

  if args.e is None:
      identifier = f'{args.l}_{args.s}_{args.n}'
      mini_identifier = f'{args.l}_{args.s}'
  else:
      identifier = f'{args.l}_{args.s}_{args.n}_{args.e}'
      mini_identifier = f'{args.l}_{args.s}_{args.e}'

  db = paths['db']
  print("Database file will be created at:", os.path.abspath(db))
  journal = optuna.storages.JournalStorage(
      optuna.storages.JournalFileStorage(
          str(paths['outputs']['journal'] / f'{identifier}.log')),
  )

  use_embeddings = False if args.e == 'stats' else True
  fixed_params = get_fixed_params(constants, config)
  print("Fixed params:", fixed_params)

  print(f'Args l: {args.l}')
  print(f'Args s: {args.s}')
  print(f'Args n: {args.n}')
  print(f'Args e: {args.e}')

  optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(
      paths['outputs']['study'] / f'{identifier}.out', mode='a'))

  study = optuna.create_study(
      study_name=identifier,
      storage=journal,
      directions=["maximize", "maximize", "minimize"],
      load_if_exists=True
  )
  print('study constructed')

  study.set_user_attr("l", args.l)
  study.set_user_attr("s", args.s)
  study.set_user_attr("n", args.n)
  study.set_user_attr("e", args.e)

  print('Optimizing..')

  if args.e == 'emb':
      embedding_combinations = list(itertools.product(
          [False, True], repeat=len(constants['embedding_sizes'])))
      results = []
      for combination in embedding_combinations:
          current_params = fixed_params.copy()
          for emb, use in zip(constants['embedding_sizes'].keys(), combination):
              current_params[f'use {emb}'] = use
          # Add this line for debugging
          print("Current params:", current_params)
          trial = optuna.trial.FixedTrial(current_params)
          result = unified(trial, learning_method=args.l,
                           subset=args.s, use_embeddings=True)
          results.append((combination, result))

      # Process and store results
      for combination, result in results:
          study.add_trial(
              optuna.trial.create_trial(
                  params={f'use {emb}': use for emb, use in zip(
                      constants['embedding_sizes'].keys(), combination)},
                  distributions={},
                  values=result,  # This should be a list of three values
              )
          )
  else:
      study.optimize(
          func=lambda trial: unified(optuna.trial.FixedTrial(
              fixed_params), learning_method=args.l, subset=args.s, use_embeddings=use_embeddings),
          n_trials=args.n,
          n_jobs=1,
          gc_after_trial=True,
          timeout=None
      )

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
    save_plot(fig, f'{args.n}', mini_identifier, f'{name}')

  for pl, name in single_plots:
    for sargs in single_args:
      fig = pl(study, **sargs)
      save_plot(fig, f'{args.n}', mini_identifier,
                f'{name}_{sargs["target_name"]}')

  for pl, name in double_plots:
    for dargs in double_args:
      fig = pl(study, **dargs)
      save_plot(fig, f'{args.n}', mini_identifier, f'{name}_{dargs["target_names"][0]}_{dargs["target_names"][1]}')

  fig = plot_param_importances(
      study, target=lambda t: t.duration.total_seconds(), target_name="duration")
  save_plot(fig, f'{args.n}', mini_identifier, f'duration')

  logger.info(f'{identifier} FINSIHED')
