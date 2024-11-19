import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path: sys.path.append(str(proj_path))

import argparse
import logging

import torch
import torchmetrics
import torcheval.metrics
from torch import nn

import numpy as np
import pandas as pd
import LibMTL as mtl

import src.metrics
import src.loss
from src.utils import get_commons
from src.datasets import SingleInputDataset, get_stl_dataloaders, get_mtl_dataloaders, split_df
from src.ffn import Encoder, Decoder
from src.trainer import MTLTrainer, train, get_optimizer, get_scheduler

def main(experiment:str):
  """experiment: Name of experiment to perform"""
  
  paths, constants, config, logger, device = get_commons(config_name='experiments4')
  logger.setLevel(logging.DEBUG)
  rng = np.random.default_rng(config['seed'])
  iteration_name = f"{config['iteration']}.csv"

  experiment_args = config['experiment'][experiment]

  print(experiment)
  print(experiment_args)

  def normalize(df:pd.DataFrame) -> pd.DataFrame:
    df['STATS'] = (df['STATS']-df['STATS'].mean()) / df['STATS'].std()
    # for target in ('IMP_TARGET', 'ORG_TARGET'):
    #   select = (target, 'bigfive_s')
    #   df[select] = (df[select]-df[select].mean()) / df[select].std()
    return df

  if experiment not in ('embeddings', 'combine-emb', 'imputation'):
    model_name = config['embeddings']['model_name']
    embedding_size = constants['valid_embedding_sizes'][model_name]

    dataframe = pd.read_csv(paths['new']['imputed-all'], header=[0, 1], index_col=0)
    if experiment == 'normalization':
      unnorm = dataframe.copy()
    dataframe = normalize(dataframe)
        
    embeddings = pd.read_csv(paths['new']['embeddings'][config['embeddings']['model_name']], index_col=0)
    stats_size = len(dataframe['STATS'].columns)

    new_columns = pd.MultiIndex.from_product([['CLS'], embeddings.columns])
    new_emb = pd.DataFrame(embeddings.values, columns=new_columns, index=embeddings.index)

    dataframe = pd.concat([dataframe, new_emb], axis=1)
    mtl_dataloaders = split_df(dataframe, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='mtl')
    if experiment == 'normalization':
      unnorm = pd.concat([unnorm, new_emb], axis=1)
      unnorm_dataloaders = split_df(unnorm, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='mtl')

  def get_mtl_task_dict():
    mtl_task_dict = {
      'mbti': {
        'metrics': ['Acc'],
        'metrics_fn': src.metrics.BinaryMultilabelAccuracy(**config['metrics']['class-args']),
        'loss_fn': src.loss.BCELoss(factor=1),
        'weight': [1]
      },
      'bigfive_c': {
        'metrics': ['Acc'],
        'metrics_fn': src.metrics.BinaryMultilabelAccuracy(**config['metrics']['class-args']),
        'loss_fn': src.loss.BCELoss(factor=1),
        'weight': [1]
      },
      'bigfive_s': {
        'metrics': ['MAE'],
        'metrics_fn': src.metrics.MAEMetric(),
        'loss_fn': mtl.loss.MSELoss(),
        'weight': [0] # 0 means high loss is bad
      },
    }
    return mtl_task_dict

  finals = { # final layer act func, size
    'mbti': ('sigmoid', [4]), 
    'bigfive_c': ('sigmoid', [5]),
    'bigfive_s': ('none', [5])
    }
  
  def get_stl_loss_fns():
    stl_loss_fns = {
        'mbti': nn.BCELoss(),
        'bigfive_c': nn.BCELoss(),
        'bigfive_s': nn.MSELoss()
        }
    return stl_loss_fns

  def get_stl_metrics():
    stl_metrics = {
        'mbti': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True), # Metric object, higher_is_better
        'bigfive_c': (torcheval.metrics.MultilabelAccuracy(**config['metrics']['class-args']), True),
        'bigfive_s': (torchmetrics.regression.MeanAbsoluteError(), False)
      }
    return stl_metrics

  def get_mtl_commons():
    mtl_commons = {
      'encoder_class': Encoder,
      'task_dict': get_mtl_task_dict(),
      'weighting': config['defaults']['weighting'],
      'architecture': config['defaults']["architecture"],
      'rep_grad': True,
      'multi_input': True,
      'optim_param': config["optim_param"],
      'scheduler_param': config["scheduler_param"],
      'device': device,
      'bs': config['dataloaders']['test']['batch_size'],
      'seed': config['seed']
    }
    return mtl_commons
  
  if experiment not in ('embeddings', 'combine-emb', 'imputation'):
    mtl_training = {
        'train_dataloaders': mtl_dataloaders["train"],
        'test_dataloaders': mtl_dataloaders["test"],
        'val_dataloaders': mtl_dataloaders["val"],
        'epochs': config["training"]["epochs"],
        'patience': config["training"]["patience"]
    }

  ##################################
  #                                #
  #     E X P E R I M E N T S      #
  #                                #
  ##################################

  # NORM
  #####################################
  if experiment == 'normalization':
    results = {}
    names = ['norm', 'unnorm']
    loaders = [mtl_dataloaders, unnorm_dataloaders]
    for name, dls in zip(names, loaders):
      logger.info(dls)

      mtl_input_nn = [config['encoder']['nn'][-1]]
      mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}

      logger.info(mtl_decoders)
      
      kwargs = {
        'arch_args': config['defaults']['kwargs']['arch_args'],
        'weight_args': config['defaults']['kwargs']['weight_args']
        }
      mtl_commons = get_mtl_commons()
      mtl_trainer = MTLTrainer(
          # task_dict=get_mtl_task_dict(),
          # weighting=config["defaults"]["weighting"],
          # architecture=config["defaults"]["architecture"],
          decoders=mtl_decoders,
          save_path=paths["training"]["weighting"],
          **mtl_commons,
          **kwargs
      )

      dls_training = {
          'train_dataloaders': dls["train"],
          'test_dataloaders': dls["test"],
          'val_dataloaders': dls["val"],
          'epochs': config["training"]["epochs"],
          'patience': config["training"]["patience"]
      }

      result = mtl_trainer.train(**dls_training)
      results[name] = result['best_result']['result']

      logger.info(results)

    data = pd.DataFrame(results)
    data.to_csv(paths['results']['normalization'] / iteration_name)

  # DECODER
  ######################################
  elif experiment == 'decoder':
    results = {}
    for dec_name, dec_args in experiment_args.items():

      class SpecialEncoder(nn.Module):
        def __init__(self, hidden=dec_args['encoder']['nn'], dropout=dec_args['encoder']['dropout']):
          super(SpecialEncoder, self).__init__()
          print(hidden, dropout)
          layers = []
          for i in range(len(hidden) - 1):
              layers.append(nn.Linear(hidden[i], hidden[i + 1]))
              if i < len(hidden) - 2:
                  layers.append(nn.ReLU())
                  layers.append(nn.Dropout(dropout))
          self.model = nn.Sequential(*layers)

        def forward(self, x:torch.Tensor) -> torch.Tensor:
          return self.model(x)
      

      mtl_input_nn = [dec_args['encoder']['nn'][-1]]
      mtl_decoders = {k: Decoder(mtl_input_nn + dec_args['mtl-decoders']['hidden_nn'] + v[1], dropout=dec_args['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}
      logger.info(mtl_decoders)
      
      kwargs = {
        'arch_args': config['defaults']['kwargs']['arch_args'],
        'weight_args': config['defaults']['kwargs']['weight_args']
        }

      mtl_commons = get_mtl_commons()
      mtl_commons['encoder_class'] = SpecialEncoder
      mtl_trainer = MTLTrainer(
          decoders=mtl_decoders,
          save_path=paths["training"]["decoder"],
          **mtl_commons,
          **kwargs
      )
      result = mtl_trainer.train(**mtl_training)
      results[dec_name] = result['best_result']['result']

      logger.info(results)

    data = pd.DataFrame(results)
    data.to_csv(paths['results']['decoder'] / iteration_name)

  # IMPUTATION
  ###############################
  elif experiment == 'imputation':
    final_results = {}
    for scale in ('org', 'imp'):
      if scale == 'org':
        dataframes = {task: pd.read_csv(paths['new']['split'][task], header=[0, 1], index_col=0) for task in constants['tasks']}
        dataframes = {task: df.join(new_emb).dropna(axis=0) for task, df in dataframes.items()}
        datasets = {task: SingleInputDataset(df, target_col='ORG_TARGET') for task, df in dataframes.items()}

        stl_dataloaders = {task: get_stl_dataloaders(dataset, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger) for task, dataset in datasets.items()}
        mtl_dataloaders = get_mtl_dataloaders(datasets, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger)
      
      elif scale == 'imp': 
        stl_dataloaders = split_df(dataframe, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='stl')
        mtl_dataloaders = split_df(dataframe, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='mtl')

      stats_size = len(dataframe['STATS'].columns)
      stl_input_nn = [embedding_size + stats_size]
      mtl_input_nn = [config['encoder']['nn'][-1]]

      stl_decoders = nn.ModuleDict({k: Decoder(stl_input_nn + config['stl-decoders']['hidden_nn'] + v[1], dropout=config['stl-decoders']['dropout'], final=v[0]) for k, v in finals.items()})
      mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}

      logger.info(stl_decoders)
      logger.info(mtl_decoders)

      stl_alg, stl_optim_arg = get_optimizer(config['optim_param'])
      stl_optimizers = {task: stl_alg(stl_decoders[task].parameters(), **stl_optim_arg) for task in constants["tasks"]}
      stl_schedulers = {k: get_scheduler(v, config['scheduler_param']) for k, v in stl_optimizers.items()}
      stl_loss_fns = get_stl_loss_fns()
      stl_metrics = get_stl_metrics()

      def train_stl(task:str, threshold) -> dict:
          result = train(
            stl_decoders[task], 
            stl_dataloaders[task], 
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

      kwargs = {
          'arch_args': config['defaults']['kwargs']['arch_args'],
          'weight_args': config['defaults']['kwargs']['weight_args']
          }
      mtl_commons = get_mtl_commons()
      mtl_trainer = MTLTrainer(
          #task_dict=get_mtl_task_dict(),
          #weighting=config['defaults']['weighting'],
          #architecture=config['defaults']["architecture"],
          decoders=mtl_decoders,
          save_path=paths["training"]["imputation"],
          **mtl_commons,
          **kwargs
      )

      mtl_training = {
        'train_dataloaders': mtl_dataloaders["train"],
        'test_dataloaders': mtl_dataloaders["test"],
        'val_dataloaders': mtl_dataloaders["val"],
        'epochs': config["training"]["epochs"],
        'patience': config["training"]["patience"]
      }

      intermediate_results = {"STL": {}, "MTL": {}}
      
      logger.info('RUNNING STL')
      for task in constants["tasks"]:
        threshold = None if task == 'bigfive_s' else config['training']['threshold']
        result = train_stl(task, threshold)
        print(task, result)
        intermediate_results["STL"][task] = result

      logger.info('\nRUNNING MTL')
      intermediate_results["MTL"] = mtl_trainer.train(**mtl_training)

      entry = {'STL': {},'MTL': {}}
      for task in constants['tasks']:
        entry['STL'][task] = intermediate_results['STL'][task]['best_val_metric']
        entry['MTL'][task] = intermediate_results['MTL']['best_result']['result'][task][0]

      final_results[scale] = entry

    data = pd.DataFrame(final_results)

    logger.info('RESULTS')
    logger.info(data)
    logger.info(data.head())

    data.to_csv(paths['results']['imputation'] / iteration_name)
  

  # EMBEDDINGS
  ####################################
  elif experiment == 'embeddings':
    results = {}
    dataframe = pd.read_csv(paths['new']['imputed-all'], header=[0, 1], index_col=0)
    # dataframe = normalize(dataframe)
    for model_name, embedding_size in constants['valid_embedding_sizes'].items():
      print(model_name)
      # if model_name == 'distilbert': continue
      embeddings = pd.read_csv(paths['new']['embeddings'][model_name], index_col=0)

      new_columns = pd.MultiIndex.from_product([['CLS'], embeddings.columns])
      new_emb = pd.DataFrame(embeddings.values, columns=new_columns, index=embeddings.index)

      dataframes = {task: pd.read_csv(paths['new']['split'][task], header=[0, 1], index_col=0) for task in constants['tasks']}
      dataframes = {task: df.join(new_emb).dropna(axis=0) for task, df in dataframes.items()}
      datasets = {task: SingleInputDataset(dataframe, target_col='ORG_TARGET') for task, dataframe in dataframes.items()}

      stl_dataloaders = {task: get_stl_dataloaders(dataset, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger) for task, dataset in datasets.items()}
      mtl_dataloaders = get_mtl_dataloaders(datasets, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger)

      stats_size = len(dataframe['STATS'].columns)
      stl_input_nn = [embedding_size + stats_size]
      mtl_input_nn = [config['encoder']['nn'][-1]]

      stl_decoders = nn.ModuleDict({k: Decoder(stl_input_nn + config['stl-decoders']['hidden_nn'] + v[1], dropout=config['stl-decoders']['dropout'], final=v[0]) for k, v in finals.items()})
      mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}

      logger.info(stl_decoders)
      logger.info(mtl_decoders)

      stl_alg, stl_optim_arg = get_optimizer(config['optim_param'])
      stl_optimizers = {task: stl_alg(stl_decoders[task].parameters(), **stl_optim_arg) for task in constants["tasks"]}
      stl_schedulers = {k: get_scheduler(v, config['scheduler_param']) for k, v in stl_optimizers.items()}
      stl_loss_fns = get_stl_loss_fns()
      stl_metrics = get_stl_metrics()

      def train_stl(task:str, threshold) -> dict:
          result = train(
            stl_decoders[task], 
            stl_dataloaders[task], 
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

      kwargs = {
          'arch_args': config['defaults']['kwargs']['arch_args'],
          'weight_args': config['defaults']['kwargs']['weight_args']
          }
      mtl_commons = get_mtl_commons()
      mtl_trainer = MTLTrainer(
          decoders=mtl_decoders,
          save_path=paths["training"]["embeddings"],
          **mtl_commons,
          **kwargs
      )

      emb_results = {"STL": {}, "MTL": {}}
      
      logger.info('RUNNING STL')
      for task in constants["tasks"]:
        threshold = None if task == 'bigfive_s' else config['training']['threshold']
        result = train_stl(task, threshold)
        print(task, result)
        emb_results["STL"][task] = result
      
      mtl_training = {
        'train_dataloaders': mtl_dataloaders["train"],
        'test_dataloaders': mtl_dataloaders["test"],
        'val_dataloaders': mtl_dataloaders["val"],
        'epochs': config["training"]["epochs"],
        'patience': config["training"]["patience"]
      }

      logger.info('\nRUNNING MTL')
      emb_results["MTL"] = mtl_trainer.train(**mtl_training)

      print(emb_results)

      data_dict = {'STL': {},'MTL': {}}
      for task in constants['tasks']:
        data_dict['STL'][task] = emb_results['STL'][task]['best_val_metric']
        data_dict['MTL'][task] = emb_results['MTL']['best_result']['result'][task][0]

      results[model_name] = data_dict
    
    print('RESULTS')
    print(results)
    data = pd.DataFrame(results)

    data.to_csv(paths['results']['embeddings'] / iteration_name)
        

  # LEARNING METHOD 
  ##########################################
  elif experiment == 'learning_method':

    dataframes = {task: pd.read_csv(paths['new']['split'][task], header=[0, 1], index_col=0) for task in constants['tasks']}
    dataframes = {task: df.join(new_emb).dropna(axis=0) for task, df in dataframes.items()}
    datasets = {task: SingleInputDataset(dataframe, target_col='ORG_TARGET') for task, dataframe in dataframes.items()}

    stl_dataloaders = {task: get_stl_dataloaders(dataset, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger) for task, dataset in datasets.items()}
    mtl_dataloaders = get_mtl_dataloaders(datasets, config['split']['train'], config['split']['test'], config['dataloaders'], logger=logger)

    stats_size = len(dataframe['STATS'].columns)
    stl_input_nn = [embedding_size + stats_size]
    mtl_input_nn = [config['encoder']['nn'][-1]]

    stl_decoders = nn.ModuleDict({k: Decoder(stl_input_nn + config['stl-decoders']['hidden_nn'] + v[1], dropout=config['stl-decoders']['dropout'], final=v[0]) for k, v in finals.items()})
    mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}

    logger.info(stl_decoders)
    logger.info(mtl_decoders)

    stl_alg, stl_optim_arg = get_optimizer(config['optim_param'])
    stl_optimizers = {task: stl_alg(stl_decoders[task].parameters(), **stl_optim_arg) for task in constants["tasks"]}
    stl_schedulers = {k: get_scheduler(v, config['scheduler_param']) for k, v in stl_optimizers.items()}
    stl_loss_fns = get_stl_loss_fns()
    stl_metrics = get_stl_metrics()

    def train_stl(task:str, threshold) -> dict:
        result = train(
          stl_decoders[task], 
          stl_dataloaders[task], 
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

    kwargs = {
        'arch_args': config['defaults']['kwargs']['arch_args'],
        'weight_args': config['defaults']['kwargs']['weight_args']
        }
    mtl_commons = get_mtl_commons()
    mtl_trainer = MTLTrainer(
        #task_dict=get_mtl_task_dict(),
        #weighting=config['defaults']['weighting'],
        #architecture=config['defaults']["architecture"],
        decoders=mtl_decoders,
        save_path=paths["training"]["learning_method"],
        **mtl_commons,
        **kwargs
    )

    results = {"STL": {}, "MTL": {}}
    
    logger.info('RUNNING STL')
    for task in constants["tasks"]:
      threshold = None if task == 'bigfive_s' else config['training']['threshold']
      result = train_stl(task, threshold)
      print(task, result)
      results["STL"][task] = result

    logger.info('\nRUNNING MTL')
    results["MTL"] = mtl_trainer.train(**mtl_training)

    print(results)

    data_dict = {'STL': {},'MTL': {}}
    for task in constants['tasks']:
      data_dict['STL'][task] = results['STL'][task]['best_val_metric']
      data_dict['MTL'][task] = results['MTL']['best_result']['result'][task][0]

    data = pd.DataFrame(data_dict)

    logger.info('RESULTS')
    logger.info(data)
    logger.info(data.head())

    data.to_csv(paths['results']['learning_method'] / iteration_name)
  
  # WEIGHTING
  ################################
  elif experiment == 'weighting':
    results = {}
    for weighting, weight_args in experiment_args.items():
      logger.info(weighting)
      logger.info(weight_args)
      mtl_input_nn = [config['encoder']['nn'][-1]]
      mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}
      logger.info(mtl_decoders)
      
      mtl_commons = get_mtl_commons()
      mtl_commons['weighting'] = weighting
      kwargs = {
        'arch_args': config['defaults']['kwargs']['arch_args'],
        'weight_args': weight_args
        }
      mtl_trainer = MTLTrainer(
          decoders=mtl_decoders,
          save_path=paths["training"]["weighting"],
          **mtl_commons,
          **kwargs
      )
      result = mtl_trainer.train(**mtl_training)
      results[weighting] = result['best_result']['result']

      logger.info(results)

    data = pd.DataFrame(results)
    data.to_csv(paths['results']['weighting'] / iteration_name)
  
  # ARCH
  #############################
  elif experiment == 'architecture':
    results = {}
    for arch, arch_args in experiment_args.items():
      logger.info(arch)
      logger.info(arch_args)
      mtl_input_nn = [config['encoder']['nn'][-1]]
      mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}
      logger.info(mtl_decoders)
      
      mtl_commons = get_mtl_commons()
      mtl_commons['architecture'] = arch
      kwargs = {
        'arch_args': arch_args,
        'weight_args': config['defaults']['kwargs']['weight_args']
        }
      mtl_trainer = MTLTrainer(
          decoders=mtl_decoders,
          save_path=paths["training"]["architecture"],
          **mtl_commons,
          **kwargs
      )
      result = mtl_trainer.train(**mtl_training)
      results[arch] = result['best_result']['result']

      logger.info(results)

    data = pd.DataFrame(results)
    data.to_csv(paths['results']['architecture'] / iteration_name)

  # COMBINE EMB
  ################################
  elif experiment == 'combine-emb':
    results = {}
    dataframe = pd.read_csv(paths['new']['imputed-all'], header=[0, 1], index_col=0)
    dataframe = normalize(dataframe)
    stats_size = len(dataframe['STATS'].columns)
    
    embeddings = {}
    embedding_size = 0
    
    selected_embs = {model_name: constants['valid_embedding_sizes'][model_name] for model_name in experiment_args}
    for model_name, emb_size in selected_embs.items():
      emb = pd.read_csv(paths['new']['embeddings'][model_name], index_col=0)
      new_columns = pd.MultiIndex.from_product([['CLS'], emb.columns])
      emb = pd.DataFrame(emb.values, columns=new_columns, index=emb.index)
      embeddings[model_name] = emb
      embedding_size += emb_size
    
    emb_data = [dataframe] + [em for em in embeddings.values()]
    combined = pd.concat(emb_data, axis=1)
    print(combined.shape)

    stl_dataloaders = split_df(combined, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='stl')
    mtl_dataloaders = split_df(combined, config['split']['train'], config['split']['test'], config['dataloaders'], constants, rng, learning='mtl')
    predictor_size = embedding_size + stats_size

    class CombineEncoder(nn.Module):
      def __init__(self):
        super(CombineEncoder, self).__init__()
        dropout=config['encoder']['dropout']
        hidden=config['encoder']['nn']
        hidden[0]=predictor_size
        layers = []
        print(f'encoder {hidden}')
        for i in range(len(hidden) - 1):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            if i < len(hidden) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

      def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    stl_input_nn = [predictor_size]
    mtl_input_nn = [config['encoder']['nn'][-1]]

    stl_decoders = nn.ModuleDict({k: Decoder(stl_input_nn + config['stl-decoders']['hidden_nn'] + v[1], dropout=config['stl-decoders']['dropout'], final=v[0]) for k, v in finals.items()})
    mtl_decoders = {k: Decoder(mtl_input_nn + config['mtl-decoders']['hidden_nn'] + v[1], dropout=config['mtl-decoders']['dropout'], final=v[0]) for k, v in finals.items()}

    mtl_training = {
      'train_dataloaders': mtl_dataloaders["train"],
      'test_dataloaders': mtl_dataloaders["test"],
      'val_dataloaders': mtl_dataloaders["val"],
      'epochs': config["training"]["epochs"],
      'patience': config["training"]["patience"]
    }

    logger.info(stl_decoders)
    logger.info(mtl_decoders)

    stl_alg, stl_optim_arg = get_optimizer(config['optim_param'])
    stl_optimizers = {task: stl_alg(stl_decoders[task].parameters(), **stl_optim_arg) for task in constants["tasks"]}
    stl_schedulers = {k: get_scheduler(v, config['scheduler_param']) for k, v in stl_optimizers.items()}
    stl_loss_fns = get_stl_loss_fns()
    stl_metrics = get_stl_metrics()

    def train_stl(task:str, threshold) -> dict:
        result = train(
          stl_decoders[task], 
          stl_dataloaders[task], 
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

    kwargs = {
        'arch_args': config['defaults']['kwargs']['arch_args'],
        'weight_args': config['defaults']['kwargs']['weight_args']
        }
    mtl_commons = get_mtl_commons()
    mtl_commons['encoder_class'] = CombineEncoder
    mtl_trainer = MTLTrainer(
        decoders=mtl_decoders,
        save_path=paths["training"]["combine-emb"],
        **mtl_commons,
        **kwargs
    )




    results = {"STL": {}, "MTL": {}}
    
    logger.info('RUNNING STL')
    for task in constants["tasks"]:
      threshold = None if task == 'bigfive_s' else config['training']['threshold']
      result = train_stl(task, threshold)
      print(task, result)
      results["STL"][task] = result

    logger.info('\nRUNNING MTL')
    results["MTL"] = mtl_trainer.train(**mtl_training)

    print(results)

    data_dict = {'STL': {},'MTL': {}}
    for task in constants['tasks']:
      data_dict['STL'][task] = results['STL'][task]['best_val_metric']
      data_dict['MTL'][task] = results['MTL']['best_result']['result'][task][0]

    data = pd.DataFrame(data_dict)

    logger.info('RESULTS')
    logger.info(data)
    logger.info(data.head())

    data.to_csv(paths['results']['combine-emb'] / iteration_name)


  else:
    raise ValueError('experiment not valid')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate embeddings using selected encoder')
  parser.add_argument('--experiment', type=str, help='Name of the experiment to perform')
  args = parser.parse_args()

  main(args.experiment)
