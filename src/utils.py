import torch
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple

def get_paths() -> dict:
    #Print directory this is in (full path)
    _project = Path('/cluster') / 'work' / 'jacobaal' / 'prosjekt_oppgave'
    #this is giga hacky
    _data = _project / 'data'
    _raw = _data / 'raw'
    _checkpoints = _project / 'checkpoints'
    _configs = _project / 'configs'
    _glove = _data / 'glove'
    _large = _data / 'large'
    _small = _data / 'small'
    _final = _data / 'final'
    _filled = _data / 'filled'
    _split = _data / 'split'
    _new = _data / 'new'
    _new_emb = _new / 'embeddings'
    _old_emb = _new / 'old_embeddings'
    _new_agg = _new / 'aggregated'
    _new_split = _new / 'split'
    _img = _project / 'img'
    _outputs = _project / 'outputs'
    _results = _project / 'results'
    model_names = ['distilbert', 'deberta', 'mdeberta', 'tw-roberta', 'xlnet'] 
    model_name = 'example_model'  

    paths = {
        'img': {
            'data': _img / 'data',
            'opt': _img / 'opt'
        }, 
        'db': _project / 'unified.db',
        'raw': {
            'essays': _raw / 'essays' / 'essays.csv',
            'kaggle_mbti': _raw / 'kaggle-mbti' / 'mbti_1.csv',
            'mypers': _raw / 'mypers' / 'mypersonality_final.csv',
            'pandora_comments': _raw / 'pandora' / 'all_comments_since_2015.csv',
            'pandora_authors': _raw / 'pandora' / 'author_profiles.csv',
            'tw_mbti': _raw / 'tw-mbti' / 'twitter_MBTI.csv',
            'merged': _raw / 'merged.csv',
            'small': _raw / 'reduced.csv'
        },
        'glove': {
            'params': _glove / 'glove' / 'glove.twitter.27B.50d.txt',
            'w2v_params': _glove / 'glove' / 'glove.w2v.twitter.27B.50d.txt'
        },
        'large': {
            'features': _large / 'features.csv',
            'cleaned': _large / 'cleaned.csv',
            'cleaned_text': _large / 'cleaned_text.csv',
            'partially_cleaned': _large / 'partially_cleaned.csv',
            'partially_cleaned_text': _large / 'partially_cleaned_text.csv',
            'embeddings': _large / model_name / f'{model_name}-test.csv',
            'aggregated': _large / 'aggregated.csv',
            'filled': _large / 'filled.csv'
        },
        'small': {
            'features': _small / 'features.csv',
            'cleaned': _small / 'cleaned.csv',
            'cleaned_text': _small / 'cleaned_text.csv',
            'partially_cleaned': _small / 'partially_cleaned.csv',
            'partially_cleaned_text': _small / 'partially_cleaned_text.csv',
            'embeddings': _small / model_name / f'{model_name}-test.csv',
            'aggregated': _small / 'aggregated.csv',
            'filled': _small / 'filled.csv'
        },
        'final': {
            'deberta': _final / 'deberta' / 'full.csv',
            'mdeberta': _final / 'mdeberta' / 'full.csv',
            'tw-roberta': _final / 'tw-roberta' / 'full.csv',
            'distilbert': _final / 'distilbert' / 'full.csv'
        },
        'filled': {
            'pandora': _filled / 'pandora.csv',
            'merged': _filled / 'merged.csv',
            'missing': _filled / 'missing.csv',
            'complete': _filled / 'complete.csv',
            'filled': _filled / 'filled.csv',
            'imputed': _filled / 'imputed.csv',
        },
        'split': {
            'deberta': {
                'mbti': _split / 'mbti.csv',
                'bigfive_c': _split / 'bigfive_c.csv',
                'bigfive_s': _split / 'bigfive_s.csv'
            }
        },
        'new': {
            'preprocessed': _new / 'preprocessed.csv',
            'unprocessed': _new / 'unprocessed.csv',
            'w-emoji': _new / 'w-emoji.csv',
            'no-emoji': _new / 'no-emoji.csv',
            'imputed': _new / 'imputed.csv',
            'imputed-all': _new / 'imputed-all.csv',
            'embeddings': {
                'xlnet': _new_emb / 'xlnet.csv',
                'deberta': _new_emb / 'deberta.csv',
                'mdeberta': _new_emb / 'mdeberta.csv',
                'tw-roberta': _new_emb / 'tw-roberta.csv',
                'distilbert': _new_emb / 'distilbert.csv',
                'acc': _new_emb / 'acc.csv'
            },
            'old_embeddings': {
                'xlnet': _old_emb / 'xlnet.csv',
                'deberta': _old_emb / 'deberta.csv',
                'mdeberta': _old_emb / 'mdeberta.csv',
                'tw-roberta': _old_emb / 'tw-roberta.csv',
                'distilbert': _old_emb / 'distilbert.csv',
                'acc': _old_emb / 'acc.csv'
            },
            'split': {
                'mbti': _new_split / 'mbti.csv',
                'bigfive_c': _new_split / 'bigfive_c.csv',
                'bigfive_s': _new_split / 'bigfive_s.csv'
            }
        },
        'training': {
            'checkpoints': _checkpoints,
            'stl': _checkpoints / 'stl',
            'normalization': _checkpoints / 'normalization',
            'embeddings': _checkpoints / 'embeddings',
            'decoder': _checkpoints / 'decoder',
            'learning_method': _checkpoints / 'learning_method',
            'weighting': _checkpoints / 'weighting',
            'architecture': _checkpoints / 'architecture',
            'imputation': _checkpoints / 'imputation',
            'combine-emb': _checkpoints / 'combine-emb',
            'opt': _checkpoints / 'opt'
        },
        'config': _configs,
        'results': {
            'normalization': _results / 'normalization',
            'embeddings': _results / 'embeddings',
            'decoder': _results / 'decoder',
            'learning_method': _results / 'learning_method',
            'weighting': _results / 'weighting',
            'architecture': _results / 'architecture',
            'imputation': _results / 'imputation',
            'combine-emb': _results / 'combine-emb',
            'opt': _results / 'opt',
            'mtl_all': _results / 'mtl_all',
        },
        'outputs': {
            'study': _outputs / 'study',
            'journal': _outputs / 'journal'
        }
    }

    for model in model_names:
        base = _split / Path(model)
        paths['split'][model] = {
                'mbti': base / 'mbti.csv',
                'bigfive_c': base / 'bigfive_c.csv',
                'bigfive_s': base / 'bigfive_s.csv'
            }

    return paths

def _load_config(configs_path: str, config_name='default') -> dict:
    path = str(configs_path / Path(f'{config_name}.yaml')) 
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def _create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Argument parser with YAML config file")
    parser.add_argument('--config', type=str, help='Path to a specific config file', default='default')
    return parser

def _setup_logger(log_file: str = 'arguments.log') -> logging.Logger:
    logger = logging.getLogger('ArgumentLogger')

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

def _log_arguments(logger: logging.Logger, config: dict):
    logger.info("Arguments:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

def _parse_args_with_config(configs_path:Path) -> dict:
    parser = _create_argparser()
    args = parser.parse_known_args()[0]
    config = _load_config(configs_path, config_name=args.config)
    return config

def get_constants():
  tasks = ['mbti', 'bigfive_c', 'bigfive_s']
  splits = ['train', 'test', 'val']
  mbti_columns = ['mbtiEXT', 'mbtiSEN', 'mbtiTHI', 'mbtiJUD']
  bigfive_c_columns = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
  bigfive_s_columns = ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']
  experiments = ['architecture', 'decoder', 'embeddings', 'combine-emb', 'imputation', 'learning_method', 'normalization', 'weighting']
  bigfive_columns = bigfive_c_columns + bigfive_s_columns
  label_columns = mbti_columns + bigfive_columns
  common_columns = ['AUTHOR', 'TEXT', 'SOURCE']
  all_columns = common_columns + label_columns

  pandora_columns = ['author', 'introverted', 'intuitive', 'thinking', 'perceiving', 'agreeableness', 'openness', 'conscientiousness', 'extraversion','neuroticism']
  embedding_sizes = {
    'distilbert': 768,
    'deberta': 768,
    'mdeberta': 768,        
    'tw-roberta': 768,
    'xlnet': 768,
  }
  valid_embedding_sizes = { #Obsolete
    'distilbert': 768,
    'deberta': 768,
    'mdeberta': 768,
    'tw-roberta': 768,
    'xlnet': 768,
    'acc': 768,
  }
  valid_models = ['deberta', 'mdeberta', 'tw-roberta', 'xlnet', 'acc']
  stats_columns = ['chars', 'uppercased', 'emojis', 'posts', 'duplicates', 'word_nonwords', 'nonword_words', 'nonword_spaces', 'space_punctuations', 'hashtags', 'urls', 'mentions']
  emb_columns = list(map(str, range(768)))
  weighting_methods = ['EW', 'GradNorm', 'MGDA', 'UW', 'DWA', 'GLS', 'GradDrop', 'IMTL', 'RLW', 'Aligned_MTL', 'DB_MTL']

  constants = {
      "tasks": tasks,
      "experiments": experiments,
      "splits": splits,
      "mbti_columns": mbti_columns,
      "bigfive_c_columns": bigfive_c_columns,
      "bigfive_s_columns": bigfive_s_columns,
      "bigfive_columns": bigfive_columns,
      "label_columns": label_columns,
      "target_columns": label_columns, # convenience
      "common_columns": common_columns,
      "all_columns": all_columns,
      "pandora_columns": pandora_columns,
      "embedding_sizes": embedding_sizes,
      "valid_embedding_sizes": valid_embedding_sizes,
      "stats_columns": stats_columns,
      "embedding_columns": emb_columns,
      "weighting_methods": weighting_methods,
      "columns": {
        "mbti": mbti_columns,
        "bigfive_c": bigfive_c_columns,
        "bigfive_s": bigfive_s_columns
      },
      "binary_map": {
          "y": 1,
          "n": 0
      },
      "MBTI_map": {
          'E': 1,
          'I': 0,
          'S': 1,
          'N': 0,
          'T': 1,
          'F': 0,
          'J': 1,
          'P': 0
      },
      "mbti_map": {
          'e': 1,
          'i': 0,
          's': 1,
          'n': 0,
          't': 1,
          'f': 0,
          'j': 1,
          'p': 0
      },
  }

  return constants

def find_device() -> torch.DeviceObjType:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"device: {device}")
  if device.type == 'cuda':
      print(torch.cuda.get_device_name(0))
      print('Memory Usage:')
      print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
      print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
  return device

def get_commons(log=True, config_name='default') -> Tuple[Dict, dict, logging.Logger, torch.DeviceObjType]:
    """
    returns:
    paths, constants, config, logger, device
    """
    paths = get_paths()
    constants = get_constants()
    config = _load_config(paths['config'], config_name=config_name) # _parse_args_with_config(paths['configs'])
    logger = _setup_logger()
    device = find_device()

    if log: _log_arguments(logger, config)

    return paths, constants, config, logger, device
