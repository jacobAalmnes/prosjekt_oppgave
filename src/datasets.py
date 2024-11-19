import logging
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import pandas as pd
import numpy as np

class SingleInputDataset(Dataset):
  def __init__(self, data:pd.DataFrame, target_col='TARGET', level=0):
    x_cols = ['STATS', 'CLS']
    
    print(f'dataset x shape: {data[x_cols].shape}')
    print(f'dataset y shape : {data[target_col].shape}')
    self.data_x = torch.Tensor(data[x_cols].values)
    self.data_y = torch.Tensor(data[target_col].values)

  def __len__(self) -> int:
    return self.data_y.shape[0]
  
  def __getitem__(self, index:int) -> torch.Tensor:
    return self.data_x[index], self.data_y[index]
    
def split_df(
    df:pd.DataFrame, 
    train_size:float, 
    test_size: float, 
    config:dict, 
    constants:dict, 
    generator:np.random.Generator, 
    learning:str, 
    subset:str='imp',
  ) -> dict:

  assert learning.lower() in ('stl', 'mtl')
  assert subset.lower() in ('org', 'imp', 'mix')
  
  datasets = {}
  for task in constants['tasks']:
    print(df.columns)
    print(df.shape)
    org_cols = [('ORG_TARGET', col) for col in constants["columns"][task]]
    imp_cols = [('IMP_TARGET', col) for col in constants["columns"][task]]
    org_targets = df.dropna(axis=0, subset=org_cols)
    org_length = len(org_targets)
    print(f'org length {org_length}')

    def get_org():
      train_length = int(train_size * org_length)
      test_length = int(test_size * org_length)
      val_length = org_length - (train_length + test_length)
      remaining = org_length - (train_length + test_length + val_length)
      print(task)
      print("Original length:", org_length)
      print("Train length:", train_length)
      print("Test length:", test_length)
      print("Validation length:", val_length)
      print(f"remaining: {remaining}")

      if remaining > 0: val_length += remaining

      print("Adjusted Train length:", train_length)
      print("Adjusted Test length:", test_length)
      print("Adjusted Validation length:", val_length)
      
      assert train_length + test_length + val_length == org_length, "Lengths do not add up to total length"
      
      dataset = SingleInputDataset(org_targets, target_col=org_cols)
      train, test, val = random_split(dataset, [train_length, test_length, val_length])
      return train, test, val
    
    def get_imp():
      total_length = len(df)
      train_length = int(train_size * total_length)
      test_length = int(test_size * total_length)

      # Test set is deterministic. Reduced to original data length
      if test_length > org_length:
        test_length = org_length
        val_length = total_length - (train_length + test_length)
        remaining = org_length - (train_length + test_length + val_length)
        if remaining > 0: val_length += remaining
        assert train_length + test_length + val_length == total_length, "Lengths do not add up to total length"
        
        test_rows = df.loc[org_targets.index].drop('ORG_TARGET', axis=1)
        train_val_rows = df.drop(org_targets.index, axis=0).drop('ORG_TARGET', axis=1)
        print(f'Is null: {test_rows["IMP_TARGET"].isnull().sum(axis=1).sum()}')
        print(f'Is null: {train_val_rows["IMP_TARGET"].isnull().sum(axis=1).sum()}')

        test = SingleInputDataset(test_rows, target_col=imp_cols)
        test_val = SingleInputDataset(train_val_rows, target_col=imp_cols)
        train, val = random_split(test_val, [train_length, val_length])
        return train, test, val

      # Test set randomly assigned from org data
      else:
        val_length = total_length - (train_length + test_length)
        remaining = org_length - (train_length + test_length + val_length) 
        if remaining > 0: val_length += remaining
        assert train_length + test_length + val_length == total_length, "Lengths do not add up to total length"
        
        chosen_indices = generator.choice(org_targets.index, size=test_length, replace=False)

        test_rows = df.loc[chosen_indices].drop('ORG_TARGET', axis=1)
        train_val_rows = df.drop(chosen_indices, axis=0).drop('ORG_TARGET', axis=1)
        print(f'Is null: {test_rows["IMP_TARGET"].isnull().sum(axis=1).sum()}')
        print(f'Is null: {train_val_rows["IMP_TARGET"].isnull().sum(axis=1).sum()}')

        test = SingleInputDataset(test_rows, target_col=imp_cols)
        test_val = SingleInputDataset(train_val_rows, target_col=imp_cols)
        train, val = random_split(test_val, [train_length, val_length])
        return train, test, val

    # ORG
    if subset == 'org':
      train, test, val = get_org()
    
    # IMP
    elif subset == 'imp':
      train, test, val = get_imp()
    
    elif subset == 'mix':
      org_train, test, val = get_org()
      imp = df.drop(org_targets.index, axis=0)
      print(org_cols)
      print(imp_cols)
      imp[org_cols] = imp[imp_cols] 
      imp_train = SingleInputDataset(imp, target_col=org_cols) # To be compatible with org
      
      train = ConcatDataset([org_train, imp_train])

    else:
      raise ValueError('Subset not recognized')

    train = DataLoader(train, **config['train'])
    test = DataLoader(test, **config['test'])
    val = DataLoader(val, **config['val'])

    datasets[task] = {
      'train': train,
      'test': test,
      'val': val
    }

  final_dict = {
    'train': {},
    'test': {},
    'val': {}
    }
  
  if learning == 'stl':
    return datasets
  
  elif learning == 'mtl':
    for task, split in datasets.items():
      for split, dataloader in split.items():
        final_dict[split][task] = dataloader
    return final_dict


def split_dataset(dataset: Dataset, train_size: float, test_size: float):
  total_length = len(dataset)
  train_length = int(train_size * total_length)
  test_length = int(test_size * total_length)
  val_length = total_length - (train_length + test_length)

  assert train_length + test_length + val_length == total_length, "Lengths do not add up to total length"
      
  train, test, val = random_split(dataset, [train_length, test_length, val_length])

  subsets = {
    'train': train,
    'test': test,
    'val': val
  }

  return subsets

def get_mtl_dataloaders(
    datasets:dict[Dataset], 
    train_size:float, 
    test_size:float, 
    args:dict, 
    logger=None,
    splits=['train', 'test', 'val'],
    tasks=['mbti', 'bigfive_c', 'bigfive_s']
    ):
    dataloaders = {split: {} for split in splits}
    for task in tasks:
        dataset = datasets[task]
        logger.debug(f'{task}: {torch.sum(torch.stack([label for _, label in dataset]), 0)}')
        subsets = split_dataset(dataset, train_size, test_size) 
        for split in splits:
            dataloaders[split][task] = DataLoader(dataset, **args[split])
    return dataloaders
  
def get_stl_dataloaders(
  dataset:Dataset, 
  train_size:float, 
  test_size:float, 
  args:dict, 
  
  logger=None
  ):
  splits = split_dataset(dataset, train_size, test_size)

  if logger:
    for k, ds in splits.items():
      logger.debug(f'{k}: {torch.sum(torch.stack([label for _, label in ds]), 0)}')

  return {split: DataLoader(subset, **args[split]) for split, subset in splits.items()}
