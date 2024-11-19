import argparse
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():

  # Arguments
  parser = argparse.ArgumentParser(description="Aggregate embeddings for each author")
  parser.add_argument('--model_name', type=str, default=None, help="Name of model whos embeddings will be aggregated")
  parser.add_argument('--test', type=str, default='test', help="If full, use full embeddings")
  args = parser.parse_args()

  for arg, value in sorted(vars(args).items()):
    logger.info(f'Argument {arg}: {value}')

  valid_model_name = ['distilbert', 'tw-roberta', 'xlnet', 'deberta', 'mdeberta']
  if args.model_name not in valid_model_name: raise ValueError('model_name must be one of: distilbert, tw-roberta, xlnet, deberta, mdeberta')
  
  # Paths
  path_data_dir = Path('data')
  path_reduced = path_data_dir / Path('reduced')
  path_data = path_data_dir / Path('merged.csv')
  path_features = path_reduced / Path('features.csv')
  path_embeddings = path_reduced / Path(args.model_name) / Path(f'{args.model_name}-{args.test}.csv')
  path_aggregated = path_data_dir / Path('aggregated') / Path(args.model_name) / Path(f'{args.test}.csv')
  
  # DataFrame
  authors = pd.read_csv(path_data, usecols=['AUTHOR'])
  embeddings = pd.read_csv(path_embeddings)
  logger.info(f'Shape authors: {authors.shape}')
  logger.info(f'Shape embeddings: {embeddings.shape}')
  data = pd.merge(authors, embeddings, left_index=True, right_index=True, copy=False).groupby('AUTHOR').mean().reset_index()
  data.to_csv(path_aggregated, index=False)

  logger.info(f'Aggregated saved to {path_aggregated}')

  final_df = pd.read_csv(path_features)
  


if __name__ == '__main__':
  main()