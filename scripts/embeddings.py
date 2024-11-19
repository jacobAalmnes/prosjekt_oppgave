import numpy as np
import argparse
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import XLNetTokenizerFast, XLNetModel
from transformers import AutoTokenizer, AutoModel
import sys

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path:
    sys.path.append(str(proj_path))

from src.utils import get_commons

class EmbedderDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None, padding=None, truncation=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['TEXT']
        author = row['AUTHOR']
        encoding = self.tokenizer(text, padding=self.padding, max_length=self.max_length, truncation=self.truncation, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}, author

def initialize_model_and_tokenizer(config, device, model_name):
    model_map = {
        'distilbert': ('distilbert-base-uncased', DistilBertModel, DistilBertTokenizerFast),
        'xlnet': ('xlnet/xlnet-base-cased', XLNetModel, XLNetTokenizerFast),
        'tw-roberta': ('cardiffnlp/twitter-roberta-base', AutoModel, AutoTokenizer),
        'deberta': ('microsoft/deberta-v3-base', AutoModel, AutoTokenizer),
        'mdeberta': ('microsoft/mdeberta-v3-base', AutoModel, AutoTokenizer),
    }

    if model_name not in model_map:
        raise ValueError(f'Unsupported model_name: {model_name}. Available options are {list(model_map.keys())}')

    model_path, model_class, tokenizer_class = model_map[model_name]
    model = model_class.from_pretrained(model_path).to(device)
    tokenizer = tokenizer_class.from_pretrained(model_path)
    
    return model, tokenizer

def generate_embeddings_aggregated(model, tokenizer, dataloader, device, non_blocking):
    model.eval()
    
    author_embeddings = {}
    author_counts = {}
    
    for batch, authors in dataloader:
        inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        for embedding, author in zip(batch_embeddings, authors):
            if author not in author_embeddings:
                author_embeddings[author] = embedding
                author_counts[author] = 1
            else:
                author_embeddings[author] += embedding
                author_counts[author] += 1

    aggregated_embeddings = {author: embedding / author_counts[author] for author, embedding in author_embeddings.items()}
    return aggregated_embeddings

def main(model_name):
    paths, constants, config, logger, device = get_commons(config_name='new')
    model, tokenizer = initialize_model_and_tokenizer(config, device, model_name)
    torch.set_float32_matmul_precision('high')
    logger.info(model_name)
    if model_name != 'xlnet' and torch.cuda.get_device_capability() in {(7, 0), (8, 0), (9, 0)}:
        model = torch.compile(model)
        logger.info('Compiled model')
    
    logger.info('Loading dataset..')
    data_path = paths['new']['w-emoji'] if model_name == 'tw-roberta' else paths['new']['no-emoji']
    data = pd.read_csv(data_path, header=[0, 1], index_col=0)
    data.columns = data.columns.droplevel(0) 
    data = data[['TEXT', 'AUTHOR']].fillna("").astype(str)
    data = data.sort_values(by=['AUTHOR'])

    
    dataset = EmbedderDataset(data, tokenizer, **config['embeddings']['tokenizer'])
    dataloader = DataLoader(dataset, **config['embeddings']['dataloader'])
    
    logger.info('Generating aggregated embeddings..')
    aggregated_embeddings = generate_embeddings_aggregated(model, tokenizer, dataloader, device, config['embeddings']['non_blocking'])
    
    aggregated_df = pd.DataFrame.from_dict(aggregated_embeddings, orient='index')
    aggregated_df.to_csv(paths["new"]["embeddings"][model_name])

    logger.info(f"Saved aggregated embeddings at {paths['new']['embeddings'][model_name]}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings using selected encoder')
    parser.add_argument('--model_name', type=str, help='Name of the model to use for embeddings')
    args = parser.parse_args()

    main(args.model_name)
