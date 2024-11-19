import numpy as np
import argparse
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import XLNetTokenizerFast, XLNetModel
from transformers import DebertaTokenizerFast
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info('Memory Usage:')
    logger.info(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    logger.info(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')


class EmbedderDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, padding, truncation, nrows):
        self.data = pd.read_csv(file_path, nrows=nrows, dtype=str)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['TEXT']

        if pd.isna(text):
            text = "" 
        else:
            text = str(text)

        encoding = self.tokenizer(text, padding=self.padding, max_length=self.max_length, truncation=self.truncation, return_tensors='pt') 
        return {key: val.squeeze(0) for key, val in encoding.items()}

def main(do_lower_case, padding_side, truncation_side, max_length, padding, truncation, batch_size, nrows, num_workers, non_blocking, pin_memory, model_name, output_name):

    path_data_dir = Path('data')
    path_preprocessed_dir = path_data_dir / Path('reduced')
    path_cleaned_text = path_preprocessed_dir / Path('cleaned_text.csv')
    path_partially_cleaned_text = path_preprocessed_dir / Path('partially_cleaned_text.csv')
    path_embeddings = path_preprocessed_dir / Path(model_name) / Path(output_name)

    tokenizer_args = {
      'do_lower_case': do_lower_case,
      'padding_side': padding_side,
      'truncation_side': truncation_side
    }
    
    if model_name == 'distilbert':
      model_path = 'distilbert-base-uncased'
      model = DistilBertModel.from_pretrained(model_path).to(device)
      tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, **tokenizer_args)
    elif model_name == 'xlnet':
      model_path = 'xlnet/xlnet-base-cased'
      model = XLNetModel.from_pretrained(model_path).to(device)
      tokenizer = XLNetTokenizerFast.from_pretrained(model_path, **tokenizer_args)
    elif model_name in ('tw-roberta', 'deberta', 'mdeberta'):
      if model_name == 'tw-roberta': model_path = 'cardiffnlp/twitter-roberta-base'
      if model_name == 'deberta': model_path = 'microsoft/deberta-v3-base'
      if model_name == 'mdeberta': model_path = 'microsoft/mdeberta-v3-base'
      model = AutoModel.from_pretrained(model_path).to(device)
      tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
    else:
      raise TypeError('argument model_name needs to be set. Available options are "distilbert", "xlnet" and "tw-roberta"')

    device_cap = torch.cuda.get_device_capability()
    supported_capabilities = {(7, 0), (8, 0), (9, 0)}
    if device_cap in supported_capabilities:
        model = torch.compile(model)

    model.eval()

    logger.info(f'Loading dataset..')
    
    path_text = path_partially_cleaned_text if model_name == 'tw-roberta' else path_cleaned_text
    dataset = EmbedderDataset(path_text, tokenizer, max_length, padding, truncation, nrows)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f'Generating embeddings..')
    embeddings = []

    for batch in dataloader:
        inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}  
        with torch.no_grad(): outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(path_embeddings, index=False)
    logger.info(f'Saved embeddings at {path_embeddings}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings using selected encoder')
    parser.add_argument('--do_lower_case', type=bool, default=True, help='Whether to lowercase the input text')
    parser.add_argument('--padding_side', type=str, default='right', help='Padding side for the input text')
    parser.add_argument('--truncation_side', type=str, default='right', help='Truncation side for the input text')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the input text')
    parser.add_argument('--padding', type=str, default='max_length', help='Padding strategy for the input text')
    parser.add_argument('--truncation', type=bool, default=True, help='Whether to truncate the input text')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to load from the dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for the dataloader')
    parser.add_argument('--non_blocking', type=bool, default=False, help='Non blocking transfer to device. Enable if using multiple')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory for the dataloader')
    parser.add_argument('--model_name', type=str, default=None, help='Which model to use for generating embeddings. Must be one of: "distilbert", "xlnet" or "tw-roberta"')
    parser.add_argument('--output_name', type=str, default='default.csv', help='Name of embedding file')
    args = parser.parse_args()

    print('\n ---- Arguments ---')
    for arg, value in sorted(vars(args).items()):
        logger.info(f"Argument {arg}: {value}")

    main(
        args.do_lower_case,
        args.padding_side,
        args.truncation_side,
        args.max_length,
        args.padding,
        args.truncation,
        args.batch_size,
        args.nrows,
        args.num_workers,
        args.non_blocking,
        args.pin_memory,
        args.model_name,
        args.output_name
    )
