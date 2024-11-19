import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path: sys.path.append(str(proj_path))

from src.utils import get_commons
import logging
import re
import logging
import emoji
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# def get_essays(paths, constants) -> pd.DataFrame:
#     essays = pd.read_csv(paths['raw']['essays'], encoding='ISO-8859-1')
#     for col in constants['bigfive_c_columns']: essays[col] = essays[col].map(constants['binary_map'])
#     essays = essays.rename({
#     '#AUTHID': 'AUTHOR',
#     },
#     axis='columns')
#     return essays

def get_kaggle_mbti(paths, constants) -> pd.DataFrame:
    kaggle_mbti = pd.read_csv(paths['raw']['kaggle_mbti'])
    kaggle_mbti['mbtiEXT'] = kaggle_mbti['type'].str[0]
    kaggle_mbti['mbtiSEN'] = kaggle_mbti['type'].str[1]
    kaggle_mbti['mbtiTHI'] = kaggle_mbti['type'].str[2]
    kaggle_mbti['mbtiJUD'] = kaggle_mbti['type'].str[3]
    for col in constants['mbti_columns']: kaggle_mbti[col] = kaggle_mbti[col].map(constants['MBTI_map'])
    kaggle_mbti = kaggle_mbti.drop('type', axis='columns')
    kaggle_mbti['posts'] = kaggle_mbti['posts'].str.split('\|\|\|')
    kaggle_mbti = kaggle_mbti.explode('posts').reset_index()
    kaggle_mbti = kaggle_mbti.rename({
    'posts': 'TEXT',
    'index': 'AUTHOR'
    }, axis='columns')
    return kaggle_mbti

def get_mypers(paths, constants) -> pd.DataFrame:
    mypers = pd.read_csv(paths['raw']['mypers'], encoding='ISO-8859-1', usecols=['#AUTHID',	'STATUS'] + constants['bigfive_columns'])
    mypers = mypers.rename({
        '#AUTHID': 'AUTHOR',
        'STATUS': 'TEXT',
    }, axis='columns')
    for col in constants['bigfive_c_columns']:
        mypers[col] = mypers[col].map(constants['binary_map'])
    
    # Normalize to percentiles
    author_scores = mypers.groupby('AUTHOR')[constants['bigfive_s_columns']].mean().reset_index()
    author_percentiles = author_scores[constants['bigfive_s_columns']].rank(pct=True) * 100
    author_percentiles['AUTHOR'] = author_scores['AUTHOR']
    mypers = mypers.drop(constants['bigfive_s_columns'], axis='columns').merge(author_percentiles, on='AUTHOR', copy=False)
    return mypers

def get_tw_mbti(paths, constants) -> pd.DataFrame:
    tw_mbti = pd.read_csv(paths['raw']['tw_mbti'])
    tw_mbti['mbtiEXT'] = tw_mbti['label'].str[0]
    tw_mbti['mbtiSEN'] = tw_mbti['label'].str[1]
    tw_mbti['mbtiTHI'] = tw_mbti['label'].str[2]
    tw_mbti['mbtiJUD'] = tw_mbti['label'].str[3]

    for col in constants['mbti_columns']: tw_mbti[col] = tw_mbti[col].map(constants['mbti_map'])
    tw_mbti = tw_mbti.drop('label', axis='columns')
    tw_mbti = tw_mbti.rename({
    'Unnamed: 0': 'AUTHOR',
    'text': 'TEXT'
    }, axis='columns')
    tw_mbti['TEXT'] = tw_mbti['TEXT'].str.split('\|\|\|')
    tw_mbti = tw_mbti.explode('TEXT').reset_index(drop=True)  
    return tw_mbti

def get_pandora(paths, constants) -> pd.DataFrame:
    pandora_authors = pd.read_csv(paths['raw']['pandora_authors'], usecols=constants['pandora_columns'])
    pandora_comments = pd.read_csv(paths['raw']['pandora_comments']) # usecols=['author', 'body']
    pandora_authors = pandora_authors.rename({
    'author': 'AUTHOR', 
    'introverted': 'mbtiEXT', # Flip
    'intuitive': 'mbtiSEN', # Flip
    'thinking': 'mbtiTHI', 
    'perceiving': 'mbtiJUD', # Flip 
    'agreeableness': 'sAGR', 
    'openness': 'sOPN', 
    'conscientiousness': 'sCON', 
    'extraversion': 'sEXT',
    'neuroticism': 'sNEU'
    },
    axis ='columns')
    pandora_authors[['mbtiEXT', 'mbtiSEN', 'mbtiJUD']] = 1 - pandora_authors[['mbtiEXT', 'mbtiSEN', 'mbtiJUD']]
    pandora_comments = pandora_comments.rename({
    'author': 'AUTHOR',
    'body': 'TEXT'
    },
    axis='columns')
    pandora = pd.merge(pandora_authors, pandora_comments, on='AUTHOR', copy=False)
    return pandora

def reduce_data(data:pd.DataFrame, generator:np.random.Generator):
        mbti_mask = data[(data['SOURCE'] == 'pandora') & 
                        (data[mbti_columns].notna().all(axis=1)) & 
                        (data[bigfive_columns].isnull().all(axis=1))]

        authors_to_remove = mbti_mask['AUTHOR'].unique()
        num_rows_to_remove = int(len(authors_to_remove) * (1 - mbti_frac))
        selected_authors = generator.choice(authors_to_remove, size=num_rows_to_remove, replace=False)
        data = data[~data['AUTHOR'].isin(selected_authors)]

        print(len(data))
        print(len(data[common_columns + mbti_columns].dropna(axis=0)))
        print(len(data[common_columns + bigfive_c_columns].dropna(axis=0)))
        print(len(data[common_columns + bigfive_s_columns].dropna(axis=0)))

        data = data.reset_index()
        return data

def process_data(data:pd.DataFrame, remove_social=True) -> pd.DataFrame:
    data['chars'] = data['TEXT'].str.len()
    logger.debug("Counted chars")
    data['uppercased'] = data['TEXT'].str.count(r'[A-Z]')
    logger.debug("Counted uppercased")
    data['emojis'] = data['TEXT'].apply(emoji.emoji_count)
    logger.debug("Counted emojis")
    data['posts'] = data['AUTHOR'].map(data['AUTHOR'].value_counts())
    logger.debug("Counted posts")

    regex_duplicate = re.compile(r'(.)\1{2,}')  # Matches characters repeated more than twice
    regex_word_nonword = re.compile(r'(\w)([^\w\s])')  # Matches a word character followed by a non-word character
    regex_nonword_word = re.compile(r'([^\w\s])(\w)')  # Matches a non-word character followed by a word character
    regex_nonword_space = re.compile(r'(\W)\s+(\W)')  # Matches a non-word character followed by spaces and another non-word character
    regex_space_punctuation = re.compile(r'\s+([,.!?])')  # Matches spaces followed by punctuation marks
    regex_hashtag = re.compile(r'#\w+')  # Matches hashtags
    regex_url = re.compile(r'http\S+')  # Matches URLs
    regex_mention = re.compile(r'@\w+')  # Matches mentions

    def process_text(text:str) -> str: 
        
        matches_url = len(regex_url.findall(text))
        text = regex_url.sub('URL', text)
        matches_mention = len(regex_mention.findall(text))
        text = regex_mention.sub('MENTION', text)  
        
        matches_hashtag = len(regex_hashtag.findall(text))
        matches_duplicate = len(regex_duplicate.findall(text))
        matches_word_nonword = len(regex_word_nonword.findall(text))
        matches_nonword_word = len(regex_nonword_word.findall(text))
        matches_nonword_space = len(regex_nonword_space.findall(text))
        matches_space_punctuation = len(regex_space_punctuation.findall(text))

        unsocial = regex_duplicate.sub(r'\1\1\1', text) # Tandem duplicated
        # text = regex_word_nonword.sub(r'\1 \2', text) BAD 
        # unsocial = regex_nonword_word.sub(r'\1 \2', unsocial)  
        # unsocial = regex_nonword_space.sub(r'\1\2', unsocial) 
        # unsocial = regex_space_punctuation.sub(r'\1', unsocial) 
        # text = regex_hashtag.sub('HASHTAG', text) BAD
        unsocial = emoji.demojize(unsocial) 


        return text, unsocial, {
            'duplicates': matches_duplicate,
            'word_nonwords': matches_word_nonword,
            'nonword_words': matches_nonword_word,
            'nonword_spaces': matches_nonword_space,
            'space_punctuations': matches_space_punctuation,
            'hashtags': matches_hashtag,
            'urls': matches_url,
            'mentions': matches_mention
        }

    def process_and_extract(text:str) -> pd.Series:
        social, unsocial, match_counts = process_text(text)
        return pd.Series(
            data=[social] + [unsocial] + list(match_counts.values()), 
            index=['social'] + ['unsocial'] + list(match_counts.keys())
            )

    logger.debug("Applying process and extract..")
    results = data['TEXT'].apply(process_and_extract)
    logger.debug("Applied process and extract..")

    social_df = results.drop(columns='unsocial').rename(columns={'social': 'TEXT'})
    unsocial_df = results.drop(columns='social').rename(columns={'unsocial': 'TEXT'})

    social_df = pd.concat([data.drop(columns=['TEXT']), social_df], axis=1)
    unsocial_df = pd.concat([data.drop(columns=['TEXT']), unsocial_df], axis=1)
    logger.debug("Assigned results")

    return social_df, unsocial_df

def multiindex(data:pd.DataFrame, constants:dict) -> pd.DataFrame:
    author_list = [('AUTHOR', 'AUTHOR')]
    # emb_list = [('CLS', emb) for emb in constants["embedding_columns"]]
    text_list = [('TEXT', 'TEXT')]
    stat_list = [('STATS', stat) for stat in constants['stats_columns']]
    target_list = [('TARGET', target) for target in constants["target_columns"]]
    tuples = author_list + stat_list + target_list + text_list # + emb_list
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['GROUP', 'FEATURE'])
    data_multiindexed = pd.DataFrame(columns=multiindex)
    data_multiindexed['AUTHOR'] = data['AUTHOR']
    data_multiindexed['STATS'] = data[constants['stats_columns']]
    data_multiindexed['TEXT'] = data['TEXT']
    # data_multiindexed['CLS'] = data[constants['embedding_columns']]
    data_multiindexed['TARGET'] = data[constants['target_columns']]
    return data_multiindexed

def main():
    paths, constants, config, _, device = get_commons()
    rng = np.random.default_rng(seed=config['seed'])
    logger.info('STARTING')

    #essays = get_essays(paths, constants)
    kaggle_mbti = get_kaggle_mbti(paths, constants)
    mypers = get_mypers(paths, constants)
    tw_mbti = get_tw_mbti(paths, constants)
    pandora = get_pandora(paths, constants)

    #essays['SOURCE'] = 'essays'
    kaggle_mbti['SOURCE'] = 'kaggle_mbti'
    mypers['SOURCE'] = 'mypers'
    pandora['SOURCE'] = 'pandora'
    tw_mbti['SOURCE'] = 'tw_mbti'
    
    #essays = essays.reindex(columns=constants["all_columns"], fill_value=None)
    kaggle_mbti = kaggle_mbti.reindex(columns=constants["all_columns"], fill_value=None)
    mypers = mypers.reindex(columns=constants["all_columns"], fill_value=None)
    pandora = pandora.reindex(columns=constants["all_columns"], fill_value=None)
    tw_mbti = tw_mbti.reindex(columns=constants["all_columns"], fill_value=None)
    datasets = [kaggle_mbti, mypers, pandora, tw_mbti]
    for dataset in datasets:
        logger.debug(dataset.shape)
    
    data = pd.concat(datasets, 
                    axis=0, 
                    ignore_index=True,
                    copy=False)
    
    logger.debug(data.shape)

    data['AUTHOR'] = data['AUTHOR'].apply(str)
    data['AUTHOR'] += data['SOURCE']
    data['AUTHOR'] = data['AUTHOR'].apply(hash)

    logger.debug(data.shape)

    # Sirsapalli and Malla (2023) found that removing posts with fewer than 3 words substancially improved performance
    data['TEXT'] = data['TEXT'].map(str)
    data = data.loc[data['TEXT'].str.split().str.len() > 2]

    logger.debug(data.shape)

    # data = reduce_data(data, rng)

    data = data.reset_index()

    data.to_csv(paths["new"]["unprocessed"])
    logger.info(f'Unprocessed saved at {paths["new"]["unprocessed"]}')

    social_df, unsocial_df = process_data(data)
    logger.info('Processed')

    social_df = multiindex(social_df, constants)
    unsocial_df = multiindex(unsocial_df, constants)
    logger.info('Multiindexed')

    social_df.to_csv(paths["new"]["w-emoji"])
    logger.info(f'Social saved at {paths["new"]["w-emoji"]}')

    # data[('TEXT', 'TEXT')] = data[('TEXT', 'TEXT')].apply(emoji.demojize)
    unsocial_df.to_csv(paths["new"]["no-emoji"])
    logger.info(f'Unsocial saved at {paths["new"]["no-emoji"]}')
    
    logger.info('FINISHED')
    
    return data


if __name__ == '__main__':
    main()
