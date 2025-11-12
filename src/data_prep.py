import pandas as pd
from datasets import load_dataset

def prepare_yelp(output_dir='data/'):
    ds = load_dataset('yelp_review_full')

    def map_label(stars):
        stars = stars + 1
        if stars <= 2:
            return 'negative'
        if stars == 3:
            return 'neutral'
        return 'positive'

    for split in ['train','test']:
        df = ds[split].to_pandas()
        df[['text','label']].to_csv(f'{output_dir}/raw/{split}.csv', index=False)
        df = df.rename(columns={'text':'text','label':'label_raw'})
        df['label'] = df['label_raw'].apply(map_label)
        df[['text','label']].to_csv(f'{output_dir}/processed/{split}.csv', index=False)

if __name__ == '__main__':
    prepare_yelp()