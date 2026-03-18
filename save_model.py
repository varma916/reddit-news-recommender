# save_model.py
# Downloads dataset and trains model automatically on Render
# Optimized for Render free tier (512MB RAM)

import os
import pickle
import pandas as pd
import numpy as np
import re
import math
import warnings
import urllib.request
import zipfile
import io
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')


# ── Download Dataset Automatically ───────────────────────────
print("Downloading dataset...")

try:
    url      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
    response = urllib.request.urlopen(url)
    zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
    zip_data.extractall('/tmp/')

    df = pd.read_csv('/tmp/newsCorpora.csv', sep='\t', header=None,
                     names=['id','title','url','publisher',
                            'category','story','hostname','timestamp'])

    df['subreddit'] = df['category'].map({
        'b': 'business',
        't': 'technology',
        'e': 'entertainment',
        'm': 'health'
    })

    np.random.seed(42)
    df['score']        = np.random.randint(1, 5000, size=len(df))
    df['num_comments'] = np.random.randint(0, 1000, size=len(df))
    df['timestamp']    = np.random.randint(
        1000000000, 1500000000, size=len(df))

    print(f"✅ Dataset loaded! Shape: {df.shape}")

except Exception as e:
    print(f"Dataset download failed: {e}")
    raise


# ── Fill Missing Columns ──────────────────────────────────────
if 'score' not in df.columns:
    df['score'] = np.random.randint(1, 5000, size=len(df))
if 'num_comments' not in df.columns:
    df['num_comments'] = np.random.randint(0, 1000, size=len(df))
if 'timestamp' not in df.columns:
    df['timestamp'] = 1000000000
if 'subreddit' not in df.columns:
    df['subreddit'] = 'general'


# ── Sample ONLY 3000 Rows for RAM efficiency ─────────────────
# Free tier has 512MB RAM
# 3000 rows cosine matrix = 3000x3000 = ~72MB — safe!
if len(df) > 3000:
    df = df.sample(n=3000, random_state=42).reset_index(drop=True)

print(f"Dataset shape after sampling: {df.shape}")


# ── Assign Categories ─────────────────────────────────────────
def assign_category(title):
    title = str(title).lower()
    if any(w in title for w in ['war','attack','kill','bomb',
                                  'military','troops','missile',
                                  'shooting','conflict','terror',
                                  'soldier','army','weapon']):
        return 'war_conflict'
    elif any(w in title for w in ['election','president','minister',
                                   'government','parliament','vote',
                                   'political','senate','congress',
                                   'democrat','republican','party']):
        return 'politics'
    elif any(w in title for w in ['economy','market','trade','bank',
                                   'stock','gdp','inflation','tax',
                                   'debt','financial','oil','price',
                                   'currency','economic']):
        return 'economy'
    elif any(w in title for w in ['virus','health','disease','hospital',
                                   'doctor','vaccine','cancer','covid',
                                   'pandemic','drug','medical','death']):
        return 'health'
    elif any(w in title for w in ['climate','environment','flood',
                                   'earthquake','storm','wildfire',
                                   'pollution','carbon','emission',
                                   'disaster','weather','hurricane']):
        return 'environment'
    elif any(w in title for w in ['technology','cyber','hack','ai',
                                   'internet','data','software','robot',
                                   'tech','digital','computer','space']):
        return 'technology'
    elif any(w in title for w in ['sport','cricket','football','soccer',
                                   'tennis','basketball','olympic',
                                   'championship','tournament','league',
                                   'match','player','coach','team']):
        return 'sports'
    else:
        return 'general'

df['subreddit'] = df['title'].apply(assign_category)
print("Categories assigned!")
print(df['subreddit'].value_counts())


# ── Clean Text ────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_title'] = df['title'].apply(clean_text)
df = df[df['clean_title'].str.strip() != ''].reset_index(drop=True)
print("Text cleaning done!")


# ── Feature Engineering ───────────────────────────────────────
scaler              = MinMaxScaler()
df['score_norm']    = scaler.fit_transform(df[['score']])
df['comments_norm'] = scaler.fit_transform(df[['num_comments']])
df['engagement_score'] = (0.7 * df['score_norm']) + \
                         (0.3 * df['comments_norm'])

def reddit_hot_score(score, timestamp):
    order = math.log10(max(abs(score), 1))
    sign  = 1 if score > 0 else -1 if score < 0 else 0
    return round(sign * order + timestamp / 45000000000, 7)

df['hot_score']      = df.apply(
    lambda x: reddit_hot_score(x['score'], x['timestamp']), axis=1)
df['hot_score_norm'] = scaler.fit_transform(df[['hot_score']])
print("Feature engineering done!")


# ── Build TF-IDF Matrix ───────────────────────────────────────
print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    max_features = 3000,
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95,
    stop_words   = 'english'
)
tfidf_matrix = tfidf.fit_transform(df['clean_title'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")


# ── Build SVD Matrix ──────────────────────────────────────────
# SVD reduces dimensions — much smaller than full cosine matrix
print("Building SVD matrix...")
n_components   = 50
svd            = TruncatedSVD(n_components=n_components,
                               random_state=42)
svd_matrix     = svd.fit_transform(tfidf_matrix)
print(f"SVD Matrix Shape: {svd_matrix.shape}")
print(f"SVD Variance Captured: {svd.explained_variance_ratio_.sum()*100:.1f}%")


# ── Save All Models ───────────────────────────────────────────
# We save tfidf_matrix as sparse (small file)
# We do NOT save full cosine_sim matrix (too large)
# Instead main.py computes similarity on demand for each query

print("\nSaving all models...")
save_path = os.path.dirname(os.path.abspath(__file__))

# Save dataframe
df.to_pickle(os.path.join(save_path, 'df.pkl'))

# Save tfidf vectorizer
with open(os.path.join(save_path, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

# Save sparse tfidf matrix (very small file)
sp.save_npz(os.path.join(save_path, 'tfidf_matrix.npz'),
            tfidf_matrix)

# Save SVD matrix (small — only 3000 x 50)
np.save(os.path.join(save_path, 'svd_matrix.npy'), svd_matrix)

print("\n✅ All models saved successfully!")
print(f"Saved to: {save_path}")
print("\nFiles saved:")
for f in os.listdir(save_path):
    print(f"  {f}")