# save_model.py
# Downloads dataset and trains model automatically on Render

import os
import pickle
import pandas as pd
import numpy as np
import re
import math
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')


# ── Download Dataset Automatically ───────────────────────────
print("Downloading dataset...")

url = "https://raw.githubusercontent.com/varma916/reddit-news-recommender/main/worldnews.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Dataset loaded from URL! Shape: {df.shape}")
except:
    print("URL failed — trying backup...")
    url2 = "https://raw.githubusercontent.com/rmisra/news-category-dataset/master/News_Category_Dataset_v3.json"
    df   = pd.read_json(url2, lines=True)
    df   = df.rename(columns={
        'headline'  : 'title',
        'category'  : 'subreddit',
        'date'      : 'date_created'
    })
    df['score']        = np.random.randint(1, 5000, size=len(df))
    df['num_comments'] = np.random.randint(0, 1000, size=len(df))
    df['timestamp']    = pd.to_datetime(
        df['date_created']).astype(np.int64) // 10**9
    print(f"✅ Backup dataset loaded! Shape: {df.shape}")

# ── Rename Columns ────────────────────────────────────────────
df = df.rename(columns={
    'up_votes'    : 'score',
    'down_votes'  : 'num_comments',
    'time_created': 'timestamp'
})

# Fill missing columns if needed
if 'score' not in df.columns:
    df['score'] = np.random.randint(1, 5000, size=len(df))
if 'num_comments' not in df.columns:
    df['num_comments'] = np.random.randint(0, 1000, size=len(df))
if 'timestamp' not in df.columns:
    df['timestamp'] = 1000000000

# Sample 20000 rows
if len(df) > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)

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
    max_features = 5000,
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95,
    stop_words   = 'english'
)
tfidf_matrix = tfidf.fit_transform(df['clean_title'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")


# ── Build Cosine Similarity Matrix ────────────────────────────
print("Building cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine Similarity Shape: {cosine_sim.shape}")


# ── Build SVD Matrix ──────────────────────────────────────────
print("Building SVD matrix...")
svd            = TruncatedSVD(n_components=100, random_state=42)
svd_matrix     = svd.fit_transform(tfidf_matrix)
svd_cosine_sim = cosine_similarity(svd_matrix, svd_matrix)
print(f"SVD Matrix Shape: {svd_matrix.shape}")
print(f"SVD Variance Captured: {svd.explained_variance_ratio_.sum()*100:.1f}%")


# ── Save All Models ───────────────────────────────────────────
print("\nSaving all models...")
save_path = os.path.dirname(os.path.abspath(__file__))

df.to_pickle(os.path.join(save_path, 'df.pkl'))

with open(os.path.join(save_path, 'cosine_sim.pkl'), 'wb') as f:
    pickle.dump(cosine_sim, f)

with open(os.path.join(save_path, 'svd_cosine_sim.pkl'), 'wb') as f:
    pickle.dump(svd_cosine_sim, f)

with open(os.path.join(save_path, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

print("\n✅ All models saved successfully!")
print(f"Saved to: {save_path}")
print("\nFiles saved:")
for f in os.listdir(save_path):
    print(f"  {f}")