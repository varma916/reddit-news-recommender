# main.py — FastAPI Backend
# Loads saved model and serves recommendations via API

import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Load saved models ─────────────────────────────────────────
MODEL_PATH = r'C:\Users\91994\Vcube\RedditRecommender'

print("Loading models...")
df             = pd.read_pickle(os.path.join(MODEL_PATH, 'df.pkl'))
cosine_sim     = pickle.load(open(os.path.join(MODEL_PATH, 'cosine_sim.pkl'),     'rb'))
svd_cosine_sim = pickle.load(open(os.path.join(MODEL_PATH, 'svd_cosine_sim.pkl'), 'rb'))
print("✅ Models loaded!")

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Reddit News Recommendation API",
    description="Recommends Reddit news posts using TF-IDF, SVD and Hybrid approaches",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Request Models ────────────────────────────────────────────
class RecommendRequest(BaseModel):
    post_title : str
    method     : str = "hybrid"   # tfidf, svd, popularity, hybrid
    top_n      : int = 10
    subreddit  : Optional[str] = None


# ── Helper Functions ──────────────────────────────────────────
def content_based_recommend(post_title, top_n=10):
    matches = df[df['title'].str.contains(
        post_title, case=False, na=False)]
    if matches.empty:
        return []
    idx        = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices    = [i[0] for i in sim_scores]
    similarity = [i[1] for i in sim_scores]
    result     = df.iloc[indices][['title','subreddit',
                                    'score','num_comments']].copy()
    result['score_val'] = similarity
    result['method']    = 'TF-IDF Content'
    return result.to_dict('records')

def svd_based_recommend(post_title, top_n=10):
    matches = df[df['title'].str.contains(
        post_title, case=False, na=False)]
    if matches.empty:
        return []
    idx        = matches.index[0]
    sim_scores = list(enumerate(svd_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices    = [i[0] for i in sim_scores]
    similarity = [i[1] for i in sim_scores]
    result     = df.iloc[indices][['title','subreddit',
                                    'score','num_comments']].copy()
    result['score_val'] = similarity
    result['method']    = 'SVD'
    return result.to_dict('records')

def popularity_based_recommend(subreddit=None, top_n=10):
    filtered = df if not subreddit else df[
        df['subreddit'].str.lower() == subreddit.lower()]
    if filtered.empty:
        filtered = df
    result = filtered.nlargest(top_n, 'hot_score_norm')[
        ['title','subreddit','score',
         'num_comments','hot_score_norm']].copy()
    result['score_val'] = result['hot_score_norm']
    result['method']    = 'Popularity'
    return result.to_dict('records')

def hybrid_recommend(post_title, top_n=10):
    matches = df[df['title'].str.contains(
        post_title, case=False, na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]

    tfidf_scores = list(enumerate(cosine_sim[idx]))
    tfidf_scores = sorted(tfidf_scores,
                          key=lambda x: x[1], reverse=True)[1:]
    svd_scores   = list(enumerate(svd_cosine_sim[idx]))
    svd_scores   = sorted(svd_scores,
                          key=lambda x: x[1], reverse=True)[1:]

    indices      = [i[0] for i in tfidf_scores]
    tfidf_vals   = [i[1] for i in tfidf_scores]
    svd_vals_map = {i[0]: i[1] for i in svd_scores}

    result = df.iloc[indices][
        ['title','subreddit','score',
         'num_comments','hot_score_norm']].copy()
    result['content_score']    = tfidf_vals
    result['svd_score']        = [svd_vals_map.get(i, 0) for i in indices]
    result['popularity_score'] = result['hot_score_norm']
    result['hybrid_score']     = (0.4 * result['content_score'] +
                                   0.3 * result['svd_score'] +
                                   0.3 * result['popularity_score'])
    result = result.sort_values(
        'hybrid_score', ascending=False).head(top_n)
    result['score_val'] = result['hybrid_score']
    result['method']    = 'Hybrid'
    return result[['title','subreddit','score',
                   'num_comments','score_val','method']].to_dict('records')


# ── API Routes ────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message"  : "Reddit News Recommendation API is running!",
        "endpoints": ["/recommend", "/categories", "/popular", "/stats"]
    }

@app.post("/recommend")
def get_recommendations(request: RecommendRequest):
    method = request.method.lower()
    if method == "tfidf":
        results = content_based_recommend(
            request.post_title, request.top_n)
    elif method == "svd":
        results = svd_based_recommend(
            request.post_title, request.top_n)
    elif method == "popularity":
        results = popularity_based_recommend(
            request.subreddit, request.top_n)
    else:
        results = hybrid_recommend(
            request.post_title, request.top_n)

    return {
        "method"         : method,
        "query"          : request.post_title,
        "total_results"  : len(results),
        "recommendations": results
    }

@app.get("/categories")
def get_categories():
    counts = df['subreddit'].value_counts().to_dict()
    return {"categories": counts}

@app.get("/popular")
def get_popular(subreddit: Optional[str] = None, top_n: int = 10):
    results = popularity_based_recommend(subreddit, top_n)
    return {
        "subreddit"      : subreddit or "all",
        "total_results"  : len(results),
        "recommendations": results
    }

@app.get("/stats")
def get_stats():
    return {
        "total_posts"   : len(df),
        "categories"    : df['subreddit'].nunique(),
        "avg_score"     : round(df['score'].mean(), 2),
        "avg_comments"  : round(df['num_comments'].mean(), 2),
        "date_range"    : {
            "from": str(df['date_created'].min()),
            "to"  : str(df['date_created'].max())
        }
    }