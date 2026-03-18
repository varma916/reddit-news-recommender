# main.py — FastAPI Backend
# Optimized for Render free tier
# Computes similarity on demand instead of loading full matrix

import os
import pickle
import subprocess
import pandas as pd
import numpy as np
import scipy.sparse as sp
import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

# ── Load saved models ─────────────────────────────────────────
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def load_models():
    global df, tfidf_matrix, svd_matrix
    print("Loading models...")
    try:
        df          = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix  = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded successfully!")

    except Exception as e:
        print(f"Models not found: {e}")
        print("Running save_model.py...")
        subprocess.run(
            ['python',
             os.path.join(MODEL_PATH, 'save_model.py')],
            check=True
        )
        df           = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix   = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded after regeneration!")

load_models()

# ── Load API Key ──────────────────────────────────────────────
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title       = "Reddit News Recommendation API",
    description = "Recommends news using TF-IDF, SVD, Hybrid and Live News",
    version     = "2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)


# ── Request Models ────────────────────────────────────────────
class RecommendRequest(BaseModel):
    post_title : str
    method     : str = "hybrid"
    top_n      : int = 10
    subreddit  : Optional[str] = None


# ── Helper Functions ──────────────────────────────────────────
def get_post_index(post_title):
    matches = df[df['title'].str.contains(
        post_title, case=False, na=False)]
    if matches.empty:
        return None
    return matches.index[0]


def content_based_recommend(post_title, top_n=10):
    idx = get_post_index(post_title)
    if idx is None:
        return []

    # Compute similarity only for this one query row
    query_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sim_scores[idx] = 0  # remove self
    top_indices = sim_scores.argsort()[::-1][:top_n]

    result = df.iloc[top_indices][
        ['title','subreddit','score','num_comments']].copy()
    result['score_val'] = sim_scores[top_indices]
    result['method']    = 'TF-IDF Content'
    return result.to_dict('records')


def svd_based_recommend(post_title, top_n=10):
    idx = get_post_index(post_title)
    if idx is None:
        return []

    # Compute SVD similarity only for this one query row
    query_vec  = svd_matrix[idx].reshape(1, -1)
    sim_scores = cosine_similarity(query_vec, svd_matrix).flatten()
    sim_scores[idx] = 0  # remove self
    top_indices = sim_scores.argsort()[::-1][:top_n]

    result = df.iloc[top_indices][
        ['title','subreddit','score','num_comments']].copy()
    result['score_val'] = sim_scores[top_indices]
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
    idx = get_post_index(post_title)
    if idx is None:
        return []

    # TF-IDF similarity
    query_tfidf    = tfidf_matrix[idx]
    tfidf_scores   = cosine_similarity(
        query_tfidf, tfidf_matrix).flatten()
    tfidf_scores[idx] = 0

    # SVD similarity
    query_svd      = svd_matrix[idx].reshape(1, -1)
    svd_scores     = cosine_similarity(
        query_svd, svd_matrix).flatten()
    svd_scores[idx] = 0

    # Popularity score
    pop_scores     = df['hot_score_norm'].values

    # Weighted hybrid
    hybrid_scores  = (0.4 * tfidf_scores +
                      0.3 * svd_scores +
                      0.3 * pop_scores)
    hybrid_scores[idx] = 0

    top_indices = hybrid_scores.argsort()[::-1][:top_n]

    result = df.iloc[top_indices][
        ['title','subreddit','score','num_comments']].copy()
    result['score_val'] = hybrid_scores[top_indices]
    result['method']    = 'Hybrid'
    return result.to_dict('records')


def get_live_news(query=None, category=None, top_n=10):
    if not NEWS_API_KEY:
        return []
    try:
        if query:
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"sortBy=publishedAt&"
                f"pageSize={top_n}&"
                f"language=en&"
                f"apiKey={NEWS_API_KEY}"
            )
        else:
            cat_param = f"&category={category}" \
                if category else ""
            url = (
                f"https://newsapi.org/v2/top-headlines?"
                f"country=us{cat_param}&"
                f"pageSize={top_n}&"
                f"apiKey={NEWS_API_KEY}"
            )

        response = req.get(url, timeout=10)
        data     = response.json()

        if data.get('status') != 'ok':
            return []

        articles = []
        for article in data.get('articles', []):
            if article.get('title') and \
               article.get('title') != '[Removed]':
                articles.append({
                    'title'      : article.get('title', ''),
                    'description': article.get('description', ''),
                    'url'        : article.get('url', ''),
                    'source'     : article.get(
                        'source', {}).get('name', ''),
                    'published'  : article.get('publishedAt', ''),
                    'subreddit'  : category or 'general',
                    'score'      : 0,
                    'num_comments': 0,
                    'score_val'  : 1.0,
                    'method'     : 'Live News'
                })
        return articles

    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


# ── API Routes ────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message"  : "Reddit News Recommendation API is running!",
        "version"  : "2.0 — Now with Live News!",
        "endpoints": ["/recommend", "/categories",
                      "/popular", "/stats",
                      "/live-news", "/live-search"]
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


@app.get("/live-news")
def live_news(category: Optional[str] = None,
              top_n: int = 10):
    results = get_live_news(category=category, top_n=top_n)
    return {
        "category"     : category or "general",
        "total_results": len(results),
        "source"       : "NewsAPI — Live",
        "articles"     : results
    }


@app.get("/live-search")
def live_search(query: str, top_n: int = 10):
    results = get_live_news(query=query, top_n=top_n)
    return {
        "query"        : query,
        "total_results": len(results),
        "source"       : "NewsAPI — Live",
        "articles"     : results
    }


@app.get("/categories")
def get_categories():
    counts = df['subreddit'].value_counts().to_dict()
    return {"categories": counts}


@app.get("/popular")
def get_popular(subreddit: Optional[str] = None,
                top_n: int = 10):
    results = popularity_based_recommend(subreddit, top_n)
    return {
        "subreddit"      : subreddit or "all",
        "total_results"  : len(results),
        "recommendations": results
    }


@app.get("/stats")
def get_stats():
    return {
        "total_posts" : len(df),
        "categories"  : df['subreddit'].nunique(),
        "avg_score"   : round(df['score'].mean(), 2),
        "avg_comments": round(df['num_comments'].mean(), 2),
        "live_news"   : "✅ Enabled"
                        if NEWS_API_KEY else "❌ No API Key"
    }