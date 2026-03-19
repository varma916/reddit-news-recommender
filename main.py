# main.py - FastAPI Backend
# Real Time News using NewsData.io + GNews API

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
from datetime import datetime, timezone

# ── Load Models ───────────────────────────────────────────────
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def load_models():
    global df, tfidf_matrix, svd_matrix
    print("Loading models...")
    try:
        df           = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix   = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"Models not found: {e}")
        subprocess.run(
            ['python',
             os.path.join(MODEL_PATH, 'save_model.py')],
            check=True)
        df           = pd.read_pickle(
            os.path.join(MODEL_PATH, 'df.pkl'))
        tfidf_matrix = sp.load_npz(
            os.path.join(MODEL_PATH, 'tfidf_matrix.npz'))
        svd_matrix   = np.load(
            os.path.join(MODEL_PATH, 'svd_matrix.npy'))
        print("✅ Models loaded after regeneration!")

load_models()

# ── API Keys ──────────────────────────────────────────────────
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
GNEWS_API_KEY    = os.getenv('GNEWS_API_KEY', '')

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title       = "Reddit News Recommendation API",
    description = "Real time news using NewsData.io + GNews + TF-IDF + SVD",
    version     = "3.0"
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


# ── Format Time Ago ───────────────────────────────────────────
def time_ago(published_str):
    try:
        if not published_str:
            return ""
        published_str = published_str.replace('Z', '+00:00')
        published     = datetime.fromisoformat(published_str)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        now  = datetime.now(timezone.utc)
        diff = now - published
        secs = int(diff.total_seconds())
        if secs < 60:
            return f"{secs} seconds ago"
        elif secs < 3600:
            return f"{secs // 60} minutes ago"
        elif secs < 86400:
            return f"{secs // 3600} hours ago"
        else:
            return f"{secs // 86400} days ago"
    except:
        return published_str


# ── NewsData.io API (Primary) ─────────────────────────────────
def get_newsdata(query=None, category=None, top_n=10):
    if not NEWSDATA_API_KEY:
        return []
    try:
        params = {
            'apikey'  : NEWSDATA_API_KEY,
            'language': 'en',
            'size'    : min(top_n, 10)
        }
        if query:
            params['q'] = query
        if category:
            cat_map = {
                'business'     : 'business',
                'technology'   : 'technology',
                'health'       : 'health',
                'sports'       : 'sports',
                'entertainment': 'entertainment',
                'science'      : 'science',
                'general'      : 'top'
            }
            params['category'] = cat_map.get(
                category, 'top')

        url      = "https://newsdata.io/api/1/latest"
        response = req.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"NewsData error: {response.status_code}")
            return []

        data     = response.json()
        articles = []

        for item in data.get('results', []):
            published = item.get('pubDate', '')
            articles.append({
                'title'      : item.get('title', ''),
                'description': item.get(
                    'description', '') or '',
                'url'        : item.get('link', ''),
                'source'     : item.get('source_name', ''),
                'published'  : published,
                'time_ago'   : time_ago(published),
                'image'      : item.get('image_url', '') or '',
                'category'   : ', '.join(
                    item.get('category', ['general'])),
                'subreddit'  : category or 'general',
                'score_val'  : 1.0,
                'method'     : 'Live News (NewsData.io)',
                'api_source' : 'newsdata'
            })

        return articles

    except Exception as e:
        print(f"NewsData.io error: {e}")
        return []


# ── GNews API (Backup) ────────────────────────────────────────
def get_gnews(query=None, category=None, top_n=10):
    if not GNEWS_API_KEY:
        return []
    try:
        if query:
            url    = "https://gnews.io/api/v4/search"
            params = {
                'q'     : query,
                'lang'  : 'en',
                'max'   : min(top_n, 10),
                'sortby': 'publishedAt',
                'apikey': GNEWS_API_KEY
            }
        else:
            url       = "https://gnews.io/api/v4/top-headlines"
            topic_map = {
                'business'     : 'business',
                'technology'   : 'technology',
                'health'       : 'health',
                'sports'       : 'sports',
                'entertainment': 'entertainment',
                'science'      : 'science',
                'general'      : 'breaking-news'
            }
            params = {
                'lang'  : 'en',
                'max'   : min(top_n, 10),
                'apikey': GNEWS_API_KEY
            }
            if category:
                params['topic'] = topic_map.get(
                    category, 'breaking-news')

        response = req.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"GNews error: {response.status_code}")
            return []

        data     = response.json()
        articles = []

        for item in data.get('articles', []):
            published = item.get('publishedAt', '')
            articles.append({
                'title'      : item.get('title', ''),
                'description': item.get(
                    'description', '') or '',
                'url'        : item.get('url', ''),
                'source'     : item.get(
                    'source', {}).get('name', ''),
                'published'  : published,
                'time_ago'   : time_ago(published),
                'image'      : item.get('image', '') or '',
                'category'   : category or 'general',
                'subreddit'  : category or 'general',
                'score_val'  : 1.0,
                'method'     : 'Live News (GNews)',
                'api_source' : 'gnews'
            })

        return articles

    except Exception as e:
        print(f"GNews error: {e}")
        return []


# ── Smart Live News ───────────────────────────────────────────
def get_live_news(query=None, category=None, top_n=10):
    # Try NewsData.io first
    if NEWSDATA_API_KEY:
        results = get_newsdata(query, category, top_n)
        if results:
            print(f"NewsData.io: {len(results)} articles")
            return results

    # Try GNews second
    if GNEWS_API_KEY:
        results = get_gnews(query, category, top_n)
        if results:
            print(f"GNews: {len(results)} articles")
            return results

    return []


# ── ML Recommendation Functions ───────────────────────────────
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
    query_vec   = tfidf_matrix[idx]
    sim_scores  = cosine_similarity(
        query_vec, tfidf_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    result      = df.iloc[top_indices][
        ['title','subreddit','score','num_comments']].copy()
    result['score_val'] = sim_scores[top_indices]
    result['method']    = 'TF-IDF Content'
    return result.to_dict('records')


def svd_based_recommend(post_title, top_n=10):
    idx = get_post_index(post_title)
    if idx is None:
        return []
    query_vec   = svd_matrix[idx].reshape(1, -1)
    sim_scores  = cosine_similarity(
        query_vec, svd_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:top_n]
    result      = df.iloc[top_indices][
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
    tfidf_scores       = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix).flatten()
    tfidf_scores[idx]  = 0
    svd_scores         = cosine_similarity(
        svd_matrix[idx].reshape(1, -1),
        svd_matrix).flatten()
    svd_scores[idx]    = 0
    pop_scores         = df['hot_score_norm'].values
    hybrid_scores      = (0.4 * tfidf_scores +
                          0.3 * svd_scores +
                          0.3 * pop_scores)
    hybrid_scores[idx] = 0
    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    result      = df.iloc[top_indices][
        ['title','subreddit','score','num_comments']].copy()
    result['score_val'] = hybrid_scores[top_indices]
    result['method']    = 'Hybrid'
    return result.to_dict('records')


# ── API Routes ────────────────────────────────────────────────
@app.get("/")
def home():
    apis = []
    if NEWSDATA_API_KEY:
        apis.append("NewsData.io")
    if GNEWS_API_KEY:
        apis.append("GNews")
    return {
        "message"  : "Reddit News Recommendation API v3.0!",
        "live_news": apis,
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
    results  = get_live_news(category=category, top_n=top_n)
    api_used = results[0].get(
        'api_source', 'unknown') if results else 'none'
    return {
        "category"     : category or "general",
        "total_results": len(results),
        "api_used"     : api_used,
        "articles"     : results
    }


@app.get("/live-search")
def live_search(query: str, top_n: int = 10):
    results  = get_live_news(query=query, top_n=top_n)
    api_used = results[0].get(
        'api_source', 'unknown') if results else 'none'
    return {
        "query"        : query,
        "total_results": len(results),
        "api_used"     : api_used,
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
    apis = []
    if NEWSDATA_API_KEY:
        apis.append("NewsData.io")
    if GNEWS_API_KEY:
        apis.append("GNews")
    return {
        "total_posts" : len(df),
        "categories"  : df['subreddit'].nunique(),
        "avg_score"   : round(df['score'].mean(), 2),
        "avg_comments": round(df['num_comments'].mean(), 2),
        "live_news"   : " + ".join(apis) if apis
                        else "Not configured"
    }