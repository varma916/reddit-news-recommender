# app.py — Streamlit Frontend
# Now with Live News tab!

import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title = "Reddit News Recommender",
    page_icon  = "📰",
    layout     = "wide"
)

FASTAPI_URL = "https://reddit-news-api.onrender.com"

st.title("📰 Reddit News Recommendation System")
st.markdown("*Powered by TF-IDF + SVD + Reddit Hot Algorithm + Live News*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
method = st.sidebar.selectbox(
    "Recommendation Method",
    ["hybrid", "tfidf", "svd", "popularity"],
    format_func=lambda x: {
        "hybrid"    : "🔀 Hybrid (Best)",
        "tfidf"     : "📝 TF-IDF Content",
        "svd"       : "🔢 SVD Semantic",
        "popularity": "🔥 Popularity"
    }[x]
)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# ── Stats Cards ───────────────────────────────────────────────
try:
    stats = requests.get(f"{FASTAPI_URL}/stats").json()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Total Posts",   f"{stats['total_posts']:,}")
    col2.metric("🏷️ Categories",   stats['categories'])
    col3.metric("⬆️ Avg Upvotes",  stats['avg_score'])
    col4.metric("💬 Avg Comments", stats['avg_comments'])
    col5.metric("📡 Live News",    stats['live_news'])
    st.markdown("---")
except:
    st.warning("⚠️ Could not connect to FastAPI!")


# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search Recommendations",
    "📡 Live News",
    "🔥 Popular Posts",
    "📊 Category Stats"
])


# ── Tab 1 — Search ────────────────────────────────────────────
with tab1:
    st.subheader("🔍 Find Similar News Posts")
    post_title = st.text_input(
        "Enter a news headline:",
        placeholder="e.g. US military troops deployed to Syria"
    )
    search_btn = st.button("🔍 Get Recommendations",
                           use_container_width=False)

    if search_btn and post_title:
        with st.spinner("Finding recommendations..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/recommend",
                    json={
                        "post_title": post_title,
                        "method"    : method,
                        "top_n"     : top_n
                    }
                )
                data = response.json()

                if data['total_results'] == 0:
                    st.error("No recommendations found. Try a different headline.")
                else:
                    st.success(f"✅ Found {data['total_results']} recommendations using **{data['method'].upper()}**")
                    st.markdown("---")
                    recs = pd.DataFrame(data['recommendations'])
                    for i, row in recs.iterrows():
                        col1, col2, col3 = st.columns([5, 1, 1])
                        with col1:
                            st.markdown(f"**{i+1}. {row['title']}**")
                            st.caption(f"🏷️ {row['subreddit']}")
                        with col2:
                            st.metric("⬆️ Upvotes", int(row['score']))
                        with col3:
                            st.metric("📊 Score", f"{row['score_val']:.3f}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error: {e}")

    elif search_btn and not post_title:
        st.warning("Please enter a headline first!")


# ── Tab 2 — Live News ─────────────────────────────────────────
with tab2:
    st.subheader("📡 Live News — Powered by NewsAPI")
    st.markdown("*Real time news updated every hour*")

    col1, col2 = st.columns(2)

    with col1:
        live_category = st.selectbox(
            "Select Category",
            ["general", "business", "technology",
             "health", "science", "sports", "entertainment"]
        )

    with col2:
        live_search_query = st.text_input(
            "Or search live news:",
            placeholder="e.g. cricket, election, climate"
        )

    live_btn = st.button("📡 Get Live News", use_container_width=False)

    if live_btn:
        with st.spinner("Fetching live news..."):
            try:
                if live_search_query:
                    response = requests.get(
                        f"{FASTAPI_URL}/live-search",
                        params={"query": live_search_query,
                                "top_n": top_n}
                    )
                else:
                    response = requests.get(
                        f"{FASTAPI_URL}/live-news",
                        params={"category": live_category,
                                "top_n": top_n}
                    )

                data = response.json()
                articles = data.get('articles', [])

                if not articles:
                    st.error("No live news found. Try a different search!")
                else:
                    st.success(f"✅ Found {len(articles)} live articles!")
                    st.markdown("---")

                    for i, article in enumerate(articles):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{i+1}. {article['title']}**")
                            if article.get('description'):
                                st.caption(article['description'][:150] + '...')
                            st.caption(
                                f"📰 {article.get('source', 'Unknown')} | "
                                f"🕐 {article.get('published', '')[:10]}"
                            )
                            if article.get('url'):
                                st.markdown(f"[Read Full Article →]({article['url']})")
                        with col2:
                            st.markdown("🔴 **LIVE**")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error fetching live news: {e}")


# ── Tab 3 — Popular ───────────────────────────────────────────
with tab3:
    st.subheader("🔥 Most Popular Posts")
    try:
        cats_response = requests.get(
            f"{FASTAPI_URL}/categories").json()
        categories    = ["All"] + list(
            cats_response['categories'].keys())
        selected_cat  = st.selectbox(
            "Filter by Category", categories)
        subreddit_filter = None if selected_cat == "All" \
            else selected_cat

        pop_response = requests.get(
            f"{FASTAPI_URL}/popular",
            params={"subreddit": subreddit_filter,
                    "top_n": top_n}
        ).json()

        pop_recs = pd.DataFrame(pop_response['recommendations'])

        if not pop_recs.empty:
            st.markdown(f"**Top {len(pop_recs)} posts in: {selected_cat}**")
            st.markdown("---")
            for i, row in pop_recs.iterrows():
                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    st.markdown(f"**{i+1}. {row['title']}**")
                    st.caption(f"🏷️ {row['subreddit']}")
                with col2:
                    st.metric("⬆️ Upvotes", int(row['score']))
                with col3:
                    st.metric("🔥 Hot", f"{row['score_val']:.3f}")
                st.markdown("---")
    except Exception as e:
        st.error(f"Error: {e}")


# ── Tab 4 — Category Stats ────────────────────────────────────
with tab4:
    st.subheader("📊 Category Distribution")
    try:
        cats    = requests.get(
            f"{FASTAPI_URL}/categories").json()
        cats_df = pd.DataFrame(
            list(cats['categories'].items()),
            columns=['Category', 'Count']
        ).sort_values('Count', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Posts per Category**")
            st.dataframe(cats_df, use_container_width=True)
        with col2:
            st.markdown("**Category Distribution Chart**")
            st.bar_chart(cats_df.set_index('Category'))
    except Exception as e:
        st.error(f"Error: {e}")


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Built with FastAPI + Streamlit | "
    "Dataset: UCI News | "
    "Live News: NewsAPI | "
    "Algorithms: TF-IDF + SVD + Hot Algorithm*"
)