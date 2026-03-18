# app.py — Streamlit Frontend
# Beautiful web UI for Reddit News Recommendation

import streamlit as st
import requests
import pandas as pd

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Reddit News Recommender",
    page_icon  = "📰",
    layout     = "wide"
)

FASTAPI_URL = "http://127.0.0.1:8000"

# ── Header ────────────────────────────────────────────────────
st.title("📰 Reddit News Recommendation System")
st.markdown("*Powered by TF-IDF + SVD + Reddit Hot Algorithm*")
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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Posts",    f"{stats['total_posts']:,}")
    col2.metric("🏷️ Categories",    stats['categories'])
    col3.metric("⬆️ Avg Upvotes",   stats['avg_score'])
    col4.metric("💬 Avg Comments",  stats['avg_comments'])
    st.markdown("---")
except:
    st.warning("⚠️ Could not connect to FastAPI. Make sure main.py is running!")


# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Search Recommendations",
    "🔥 Popular Posts",
    "📊 Category Stats"
])


# ── Tab 1 — Search ────────────────────────────────────────────
with tab1:
    st.subheader("🔍 Find Similar News Posts")
    st.markdown("Enter a news headline to get recommendations")

    post_title = st.text_input(
        "Enter a news headline:",
        placeholder="e.g. US military troops deployed to Syria"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.button("🔍 Get Recommendations",
                               use_container_width=True)

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
                    st.success(f"✅ Found {data['total_results']} recommendations using **{data['method'].upper()}** method")
                    st.markdown("---")

                    recs = pd.DataFrame(data['recommendations'])

                    for i, row in recs.iterrows():
                        with st.container():
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
                st.error(f"Error connecting to API: {e}")

    elif search_btn and not post_title:
        st.warning("Please enter a news headline first!")


# ── Tab 2 — Popular ───────────────────────────────────────────
with tab2:
    st.subheader("🔥 Most Popular Reddit Posts")

    try:
        cats_response = requests.get(
            f"{FASTAPI_URL}/categories").json()
        categories = ["All"] + list(cats_response['categories'].keys())
        selected_cat = st.selectbox("Filter by Category", categories)
        subreddit_filter = None if selected_cat == "All" else selected_cat

        pop_response = requests.get(
            f"{FASTAPI_URL}/popular",
            params={"subreddit": subreddit_filter, "top_n": top_n}
        ).json()

        pop_recs = pd.DataFrame(pop_response['recommendations'])

        if not pop_recs.empty:
            st.markdown(f"**Showing top {len(pop_recs)} posts in: {selected_cat}**")
            st.markdown("---")

            for i, row in pop_recs.iterrows():
                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    st.markdown(f"**{i+1}. {row['title']}**")
                    st.caption(f"🏷️ {row['subreddit']}")
                with col2:
                    st.metric("⬆️ Upvotes", int(row['score']))
                with col3:
                    st.metric("🔥 Hot Score", f"{row['score_val']:.3f}")
                st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")


# ── Tab 3 — Category Stats ────────────────────────────────────
with tab3:
    st.subheader("📊 Category Distribution")

    try:
        cats = requests.get(f"{FASTAPI_URL}/categories").json()
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
    "Dataset: Reddit r/worldnews | "
    "Algorithms: TF-IDF + SVD + Hot Algorithm*"
)