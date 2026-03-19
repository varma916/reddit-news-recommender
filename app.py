import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title = "Reddit News Recommender",
    page_icon  = "📰",
    layout     = "wide"
)

FASTAPI_URL = "https://reddit-news-api.onrender.com"

st.title("Reddit News Recommendation System")
st.markdown("*Powered by TF-IDF + SVD + Reddit Hot Algorithm + Real Time News*")
st.markdown("---")

try:
    stats = requests.get(
        f"{FASTAPI_URL}/stats", timeout=30).json()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Posts",  f"{stats['total_posts']:,}")
    col2.metric("Categories",   stats['categories'])
    col3.metric("Avg Upvotes",  stats['avg_score'])
    col4.metric("Avg Comments", stats['avg_comments'])
    col5.metric("Live News",    stats['live_news'])
    st.markdown("---")
except:
    st.warning("Could not connect to FastAPI!")

tab1, tab2, tab3 = st.tabs([
    "Search Recommendations",
    "Live News",
    "Category Stats"
])

with tab1:
    st.subheader("Find Similar News Posts")
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        post_title = st.text_input(
            "Enter a news headline:",
            placeholder="e.g. US military troops deployed")
    with col2:
        method = st.selectbox(
            "Recommendation Method",
            ["hybrid", "tfidf", "svd", "popularity"],
            format_func=lambda x: {
                "hybrid"    : "Hybrid (Best)",
                "tfidf"     : "TF-IDF Content",
                "svd"       : "SVD Semantic",
                "popularity": "Popularity"
            }[x])
    with col3:
        top_n = st.slider("Results", 5, 20, 10)

    if st.button("Get Recommendations"):
        if not post_title:
            st.warning("Please enter a headline!")
        else:
            with st.spinner("Finding recommendations..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/recommend",
                        json={
                            "post_title": post_title,
                            "method"    : method,
                            "top_n"     : top_n
                        },
                        timeout=30)
                    data = response.json()
                    if data['total_results'] == 0:
                        st.error("No recommendations found!")
                    else:
                        st.success(
                            f"Found {data['total_results']} "
                            f"recommendations using "
                            f"{data['method'].upper()}!")
                        st.markdown("---")
                        recs = pd.DataFrame(
                            data['recommendations'])
                        for i, row in recs.iterrows():
                            c1, c2, c3 = st.columns(
                                [5, 1, 1])
                            with c1:
                                st.markdown(
                                    f"**{i+1}. {row['title']}**")
                                st.caption(
                                    f"Category: {row['subreddit']}")
                            with c2:
                                st.metric("Upvotes",
                                          int(row['score']))
                            with c3:
                                st.metric("Score",
                                    f"{row['score_val']:.3f}")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.subheader("Live News")
    st.markdown("*Real time news updated every few minutes!*")

    col1, col2 = st.columns(2)
    with col1:
        live_category = st.selectbox(
            "Select Category",
            ["general", "business", "technology",
             "health", "sports", "entertainment",
             "science"])
    with col2:
        live_query = st.text_input(
            "Or search live news:",
            placeholder="e.g. cricket, election, AI")

    live_top_n = st.slider(
        "Number of Articles", 5, 20, 10)

    if st.button("Get Live News"):
        with st.spinner("Fetching latest news..."):
            try:
                if live_query:
                    response = requests.get(
                        f"{FASTAPI_URL}/live-search",
                        params={
                            "query" : live_query,
                            "top_n" : live_top_n
                        },
                        timeout=30)
                else:
                    response = requests.get(
                        f"{FASTAPI_URL}/live-news",
                        params={
                            "category": live_category,
                            "top_n"   : live_top_n
                        },
                        timeout=30)

                data     = response.json()
                articles = data.get('articles', [])
                api_used = data.get('api_used', '')

                if not articles:
                    st.error("No live news found!")
                else:
                    st.success(
                        f"Found {len(articles)} real time "
                        f"articles via {api_used}!")
                    st.markdown("---")

                    for i, article in enumerate(articles):
                        with st.container():
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                st.markdown(
                                    f"**{i+1}. {article['title']}**")

                                if article.get('description'):
                                    st.caption(
                                        str(article['description'])[:200])

                                info_parts = []
                                if article.get('source'):
                                    info_parts.append(
                                        f"Source: {article['source']}")
                                if article.get('time_ago'):
                                    info_parts.append(
                                        f"Published: {article['time_ago']}")
                                if article.get('category'):
                                    info_parts.append(
                                        f"Category: {article['category']}")

                                if info_parts:
                                    st.caption(
                                        " | ".join(info_parts))

                                if article.get('url'):
                                    st.markdown(
                                        f"[Read Full Article]"
                                        f"({article['url']})")

                            with col2:
                                time_str = article.get(
                                    'time_ago', '')
                                if 'minute' in time_str or \
                                   'second' in time_str:
                                    st.success("LIVE")
                                elif 'hour' in time_str:
                                    hrs = time_str.split()[0]
                                    st.info(f"{hrs}h ago")
                                else:
                                    st.warning(time_str)

                            st.markdown("---")

            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.subheader("Category Distribution")
    try:
        cats    = requests.get(
            f"{FASTAPI_URL}/categories",
            timeout=30).json()
        cats_df = pd.DataFrame(
            list(cats['categories'].items()),
            columns=['Category', 'Count']
        ).sort_values('Count', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Posts per Category**")
            st.dataframe(cats_df, use_container_width=True)
        with col2:
            st.markdown("**Category Distribution**")
            st.bar_chart(cats_df.set_index('Category'))
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "*Built with FastAPI + Streamlit | "
    "Real Time News: NewsData.io + GNews | "
    "Algorithms: TF-IDF + SVD + Hot Algorithm*"
)