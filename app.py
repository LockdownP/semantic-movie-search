import streamlit as st
import pandas as pd

from search import search_movies
from config import DATA_PATH


st.set_page_config(
    page_title="Semantic Movie Search",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


df = load_dataset()

genres = sorted({
    genre
    for value in df["genre"].dropna()
    for genre in value.split()
})

min_year = int(df["year"].min())
max_year = int(df["year"].max())


# Sidebar
with st.sidebar:

    st.header("Search Filters")

    genre = st.selectbox(
        "Genre",
        ["All"] + genres
    )

    year_range = st.slider(
        "Release Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=15,
        value=5
    )

    threshold = st.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )

    st.divider()

    st.caption(
        "Powered by Qdrant & Sentence Transformers"
    )


# Main page
st.title("🎬 Semantic Movie Search")

st.write(
    "ค้นหาภาพยนตร์ด้วยความหมาย "
    "รองรับภาษาไทยและภาษาอังกฤษ"
)


# Search form
with st.form("search_form"):

    query = st.text_input(
        "อยากดูหนังเกี่ยวกับอะไร?",
        placeholder="เช่น หนังเกี่ยวกับการย้อนเวลา"
    )

    submitted = st.form_submit_button(
        "🔍 Search",
        use_container_width=True
    )


# Handle search
if submitted:

    st.session_state.pop("results", None)

    if not query.strip():

        st.warning("กรุณากรอกคำค้นหา")

    else:

        try:

            with st.spinner("กำลังค้นหา..."):

                results = search_movies(
                    query=query,
                    top_k=top_k,
                    genre=genre,
                    year_from=year_range[0],
                    year_to=year_range[1],
                    score_threshold=threshold
                )

                st.session_state["results"] = results
                st.session_state["last_query"] = query

        except Exception as e:

            st.error(f"Search error: {e}")


# Display results
if "results" in st.session_state:

    results = st.session_state["results"]

    st.subheader("Search Results")

    st.caption(
        f"คำค้น: {st.session_state['last_query']}"
    )

    st.write(f"พบภาพยนตร์ {len(results)} เรื่อง")

    if not results:

        st.info(
            "ไม่พบผลลัพธ์ "
            "ลองลดคะแนนขั้นต่ำหรือเปลี่ยน Filter"
        )

    for i, movie in enumerate(results, 1):

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:

                st.subheader(
                    f"{i}. {movie['title']} "
                    f"({movie['year']})"
                )

                st.caption(movie["genre"])

                st.write(movie["overview"])

            with col2:

                score = float(movie["score"])

                st.metric(
                    "Similarity",
                    f"{score:.4f}"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )


st.divider()

st.caption(
    "Python • Qdrant • Streamlit • Docker"
)
