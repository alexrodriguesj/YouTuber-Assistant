import streamlit as st
import langchain_helper as lch
import textwrap

# Logo URL
logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1280px-YouTube_full-color_icon_%282017%29.svg.png"

# Custom CSS for alignment
st.markdown("""
    <style>
    .title-container {
        display: flex;
        align-items: center;
    }
    .title-container img {
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with YouTube logo
st.markdown("""
    <div class="title-container">
        <img src="{}" width="50">
        <h1>YouTube Assistant!</h1>
    </div>
    """.format(logo_url), unsafe_allow_html=True)

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(label="Video URL", max_chars=50)
        query = st.sidebar.text_area(
            label="Ask a question about the video content!", max_chars=100, key="query"
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.create_vector_from_yt_url(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response["answer"], width=85))
