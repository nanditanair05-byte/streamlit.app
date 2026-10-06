import streamlit as st


st.set_page_config(
    page_title="Iris | Flower Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## 🌿 Flower Studio")
st.sidebar.caption("An interactive guide to the Iris dataset")

pages = [
    st.Page("pages/1_Discover.py", title="01 · Discover", icon="🌸", default=True),
    st.Page("pages/2_Explore_the_data.py", title="02 · Explore the data", icon="📊"),
    st.Page("pages/3_Predict_a_species.py", title="03 · Predict a species", icon="🔎"),
]

st.navigation(pages, position="sidebar").run()
