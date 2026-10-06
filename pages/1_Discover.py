from urllib.error import URLError

from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import streamlit as st
from sklearn.datasets import load_iris


SPECIES_PHOTOS = {
    "versicolor": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor.jpg?width=900",
        "https://commons.wikimedia.org/wiki/File:Iris_versicolor.jpg",
    ),
    "virginica": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg?width=900",
        "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg",
    ),
}


@st.cache_data(show_spinner=False)
def get_flower_data():
    iris = load_iris(as_frame=True)
    flower_data = iris.frame.copy()
    flower_data["species"] = flower_data["target"].map(
        dict(enumerate(iris.target_names))
    )
    return iris.target_names, flower_data


@st.cache_data(show_spinner=False)
def find_setosa_photo() -> tuple[str, str]:
    params = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": "Iris setosa flower",
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 900,
            "format": "json",
        }
    )
    request = Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "IrisFlowerStudio/1.0 (Streamlit educational app)"},
    )
    with urlopen(request, timeout=10) as response:
        results = json.load(response)
    candidates = sorted(
        results.get("query", {}).get("pages", {}).values(),
        key=lambda item: item.get("index", 999),
    )
    for image_page in candidates:
        title = image_page.get("title", "")
        if "setosa" not in title.lower():
            continue
        image_info = image_page.get("imageinfo", [{}])[0]
        image_url = image_info.get("thumburl") or image_info.get("url")
        if image_url:
            source_url = "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
            return image_url, source_url
    raise ValueError("Wikimedia Commons did not return a usable Iris setosa photograph.")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(145deg, #fbfaff 0%, #f7f8fc 55%, #f0f4fb 100%); color: #20263b; }
    h1, h2, h3 { color: #20263b; }
    h1, h2 { font-family: 'Playfair Display', Georgia, serif !important; }
    [data-testid="stSidebar"] { background: #211d38; }
    [data-testid="stSidebar"] * { color: #f7f4ff; }
    .hero { padding: 2.1rem 2.25rem; border-radius: 24px; background: linear-gradient(120deg, #eee9ff, #f9f4ff 58%, #eaf4ff); border: 1px solid #e7e0fa; }
    .eyebrow { color: #735bc1; text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 700; }
    .hero-title { font-family: 'Playfair Display', Georgia, serif; font-size: clamp(2.1rem, 4vw, 3.65rem); line-height: 1.1; color: #24213b; margin: .35rem 0 .8rem; }
    .hero-copy { color: #5f6375; font-size: 1.05rem; max-width: 650px; line-height: 1.7; }
    .info-card { background: white; padding: 1.15rem 1.25rem; border: 1px solid #eceaf2; border-radius: 18px; height: 100%; box-shadow: 0 8px 24px #3026530a; }
    .info-card p { color: #6d7486; line-height: 1.6; margin-bottom: 0; }
    .species-card { background: white; border: 1px solid #eceaf2; border-radius: 18px; padding: 1rem; text-align: center; box-shadow: 0 8px 24px #3026530a; }
    .species-card p { color: #6d7486; font-size: .9rem; }
    [data-testid="stImage"] img { border-radius: 16px; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #eceaf2; border-radius: 16px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

target_names, flower_data = get_flower_data()

left, right = st.columns([1.45, 0.8], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">A little field guide · 01</div>
          <div class="hero-title">Meet the Iris<br>flower family.</div>
          <div class="hero-copy">Three species. Four simple measurements. One of the most loved
          datasets in machine learning. Start with the flower, then explore the data behind it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.image(SPECIES_PHOTOS["virginica"][0], width="stretch")
    st.caption("A real Iris bloom")
    st.markdown(f"[Photo source and licence]({SPECIES_PHOTOS['virginica'][1]})")

st.write("")
st.subheader("The essentials")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="info-card"><h3>🌸 A real flower</h3><p>Iris is a large genus of flowering plants. '
        "Its showy blooms come in many colours and grow from bulbs or rhizomes.</p></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="info-card"><h3>📏 Four measurements</h3><p>The classic dataset records sepal length, '
        "sepal width, petal length, and petal width, all in centimetres.</p></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="info-card"><h3>🧪 A famous dataset</h3><p>Introduced by statistician Ronald Fisher '
        "in 1936, it contains 150 flowers—50 examples from each species.</p></div>",
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("Three species in the collection")
species_columns = st.columns(3)
species_details = [
    ("setosa", "The smallest petals in this dataset; often the easiest class to distinguish."),
    ("versicolor", "A middle-sized species whose measurements overlap with the others."),
    ("virginica", "Typically the largest flowers in this collection, especially by petal size."),
]
for column, (species, description) in zip(species_columns, species_details):
    with column:
        if species == "setosa":
            try:
                photo_url, source_url = find_setosa_photo()
            except (URLError, TimeoutError, ValueError) as error:
                st.error(f"Could not load a Setosa photo from Wikimedia Commons: {error}")
                photo_url = None
                source_url = None
            if photo_url:
                st.image(photo_url, width="stretch")
        else:
            photo_url, source_url = SPECIES_PHOTOS[species]
            st.image(photo_url, width="stretch")
        st.markdown(
            f'<div class="species-card"><h3>{species.title()}</h3>'
            f'<p><i>Iris {species}</i><br>{description}</p></div>',
            unsafe_allow_html=True,
        )
        if source_url:
            st.markdown(f"[View photo and licence on Wikimedia Commons]({source_url})")