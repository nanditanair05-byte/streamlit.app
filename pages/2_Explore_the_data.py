from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import streamlit as st
from sklearn.datasets import load_iris


FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
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
    flower_data = flower_data.drop(columns="target").rename(
        columns={
            "sepal length (cm)": "Sepal length (cm)",
            "sepal width (cm)": "Sepal width (cm)",
            "petal length (cm)": "Petal length (cm)",
            "petal width (cm)": "Petal width (cm)",
        }
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


def show_species_photo(species: str) -> None:
    if species == "setosa":
        try:
            photo_url, source_url = find_setosa_photo()
        except (URLError, TimeoutError, ValueError) as error:
            st.error(f"Could not load a Setosa photo from Wikimedia Commons: {error}")
            return
    else:
        photo_url, source_url = SPECIES_PHOTOS[species]
    st.image(photo_url, width="stretch")
    st.markdown(f"[View photo and licence on Wikimedia Commons]({source_url})")


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
    .species-card { background: white; border: 1px solid #eceaf2; border-radius: 18px; padding: 1rem; text-align: center; box-shadow: 0 8px 24px #3026530a; }
    .species-card p { color: #6d7486; font-size: .9rem; }
    [data-testid="stImage"] img { border-radius: 16px; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #eceaf2; border-radius: 16px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

target_names, flower_data = get_flower_data()

st.markdown('<div class="eyebrow">Field notes · 02</div>', unsafe_allow_html=True)
st.title("Explore the data")
st.write(
    "Look for patterns across all 150 flowers. Petal measurements are especially useful for separating species."
)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Flowers", len(flower_data))
metric2.metric("Species", flower_data["species"].nunique())
metric3.metric("Measurements", len(FEATURES))

st.subheader("A closer look at each species")
image_columns = st.columns(3)
for column, species in zip(image_columns, ["setosa", "versicolor", "virginica"]):
    with column:
        show_species_photo(species)
        st.markdown(
            f'<div class="species-card"><h3>{species.title()}</h3>'
            f'<p>{(flower_data["species"] == species).sum()} samples in this dataset</p></div>',
            unsafe_allow_html=True,
        )

chart_left, chart_right = st.columns([1.35, 0.9])
with chart_left:
    st.markdown("### Petal length vs. width")
    st.scatter_chart(
        flower_data,
        x="Petal length (cm)",
        y="Petal width (cm)",
        color="species",
        height=390,
    )
    st.caption("Each point is one flower. Colours identify its species.")
with chart_right:
    st.markdown("### Average measurements")
    species_means = flower_data.groupby("species", sort=False)[
        ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
    ].mean()
    st.bar_chart(
        species_means[["Petal length (cm)", "Petal width (cm)"]],
        height=340,
    )
    st.caption("Mean petal measurements by species, in centimetres.")

st.markdown("### Summary statistics")
summary = flower_data.groupby("species", sort=False)[
    ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
].mean().round(2)
st.dataframe(summary, width="stretch")