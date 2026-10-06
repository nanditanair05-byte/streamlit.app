from typing import Optional
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris


# ---------------------------------------------------------
# Feature names
# ---------------------------------------------------------

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


# ---------------------------------------------------------
# Model path
# ---------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parents[1] / "iris_model.pkl"


# ---------------------------------------------------------
# Flower images
# ---------------------------------------------------------

SPECIES_PHOTOS = {
    "versicolor": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Iris_versicolor.jpg?width=900",
        "https://commons.wikimedia.org/wiki/File:Iris_versicolor.jpg",
    ),

    "virginica": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Iris_virginica.jpg?width=900",
        "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg",
    ),
}


# ---------------------------------------------------------
# Get flower names
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_flower_data():

    iris = load_iris(as_frame=True)

    return iris.target_names


# ---------------------------------------------------------
# Find Setosa image
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def find_setosa_photo():

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
        headers={
            "User-Agent": "IrisFlowerStudio/1.0 "
            "(Streamlit educational app)"
        },
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

        image_info = image_page.get(
            "imageinfo",
            [{}]
        )[0]

        image_url = (
            image_info.get("thumburl")
            or image_info.get("url")
        )

        if image_url:

            source_url = (
                "https://commons.wikimedia.org/wiki/"
                + title.replace(" ", "_")
            )

            return image_url, source_url

    raise ValueError(
        "Wikimedia Commons did not return a usable "
        "Iris setosa photograph."
    )


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

@st.cache_resource
def get_iris_model():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as model_file:

        return pickle.load(model_file)


# ---------------------------------------------------------
# Display flower photo
# ---------------------------------------------------------

def show_species_photo(
    species: str,
    *,
    caption: Optional[str] = None
) -> None:

    if species == "setosa":

        try:

            photo_url, source_url = find_setosa_photo()

        except (
            URLError,
            TimeoutError,
            ValueError
        ) as error:

            st.error(
                "Could not load a Setosa photo from "
                f"Wikimedia Commons: {error}"
            )

            return

    else:

        photo_url, source_url = SPECIES_PHOTOS[species]

    st.image(
        photo_url,
        caption=caption,
        width="stretch"
    )

    st.markdown(
        f"[View photo and licence on Wikimedia Commons]"
        f"({source_url})"
    )


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
        linear-gradient(
            145deg,
            #fbfaff 0%,
            #f7f8fc 55%,
            #f0f4fb 100%
        );

        color: #20263b;
    }

    h1,
    h2,
    h3 {
        color: #20263b;
    }

    h1,
    h2 {
        font-family:
            'Playfair Display',
            Georgia,
            serif !important;
    }

    [data-testid="stSidebar"] {
        background: #211d38;
    }

    [data-testid="stSidebar"] * {
        color: #f7f4ff;
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    .small-note {
        color: #73798a;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #eceaf2;
        border-radius: 16px;
        padding: 1rem;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 0;
        background: #7256c8;
        color: white;
        font-weight: 700;
        padding: 0.7rem 1.25rem;
    }

    div.stButton > button:hover {
        background: #5c43ac;
        color: white;
        border: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load target names
# ---------------------------------------------------------

target_names = get_flower_data()


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.markdown(
    '<div class="eyebrow">A tiny botanical lab · 03</div>',
    unsafe_allow_html=True
)

st.title("Which Iris are you?")

st.write(
    "Enter four flower measurements and let the trained "
    "model estimate the species."
)

st.markdown(
    '<div class="small-note">'
    "Use centimetres. The sliders start near a typical "
    "Iris measurement; try changing the petal sizes to "
    "see how the prediction responds."
    "</div>",
    unsafe_allow_html=True
)

st.write("")


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

input_left, input_right = st.columns(
    [1.15, 0.85],
    vertical_alignment="center"
)


with input_left:

    sepal_length = st.slider(
        "Sepal length (cm)",
        4.0,
        8.0,
        5.8,
        0.1
    )

    sepal_width = st.slider(
        "Sepal width (cm)",
        2.0,
        4.5,
        3.0,
        0.1
    )

    petal_length = st.slider(
        "Petal length (cm)",
        1.0,
        7.0,
        3.8,
        0.1
    )

    petal_width = st.slider(
        "Petal width (cm)",
        0.1,
        2.5,
        1.2,
        0.1
    )

    predict = st.button(
        "Identify this flower",
        type="primary",
        width="stretch"
    )


with input_right:

    st.image(
        SPECIES_PHOTOS["versicolor"][0],
        width="stretch"
    )

    st.caption(
        "A closer look at an Iris flower · "
        "Wikimedia Commons"
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if predict:

    # Check whether model exists

    if not MODEL_PATH.is_file():

        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Place iris_model.pkl beside this app "
            "and try again."
        )

    else:

        # Load model

        model = get_iris_model()


        # Create input DataFrame

        measurements = pd.DataFrame(
            [
                [
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                ]
            ],
            columns=FEATURES
        )


        # Make prediction

        predicted_class = model.predict(
            measurements
        )[0]


        # Convert prediction to species name

        if isinstance(
            predicted_class,
            (
                int,
                np.integer,
                float,
                np.floating
            )
        ):

            species_name = str(
                target_names[
                    int(predicted_class)
                ]
            )

        else:

            species_name = str(
                predicted_class
            )


        # Display prediction

        st.success(
            f"Most likely species: "
            f"**{species_name.title()}**"
        )


        # Display flower image

        if species_name.lower() in {
            "setosa",
            "versicolor",
            "virginica"
        }:

            show_species_photo(
                species_name.lower(),
                caption=f"Iris {species_name}"
            )


        # Display entered measurements

        st.caption(
            f"Your measurements: "
            f"sepal {sepal_length:.1f} × "
            f"{sepal_width:.1f} cm · "
            f"petal {petal_length:.1f} × "
            f"{petal_width:.1f} cm"
        )


        # -------------------------------------------------
        # Prediction probabilities
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                measurements
            )[0]


            probability_data = pd.DataFrame(
                {
                    "Species": target_names,
                    "Model confidence": probabilities
                }
            ).set_index("Species")


            st.markdown(
                "### Model confidence"
            )


            st.bar_chart(
                probability_data,
                height=220
            )


            st.caption(
                "Confidence scores are model estimates, "
                "not botanical certainty."
            )
            