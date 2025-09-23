import streamlit as st
from PIL import Image
import time
import llms_generator as generator
import os
from openai import OpenAI
import json
import random
import openai   # <-- this is the missing piece


st.markdown(
    """
    <style>
    /* -------------------------------
       Import Custom Font
    --------------------------------*/
    @import url('https://db.onlinewebfonts.com/c/ac9525e5f200f57332b3080d0db9d8f6?family=Sailec+Medium');

    /* -------------------------------
       Background & Base Font
    --------------------------------*/
    .stApp {
        background-color: #ffffff;
        font-family: "Sailec Medium", Helvetica, Arial, sans-serif;
    }

    /* -------------------------------
       Global Text Color (no spans!)
    --------------------------------*/
    .stApp :is(h1,h2,h3,h4,h5,h6,p,label,li,strong,em) {
        color: #48546e;
    }

    /* -------------------------------
       Input Fields (lighter grey background)
    --------------------------------*/
    textarea, input[type="text"], input[type="number"], input[type="password"] {
        background-color: #666666 !important;  /* medium grey */
        color: #ffffff !important;             /* white text */
        border-radius: 6px !important;
        border: 1px solid #48546e !important;
    }

    /* -------------------------------
       Buttons (Run & Download)
    --------------------------------*/
    .stButton > button,
    .stDownloadButton > button {
        background-color: #48546e !important;
        border-radius: 8px !important;
        border: 2px solid #48546e !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton > button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        text-shadow: none !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #384252 !important;
        border-color: #384252 !important;
    }
    .stButton > button:hover *,
    .stDownloadButton > button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    /* Extra future-proofing */
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[data-baseweb="button"] {
        color: #ffffff !important;
    }
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[data-baseweb="button"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* --- Input Fields Background + Width --- */
    textarea, input[type="text"], input[type="number"], input[type="password"] {
        background-color: #666666 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #48546e !important;
    }
    div[data-baseweb="input"] {
        width: 33% !important;
        min-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)







### ----------------------------------------------------------------------------- ###
### CONFIGURATION ###
client = OpenAI(api_key=st.secrets["openai_api_key"])
EXCLUDE_SEGMENTS = ["index", "home", "homepage", "privacy", "terms", "legal", "sitemap",
        "sitemap.xml", "robots.txt", "author", "authors", "admin", "login",
        "user-data", "settings", "internal-docs", "pricing", "sales-materials", "confidential", "beta", "staging", "dev", "404",
        "search", "thank-you", "cart", "tag", "category", "archive"]

import streamlit as st

# === UI HEADER ===
st.image("bossdata.svg", width=200)
st.title("Llms.txt Generator")
st.write(
    "Input your page sitemap xml. Like https://bossdata.be/page-sitemap.xml. Only include pages that you want. "
    "For example, don't use specific detailed product pages, but product category pages instead. "
    "Just remove the lines in the sitemap that you don't want to include and save. Then upload your new version into this app."
)

# ✅ New: toggle to enable/disable relevance filtering (Phase 2)
run_filtering = st.checkbox("Run relevance filtering (Phase 2)", value=False, help="If unchecked, all fetched URLs are kept.")

uploaded_file = st.file_uploader("Upload sitemap XML file", type=["xml"])
sitemap_location = uploaded_file if uploaded_file is not None else None

size_input = st.text_input("Amount of urls (leave blank for max - 10000):", placeholder="10000")

# ✅ Use only one button
generate_clicked = st.button("Generate", key="generate_btn")

if generate_clicked:
    if sitemap_location is None:
        st.warning("Please upload a sitemap XML first.")
    else:
        # Parse size safely
        try:
            size = 10000 if size_input.strip() == "" else min(int(size_input), 10000)
        except Exception:
            st.warning("Invalid size. Using default 10000.")
            size = 10000

        output_container = st.empty()

        st.write("llms.txt generation started!")
        st.write(f"Phase 1: fetching up to {size} urls (according to size)")

        # Phase 1: read file and extract list
        urls = generator.extract_urls_from_sitemap(sitemap_location)
        st.write(urls)
        selected_urls = urls[:int(size)]

        # Phase 2: (optional) relevance filtering
        relevant_urls = []
        if run_filtering:
            st.write("Phase 2: Filtering relevant URLs to include")
            for i, u in enumerate(selected_urls, start=1):
                output_container.write(f"[{i}/{len(selected_urls)}] Processing: {u}")
                if generator.is_relevant_page(u, client, EXCLUDE_SEGMENTS):
                    relevant_urls.append(u)
                else:
                    continue
            st.write(f"✅ Filtering complete! Kept {len(relevant_urls)} of {len(selected_urls)} URLs.")
        else:
            st.write("Phase 2: Skipped (checkbox off) — keeping all fetched URLs.")
            relevant_urls = selected_urls

        # Phase 3: summarize relevant pages
        st.write(f"Phase 3: Generating title & description for {len(relevant_urls)} urls")
        results = []
        for i, u in enumerate(relevant_urls, start=1):
            output_container.write(f"[{i}/{len(relevant_urls)}] {u}")
            page_text = generator.fetch_text(u)
            item = generator.describe_page(u, page_text, client)
            results.append(item)

        # Phase 4: clustering & llms.txt creation
        st.write("Phase 4: Clustering & creating final file")
        USE_EMBEDDINGS = True
        EMBED_MODEL = "text-embedding-3-large"
        SIM_THRESHOLD = 0.92
        llms_output = generator.build_llms_txt_from_results(
            results, client, USE_EMBEDDINGS, EMBED_MODEL, SIM_THRESHOLD
        )

        st.write("Final llms.txt is Ready. Download here:")
        st.download_button(
            label="Download",
            data=llms_output,
            file_name="llms.txt",
            mime="text/plain"
        )




