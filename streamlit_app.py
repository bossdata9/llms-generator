import streamlit as st
import llms_generator as generator
from openai import OpenAI  
import asyncio



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

        st.write("llms.txt generation started!")
        st.write(f"Phase 1: fetching up to {size} urls (according to size)")

        # Phase 1: read file and extract list
        urls = generator.extract_urls_from_sitemap(sitemap_location)
        ##st.write(urls)
        selected_urls = urls[:int(size)]

        
        st.write(f"Phase 2: Generating title & description for {len(selected_urls)} urls")

        total = len(selected_urls)
        batch_size = 20  # tweak to your liking
        progress = st.progress(0)
        status = st.empty()

        results = []
        for i in range(0, total, batch_size):
            batch = selected_urls[i:i+batch_size]
            # Run the async summarizer for this batch
            batch_results = asyncio.run(
                generator.summarize_urls_async(
                    batch,
                    openai_api_key=st.secrets["openai_api_key"],
                    max_concurrency=8,              # keep your concurrency
                    model="gpt-4.1-mini",
                )
            )
            results.extend(batch_results)

            done = min(i + batch_size, total)
            progress.progress(done / total)
            status.write(f"Phase 3: {done}/{total} pages summarized")


        # Phase 4: clustering & llms.txt creation
        st.write("Phase 3: Clustering & creating final file")
        # existing vars
        USE_EMBEDDINGS = True
        EMBED_MODEL = "text-embedding-3-large"
        SIM_THRESHOLD = 0.86

        # new: async client for embeddings/merge
        client_async = generator.AsyncOpenAI(api_key=st.secrets["openai_api_key"])

        # optional: small spinner while clustering/embedding
        with st.spinner("Phase 4: Clustering & creating final file…"):
            llms_output = asyncio.run(
                generator.build_llms_txt_from_results_async(
                    results=results,           # from your Phase 3
                    client_sync=client,        # your existing OpenAI() client
                    client_async=client_async, # async client for embeddings
                    USE_EMBEDDINGS=USE_EMBEDDINGS,
                    EMBED_MODEL=EMBED_MODEL,
                    SIM_THRESHOLD=SIM_THRESHOLD,
                )
            )

        st.write("Final llms.txt is Ready. Download here:")
        st.download_button(
            label="Download",
            data=llms_output,
            file_name="llms.txt",
            mime="text/plain"
        )




