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

### THE ACTUAL APP ###
st.image("bossdata.svg", width=200)
st.title("Llms.txt Generator")
st.write(
    "input your website main domain. Like https://bossdata.be"
)
base_url = st.text_input("Website domain:", placeholder = "https://bossdata.be")
size = st.text_input("Amount of urls (leave blank for max - 100):", placeholder = 100)


if st.button("Generate"):
    size = 100 if size.strip() == "" else min(int(size), 100)

    output_container = st.empty()

    st.write("llms.txt generation started!")
    st.write("Phase 1: fetching " + str(size) + " urls (according to size)")

    # Fase 1: bestand uitlezen en omzetten in lijst
    sitemap_url = base_url + "/page-sitemap.xml"
    urls = generator.extract_urls_from_sitemap(sitemap_url) # urls is defined here
    selected_urls = urls[:int(size)] # selected_urls is defined here

    # Fase 2: Enkel relevante URLS selecteren
    st.write("Phase 2: Filtering relevant URLS to include")
    relevant_urls = []
    #output_container.write(f"ℹ️ Starting URL filtering process for {len(selected_urls)} URLs...")
    time.sleep(2)

    for i, u in enumerate(selected_urls, start=1): # This loop now runs *after* selected_urls is defined.
        output_container.write(f"[{i}/{len(selected_urls)}] Processing: {u}")
        #if any(seg in u.lower() for seg in EXCLUDE_SEGMENTS):
            #output_container.write(f"🗑️ Dropped by pre-filter: {u}")
            #continue
        if generator.is_relevant_page(u, client, EXCLUDE_SEGMENTS):
            print("testing url: " + u)
            #output_container.write(f"✅ Kept: {u}")
            relevant_urls.append(u)
        else:
            continue
           #output_container.write(f"❌ Dropped by relevance check: {u}")

    #output_container.write(f"✅ Filtering complete! Found {len(relevant_urls)} relevant URLs.")

    # # Fase 3: Relevante URLS samenvatten
    st.write("Phase 3: Generating title & description for " + str(len(relevant_urls)) + " urls")
    results = []
    for i, u in enumerate(relevant_urls, start=1):
        output_container.write(f"[{i}/{len(relevant_urls)}] {u}")
        page_text = generator.fetch_text(u)
        item = generator.describe_page(u, page_text, client)
        results.append(item)
        time.sleep(0.5 + random.random()*0.5)

    #output_container.write("\n✅ Finished. Results:")
    #output_container.write(json.dumps(results, indent=2))

    # # Fase 4: clusteren en opmaken LLMS.TXT
    st.write("Phase 4: Clustering & creating final file")
    USE_EMBEDDINGS = True
    EMBED_MODEL = "text-embedding-3-small"
    SIM_THRESHOLD = 0.82
    llms_output = generator.build_llms_txt_from_results(results, client, USE_EMBEDDINGS, EMBED_MODEL, SIM_THRESHOLD)
    
    st.write("Final llms.txt is Ready. Download here:")
    st.download_button(
    label="Download",
    data=llms_output,
    file_name="llms.txt",
    mime="text/plain"
)



