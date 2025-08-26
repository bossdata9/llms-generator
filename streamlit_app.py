import streamlit as st
from PIL import Image
import time
import llms_generator as generator

# --- Solid Background Color ---
# Use a hex code, an RGB value, or a color name.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff; /* A light gray background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp span, .stApp button {
        color: #48546e; /* A professional-looking dark blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Target the text input container */
    div.st-af-Textarea,
    div.st-emotion-cache-1ftn37z {
        background-color: #333333; /* Dark gray background */
    }

    /* Target the text inside the input field */
    div.st-emotion-cache-16ids61-Textarea textarea,
    div.st-emotion-cache-1ftn37z input {
        color: #FFFFFF; /* White text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url(https://db.onlinewebfonts.com/c/ac9525e5f200f57332b3080d0db9d8f6?family=Sailec+Medium);

    .stApp {
        font-family: "Sailec Medium", sans-serif; /* Use the exact name with quotes */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Change the button's background and text color */
    div.stButton > button:first-child {
        background-color: #48546e; /* A shade of blue */
        color: white; /* This changes the text color to white */
    }

    /* Add a hover effect */
    div.stButton > button:first-child:hover {
        background-color: #005f99; /* A darker shade of blue for hover */
        color: white; /* Keep the text white on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)








### -----------------------------------------------------------------------------
### THE ACTUAL APP ###
st.image("bossdata.svg", width=200)
st.title("Llms.txt Generator")
st.write(
    "input your website main domain. Like https://bossdata.be"
)
url = st.text_input("Website domain:")

if st.button("Run"):
    st.write("The app is now running!")
    output_container = st.empty()

    sitemap_url = url + "page-sitemap.xml"
    #urls = generator.extract_urls_from_sitemap(sitemap_url)
    #output_container.write(urls)


