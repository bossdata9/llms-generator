import streamlit as st
import asyncio
from openai import OpenAI, AsyncOpenAI
import llms_generator as generator

# =========================
# ========== SETUP ========
# =========================

# Global styles
st.markdown(
    """
    <style>
    /* Import Custom Font */
    @import url('https://db.onlinewebfonts.com/c/ac9525e5f200f57332b3080d0db9d8f6?family=Sailec+Medium');

    /* App base */
    .stApp {
        background-color: #ffffff;
        font-family: "Sailec Medium", Helvetica, Arial, sans-serif;
    }

    /* Global text color */
    .stApp :is(h1,h2,h3,h4,h5,h6,p,label,li,strong,em) {
        color: #48546e;
    }

    /* Inputs */
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

    /* Buttons */
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

# Sync OpenAI client (used for site description in Phase 3)
client = OpenAI(api_key=st.secrets["openai_api_key"])

# =========================
# ========= APP UI ========
# =========================

st.image("bossdata.svg", width=200)
st.title("Llms.txt Generator")

# --- Box: Why & Instructions ---
with st.container(border=True):
    with st.expander("Why llms.txt?", expanded=False):
        st.markdown("""
**What it is**  
A human-curated, machine-readable index that tells AI systems which pages best explain your site.

**Why it helps**
- **Curate what AI reads:** Highlight canonical pages (categories, key content) and hide noisy utility pages.
- **Better AI answers:** Concise titles/descriptions reduce hallucinations and misclassification about your brand/products.
- **Faster ingestion:** One lightweight file is easier for AI crawlers (and your own pipelines) than scraping the whole site.
- **Consistent taxonomy:** Cluster labels provide a simple information architecture AI can follow.
- **Brand safety & governance:** Exclude experimental/sensitive/low-value URLs.
- **Easy to maintain:** Plain text, versionable, lives at `/llms.txt`.

**Heads-up**
- Not an official web standard. Treat it as guidance for AI tools; still keep sitemaps and meta tags up to date.
- Re-generate after major site updates.
""")

    with st.expander("Instructions", expanded=True):
        st.markdown("""
1. **Get your sitemap and download it**  
   Find your site’s sitemap(s), e.g. `/sitemap.xml`, `/page-sitemap.xml`, `/posts-sitemap.xml`. Make sure it is complete and download it.

2. **Prune irrelevant URLs**  
   Delete utility/boilerplate pages: login/admin/account, cookie policy, privacy, terms, sitemap, 404, search, tag/archive, cart/checkout, thank-you.

3. **Remove product detail pages**  
   If you have many PDPs/variants, remove them. **Keep category/collection pages** instead (that’s sufficient).

4. **Save your cleaned sitemap**  
   Save as UTF-8 XML (e.g. `sitemap_llms.xml`).

5. **Use it as input**  
   Upload the cleaned sitemap below and click **Generate** (optionally set “Amount of URLs”).

6. **Review the result**  
   Sanity-check titles, descriptions, and clusters. Edit if needed; the output won’t be 100% perfect.

7. **Publish**  
   Give `llms.txt` to your developer to place at the **domain root**, e.g. `https://yourdomain.com/llms.txt` (must return **HTTP 200 OK**).
""")

# --- Box: Run the generator (inputs) ---
with st.container(border=True):
    st.subheader("Run the generator")
    uploaded_file = st.file_uploader("Upload sitemap XML file", type=["xml"])
    sitemap_location = uploaded_file if uploaded_file is not None else None
    size_input = st.text_input("Amount of urls (leave blank for max - 10000):", placeholder="10000")
    generate_clicked = st.button("Generate", key="generate_btn")

# =========================
# ====== APP LOGIC ========
# =========================

if generate_clicked:
    if sitemap_location is None:
        st.warning("Please upload a sitemap XML first.")
    else:
        # -----------------------
        # Phase 1 — Fetch URLs
        # -----------------------
        with st.container(border=True):
            st.subheader("Phase 1 — Fetch URLs")
            try:
                size = 10000 if (size_input or "").strip() == "" else min(int(size_input), 10000)
            except Exception:
                st.warning("Invalid size. Using default 10000.")
                size = 10000

            st.write(f"Fetching up to **{size}** URLs from uploaded sitemap…")
            urls = generator.extract_urls_from_sitemap(sitemap_location)
            selected_urls = urls[:int(size)]
            st.caption(f"Found {len(urls)} total; processing top {len(selected_urls)}.")

        # -----------------------
        # Phase 2 — Summaries
        # -----------------------
        with st.container(border=True):
            st.subheader("Phase 2 — Generate titles & descriptions")
            st.write(f"Generating summaries for **{len(selected_urls)}** URLs (async + batched)…")

            total = len(selected_urls)
            if total == 0:
                st.warning("No URLs to process. Please check your sitemap.")
            else:
                batch_size = 20  # tweak to your liking
                progress = st.progress(0)
                status = st.empty()

                results = []
                for i in range(0, total, batch_size):
                    batch = selected_urls[i:i+batch_size]
                    # Async summarizer for this batch
                    batch_results = asyncio.run(
                        generator.summarize_urls_async(
                            batch,
                            openai_api_key=st.secrets["openai_api_key"],
                            max_concurrency=8,
                            model="gpt-4.1-mini",
                        )
                    )
                    results.extend(batch_results)

                    done = min(i + batch_size, total)
                    progress.progress(done / total)
                    status.write(f"Phase 2: {done}/{total} pages summarized")

        # -----------------------
        # Phase 3 — Cluster & Build llms.txt
        # -----------------------
        with st.container(border=True):
            st.subheader("Phase 3 — Clustering & building llms.txt")

            USE_EMBEDDINGS = True
            EMBED_MODEL = "text-embedding-3-large"
            SIM_THRESHOLD = 0.86  # more merges; tweak if needed

            async def _run_phase3_build():
                # Create/close async client *inside* the loop to avoid cleanup warnings
                async with AsyncOpenAI(api_key=st.secrets["openai_api_key"]) as client_async:
                    return await generator.build_llms_txt_from_results_async(
                        results=results,
                        client_sync=client,        # sync client (site description)
                        client_async=client_async, # async client (embeddings + labels)
                        USE_EMBEDDINGS=USE_EMBEDDINGS,
                        EMBED_MODEL=EMBED_MODEL,
                        SIM_THRESHOLD=SIM_THRESHOLD,
                    )

            with st.spinner("Clustering pages and generating llms.txt…"):
                llms_output = asyncio.run(_run_phase3_build())

            st.success("Final llms.txt is ready. Download below.")
            st.download_button(
                label="Download llms.txt",
                data=llms_output,
                file_name="llms.txt",
                mime="text/plain"
            )
