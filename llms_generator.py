# -*- coding: utf-8 -*-
"""LLMS.TXT Generator (async, trimmed)"""

# ======================================================================
# Imports
# ======================================================================
import xml.etree.ElementTree as ET
import json
from openai import OpenAI          # sync client used for site description
from openai import AsyncOpenAI     # async client used for summaries, embeddings, labels
import re
from urllib.parse import urlparse
from collections import defaultdict
import math
from datetime import datetime
import asyncio

# ----------------------------------------------------------------------
# Tiny logger helper (prints to console with timestamps)
# ----------------------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ======================================================================
# Core helpers
# ======================================================================

def _local_name(tag: str) -> str:
    """Return the local name without namespace, e.g. '{ns}url' -> 'url'."""
    return tag.split('}', 1)[-1] if '}' in tag else tag

def extract_urls_from_sitemap(xml_file):
    """
    Parse a sitemap XML and return all <loc> values that are direct children of <url>.
    Accepts a file-like object (e.g., from Streamlit) or a string path.
    """
    log("extract_urls_from_sitemap: start")
    # Read bytes from file-like or path
    if hasattr(xml_file, "read"):
        log("extract_urls_from_sitemap: detected file-like object")
        xml_bytes = xml_file.read()
    else:
        log(f"extract_urls_from_sitemap: reading from path: {xml_file}")
        with open(xml_file, "rb") as f:
            xml_bytes = f.read()

    size_kb = round(len(xml_bytes) / 1024, 2)
    log(f"extract_urls_from_sitemap: loaded XML ({size_kb} KB)")

    # Parse XML
    try:
        root = ET.fromstring(xml_bytes)
        log("extract_urls_from_sitemap: XML parsed successfully")
    except ET.ParseError as e:
        log(f"extract_urls_from_sitemap: XML parse error -> {e}")
        raise ValueError(f"Failed to parse XML: {e}") from e

    urls = []
    url_nodes = 0
    for url_el in root.iter():
        if _local_name(url_el.tag) != "url":
            continue
        url_nodes += 1
        # Direct child <loc>
        for child in list(url_el):
            if _local_name(child.tag) == "loc" and child.text:
                loc = child.text.strip()
                if loc:
                    urls.append(loc)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    log(f"extract_urls_from_sitemap: found {len(urls)} <loc> across {url_nodes} <url> nodes; {len(deduped)} unique URLs")
    return deduped

# ======================================================================
# Async summarization (URL -> short title/description)
# ======================================================================

async def describe_page_async(
    url: str,
    client_async: AsyncOpenAI,
    meta_title: str = "",
    h1: str = "",
    model: str = "gpt-4.1-mini",
):
    log(f"describe_page_async: summarizing {url} with model={model}")

    base_prompt = f"""You will summarize the page for: {url}

Rules:
- Focus only on the page’s primary content (topic, offering, purpose).
- Ignore cookie banners, privacy/legal boilerplate, navigation, and footers.
- Prefer signals from the HTML <title> and H1 if available.

Constraints:
- Title: exactly 3–4 words, English.
- Description: exactly 9–10 words, plain, neutral English.
Return JSON with keys: title, description.

Signals available:
- HTML title: {meta_title}
- H1: {h1}
"""
    try:
        resp = await client_async.responses.create(
            model=model,
            input=[{"role": "user", "content": base_prompt}],
            temperature=0.3,
        )
        txt = (resp.output_text or "").strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
            txt = re.sub(r"\n?```$", "", txt)
            txt = txt.strip()

        data = json.loads(txt)
        title = (data.get("title") or "").strip()
        desc  = (data.get("description") or "").strip()
        log(f"describe_page_async: ok -> title='{title}' | desc='{desc}'")
        return {"url": url, "title": title, "description": desc}
    except Exception as e:
        log(f"describe_page_async: ERROR parsing model output -> {e}")
        return {"url": url, "title": "", "description": ""}

async def summarize_urls_async(
    urls: list[str],
    openai_api_key: str,
    max_concurrency: int = 8,
    model: str = "gpt-4.1-mini",
) -> list[dict]:
    """
    Summarize many URLs concurrently. Creates/closes the async client inside to avoid
    event-loop-close warnings.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async with AsyncOpenAI(api_key=openai_api_key) as client_async:
        async def one(u: str):
            async with sem:
                return await describe_page_async(u, client_async, model=model)

        tasks = [asyncio.create_task(one(u)) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log(f"summarize_urls_async: ERROR for {urls[i]} -> {r}")
            out.append({"url": urls[i], "title": "", "description": ""})
        else:
            out.append(r)
    return out

# ======================================================================
# Site description (sync)
# ======================================================================

def generate_site_description(results, client: OpenAI, model="gpt-4.1-mini"):
    log(f"generate_site_description: start with {len(results)} results, model={model}")
    if not results: 
        log("generate_site_description: empty results -> fallback description")
        return "General Website", "No description available."
    first_url = results[0]["url"]
    domain = urlparse(first_url).netloc.lower()
    sample_text = "\n".join(
        f"- {r.get('title','') or ''} — {r.get('description','') or ''}"
        for r in results[:30]
    )
    prompt = f"""
You are given a list of page titles and descriptions from the website {domain}.

Write a concise general description (2–3 sentences, neutral English) that summarizes
the main purpose of the site as a whole.

Return only the description text, no extra formatting.

Pages:
{sample_text}
"""
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        site_description = resp.output_text.strip()
        log(f"generate_site_description: ok for domain={domain}")
        return domain, site_description
    except Exception as e:
        log(f"generate_site_description: ERROR -> {e}")
        return domain, "No description available."

# ======================================================================
# Clustering helpers
# ======================================================================

def url_first_segment(u: str) -> str:
    p = urlparse(u)
    segs = [s for s in p.path.split("/") if s]
    return segs[0].lower() if segs else ""

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def dot(a, b): return sum(x*y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a)) or 1e-12
def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))

def represent_item(item):
    return normalize_space(f'{item.get("title","")} — {item.get("description","")}')

def cluster_centroid(vecs):
    if not vecs: return []
    dim = len(vecs[0])
    s = [0.0]*dim
    for v in vecs:
        for i in range(dim): s[i] += v[i]
    return [x/len(vecs) for x in s]

def seed_clusters(results):
    log(f"seed_clusters: bucketing {len(results)} items by first path segment")
    buckets = defaultdict(list)
    for item in results:
        seg = url_first_segment(item.get("url","")) or "root"
        buckets[seg].append(item)
    sizes = {k: len(v) for k, v in buckets.items()}
    log(f"seed_clusters: {len(buckets)} buckets -> sizes {sizes}")
    return [items for _, items in sorted(buckets.items(), key=lambda kv: kv[0])]

# ======================================================================
# Async embeddings + merge
# ======================================================================

async def embed_texts_async(texts, client_async: AsyncOpenAI, EMBED_MODEL: str):
    log(f"embed_texts_async: embedding {len(texts)} texts with model={EMBED_MODEL}")
    resp = await client_async.embeddings.create(model=EMBED_MODEL, input=texts)
    log("embed_texts_async: embeddings created")
    return [d.embedding for d in resp.data]

async def semantically_merge_async(clusters, client_async: AsyncOpenAI, EMBED_MODEL: str, SIM_THRESHOLD: float):
    log(f"semantically_merge_async: start with {len(clusters)} clusters | threshold={SIM_THRESHOLD} | model={EMBED_MODEL}")
    texts = []
    idx_map = []
    for ci, items in enumerate(clusters):
        for ii, it in enumerate(items):
            texts.append(represent_item(it))
            idx_map.append((ci, ii))
    if not texts:
        log("semantically_merge_async: no texts -> return original clusters")
        return clusters

    vecs = await embed_texts_async(texts, client_async, EMBED_MODEL)
    per_cluster_vecs = defaultdict(list)
    for (ci, _), v in zip(idx_map, vecs):
        per_cluster_vecs[ci].append(v)
    centroids = [cluster_centroid(per_cluster_vecs[i]) for i in range(len(clusters))]

    merged = []
    used = set()
    merge_ops = []
    for i in range(len(clusters)):
        if i in used: 
            continue
        base_items = list(clusters[i])
        base_centroid = centroids[i]
        used.add(i)
        for j in range(i+1, len(clusters)):
            if j in used or not base_centroid or not centroids[j]:
                continue
            sim = cosine(base_centroid, centroids[j])
            if sim >= SIM_THRESHOLD:
                merge_ops.append((i, j, round(sim, 4)))
                base_items.extend(clusters[j])
                used.add(j)
                base_centroid = cluster_centroid([base_centroid, centroids[j]])
        merged.append(base_items)

    if merge_ops:
        log(f"semantically_merge_async: merged pairs {merge_ops}")
    else:
        log("semantically_merge_async: no merges performed")

    log(f"semantically_merge_async: end -> {len(merged)} clusters")
    return merged

# ======================================================================
# Async labeling + rendering
# ======================================================================

async def cluster_label_from_items_async(
    items,
    client_async: AsyncOpenAI,
    fallback: str = "Content",
    model: str = "gpt-4.1-mini",
) -> str:
    log(f"cluster_label_from_items_async: labeling {len(items)} items with model={model}")
    titles = [ (it.get("title","") or "").strip() for it in items if it.get("title") ]
    titles = [t for t in titles if t]
    if not titles:
        log("cluster_label_from_items_async: no titles -> fallback")
        return fallback

    titles_text = "\n".join(f"- {t}" for t in titles[:30])
    prompt = f"""
Given the following page titles, suggest a concise English cluster label (2–4 words)
that best represents them as a group.

Return only the label, with no quotes or extra text.

Titles:
{titles_text}
""".strip()

    try:
        resp = await client_async.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        label = (resp.output_text or "").strip()
        if not label:
            label = fallback
        log(f"cluster_label_from_items_async: label='{label}'")
        return label
    except Exception as e:
        log(f"cluster_label_from_items_async: ERROR -> {e} (fallback='{fallback}')")
        return fallback

async def label_clusters_async(
    clusters: list,
    client_async: AsyncOpenAI,
    model: str = "gpt-4.1-mini",
    max_concurrency: int = 8,
) -> list[str]:
    """Label all clusters concurrently; returns labels aligned to clusters order."""
    sem = asyncio.Semaphore(max_concurrency)

    async def one(idx: int, items):
        async with sem:
            label = await cluster_label_from_items_async(items, client_async, model=model)
            return (idx, label)

    tasks = [asyncio.create_task(one(i, items)) for i, items in enumerate(clusters)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    labels = ["Content"] * len(clusters)
    for r in results:
        if isinstance(r, Exception):
            continue
        i, label = r
        labels[i] = label or "Content"
    return labels

def write_llms_txt_with_labels(clusters, results, client: OpenAI, labels: list[str]):
    log(f"write_llms_txt_with_labels: rendering {sum(len(c) for c in clusters)} items across {len(clusters)} clusters")
    lines = []
    domain, site_description = generate_site_description(results, client)

    domain_line = f"# {domain.strip().capitalize()}"
    lines.append(domain_line)
    log(f"write_llms_txt_with_labels: header -> {domain_line}")

    if site_description and str(site_description).strip():
        lines.append("")
        lines.append(f"> {site_description.strip()}")
        lines.append("")
        log("write_llms_txt_with_labels: added site description")

    for idx, (cluster_items, label) in enumerate(zip(clusters, labels), start=1):
        label = label or "Content"
        lines.append(f"## Cluster: {label}")
        log(f"write_llms_txt_with_labels: cluster {idx} label='{label}' size={len(cluster_items)}")

        for it in cluster_items:
            url = (it.get("url", "") or "").strip()
            title = normalize_space(it.get("title", "") or "").strip()
            desc  = normalize_space(it.get("description", "") or "").strip()
            if not title:
                title = url
            bullet = f"- [{title}]({url})"
            if desc:
                bullet += f": {desc}"
            lines.append(bullet)
        lines.append("")

    output = "\n".join(lines)
    log(f"write_llms_txt_with_labels: done (length={len(output)} chars)")
    return output

# ======================================================================
# Async builder (embeddings + labeling async; site description sync)
# ======================================================================

async def build_llms_txt_from_results_async(
    results,
    client_sync: OpenAI,               # sync client for site description
    client_async: AsyncOpenAI,         # async client for embeddings + labels
    USE_EMBEDDINGS: bool,
    EMBED_MODEL: str,
    SIM_THRESHOLD: float,
    label_model: str = "gpt-4.1-mini",
    label_max_concurrency: int = 8,
):
    log(f"build_llms_txt_from_results_async: start | results={len(results)} | use_embeddings={USE_EMBEDDINGS} | model={EMBED_MODEL} | thr={SIM_THRESHOLD}")
    clusters = seed_clusters(results)
    total_items = sum(len(c) for c in clusters)
    log(f"build_llms_txt_from_results_async: seeded {len(clusters)} clusters with {total_items} items total")

    if USE_EMBEDDINGS and total_items > 1:
        clusters = await semantically_merge_async(clusters, client_async, EMBED_MODEL, SIM_THRESHOLD)

    for c in clusters:
        c.sort(key=lambda it: it.get("url", ""))

    # Label clusters concurrently
    labels = await label_clusters_async(
        clusters,
        client_async=client_async,
        model=label_model,
        max_concurrency=label_max_concurrency,
    )

    log("build_llms_txt_from_results_async: clusters sorted & labeled")
    final_txt = write_llms_txt_with_labels(clusters, results, client=client_sync, labels=labels)
    log("build_llms_txt_from_results_async: completed llms.txt build")
    return final_txt
