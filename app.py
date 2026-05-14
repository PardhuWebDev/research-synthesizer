import sys
sys.path.append(".")

import streamlit as st
from graph.graph import pipeline
from graph.state import ResearchState

st.set_page_config(
    page_title="Research Synthesizer",
    page_icon="🔬",
    layout="wide"
)

# ── Styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }

    .title {
        font-size: 2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 1.5px solid #E5E7EB;
    }
    .stat-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 600;
        color: #111827;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #6B7280;
        margin-top: 0.1rem;
    }
    .conflict-card {
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-left: 4px solid #F97316;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .conflict-card.high {
        background: #FEF2F2;
        border-color: #FECACA;
        border-left-color: #EF4444;
    }
    .conflict-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #EF4444;
        margin-bottom: 0.5rem;
    }
    .claim-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .claim-pill {
        background: #ffffff;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        color: #374151;
        flex: 1;
    }
    .paper-tag {
        font-size: 0.72rem;
        color: #9CA3AF;
        margin-bottom: 0.2rem;
    }
    .report-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown('<div class="title">Research Synthesizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a research topic to fetch papers, extract claims, and detect contradictions.</div>', unsafe_allow_html=True)

# ── Search ───────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("", placeholder="e.g. transformer attention mechanisms, diffusion models, RAG systems", label_visibility="collapsed")
with col2:
    search = st.button("Analyze", use_container_width=True, type="primary")

if search and query.strip():
    with st.spinner("Running pipeline — fetching papers, extracting claims, detecting contradictions..."):
        initial_state: ResearchState = {
            "query": query.strip(),
            "papers": [],
            "chunks": [],
            "retrieved": [],
            "claims": {},
            "contradictions": [],
            "report": "",
            "error": None
        }
        result = pipeline.invoke(initial_state)

    total_claims = sum(len(c) for c in result["claims"].values())

    # ── Stats ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(result["papers"])}</div><div class="stat-label">Papers Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total_claims}</div><div class="stat-label">Claims Extracted</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(result["contradictions"])}</div><div class="stat-label">Contradictions Found</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(result["retrieved"])}</div><div class="stat-label">Chunks Retrieved</div></div>', unsafe_allow_html=True)

    # ── Report ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Research Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-box">{result["report"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    # ── Contradictions ────────────────────────────────────────────
    if result["contradictions"]:
        st.markdown('<div class="section-title">Detected Contradictions</div>', unsafe_allow_html=True)
        for c in result["contradictions"]:
            level = "high" if c["confidence"] == "HIGH" else ""
            st.markdown(f"""
            <div class="conflict-card {level}">
                <div class="conflict-label">{c["confidence"]} CONFIDENCE &nbsp;·&nbsp; Score: {c["score"]}</div>
                <div class="claim-row">
                    <div class="claim-pill">
                        <div class="paper-tag">{c["paper_a_title"][:60]}</div>
                        {c["claim_a"]}
                    </div>
                    <div class="claim-pill">
                        <div class="paper-tag">{c["paper_b_title"][:60]}</div>
                        {c["claim_b"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">Detected Contradictions</div>', unsafe_allow_html=True)
        st.info("No contradictions detected in this corpus.")

    # ── Papers ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Papers Analyzed</div>', unsafe_allow_html=True)
    for p in result["papers"]:
        with st.expander(f"{p['title']} ({p['published']})"):
            st.markdown(f"**Authors:** {', '.join(p['authors'][:3])}")
            st.markdown(f"**Abstract:** {p['abstract'][:400]}...")
            st.markdown(f"[View on Arxiv]({p['url']})")
            claims = result["claims"].get(p["id"], [])
            if claims:
                st.markdown("**Extracted Claims:**")
                for i, claim in enumerate(claims, 1):
                    st.markdown(f"{i}. {claim}")

elif search and not query.strip():
    st.warning("Please enter a research topic.")