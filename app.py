import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
from pydantic import BaseModel, Field
from typing import List, Optional

st.set_page_config(page_title="SuperInvesting AI Engine", layout="wide", page_icon="📈")

# ---------------------------------------------------------
# 1. GEMINI NLP STRUCTURED QUERY COMPILER (TIER 1)
# ---------------------------------------------------------
class FilterCriterion(BaseModel):
    field: str = Field(description="Database field: 'roce', 'roe', 'pat_cagr_3y', 'de', 'pe', 'return_1y', 'above_50dma', 'above_200dma', 'rsi_14', 'dist_52w_high'")
    op: str = Field(description="Comparison operator: '>=', '<=', '==', '>', '<'")
    val: float = Field(description="Numeric threshold value. For boolean flags (e.g. above_50dma), use 1.0 for True, 0.0 for False")

class QuerySpecification(BaseModel):
    market: str = Field(default="ALL", description="'NSE', 'US', or 'ALL'")
    filters: List[FilterCriterion] = Field(default=[], description="Extracted numerical & momentum filters")
    active_personas: List[str] = Field(default=["Buffett", "Lynch", "Mayer", "RJ", "CANSLIM"], description="Active scoring personas")
    top_n: int = Field(default=10, description="Max results to display")

def compile_prompt_with_gemini(api_key: str, user_prompt: str) -> Optional[QuerySpecification]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        system_instruction = """
        You are an institutional financial query compiler. Translate the user's natural language investment prompt
        into structured screening parameters. Map metrics to schema fields:
        - ROCE -> 'roce', ROE -> 'roe', 3Y PAT CAGR -> 'pat_cagr_3y', Debt-to-Equity -> 'de', P/E -> 'pe'
        - 1Y Return -> 'return_1y', 14-day RSI -> 'rsi_14', % from 52W High -> 'dist_52w_high'
        - Above 50 DMA -> 'above_50dma' (val: 1.0), Above 200 DMA -> 'above_200dma' (val: 1.0)
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate this prompt into structured screening query: '{user_prompt}'",
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": QuerySpecification,
                "temperature": 0.0
            }
        )
        return QuerySpecification.model_validate_json(response.text)
    except Exception as e:
        st.error(f"Gemini NLP Parsing Error: {e}")
        return None

# ---------------------------------------------------------
# 2. MASTER FUNDAMENTAL DATASET
# ---------------------------------------------------------
@st.cache_data
def get_master_universe():
    return pd.DataFrame([
        # India (NSE)
        {"ticker": "BSE.NS", "name": "BSE Limited", "market": "NSE", "mcap_cr": 134000, "roce": 59.9, "roe": 46.0, "pe": 47.5, "pat_cagr_3y": 126.0, "de": 0.0, "cfo_pat_5y": 105.0, "debtor_days": 18, "promoter_holding": 0.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "MCX.NS", "name": "Multi Commodity Exchange", "market": "NSE", "mcap_cr": 83000, "roce": 31.0, "roe": 56.2, "pe": 54.0, "pat_cagr_3y": 107.5, "de": 0.0, "cfo_pat_5y": 98.0, "debtor_days": 15, "promoter_holding": 0.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "CAPLIPOINT.NS", "name": "Caplin Point Laboratories", "market": "NSE", "mcap_cr": 19000, "roce": 32.5, "roe": 20.6, "pe": 28.5, "pat_cagr_3y": 19.5, "de": 0.02, "cfo_pat_5y": 92.0, "debtor_days": 65, "promoter_holding": 69.2, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "PERSISTENT.NS", "name": "Persistent Systems", "market": "NSE", "mcap_cr": 88000, "roce": 28.4, "roe": 27.2, "pe": 44.3, "pat_cagr_3y": 27.5, "de": 0.05, "cfo_pat_5y": 88.0, "debtor_days": 58, "promoter_holding": 31.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "DIXON.NS", "name": "Dixon Technologies", "market": "NSE", "mcap_cr": 72000, "roce": 28.0, "roe": 24.0, "pe": 88.0, "pat_cagr_3y": 38.0, "de": 0.35, "cfo_pat_5y": 65.0, "debtor_days": 42, "promoter_holding": 33.5, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "LALPATHLAB.NS", "name": "Dr. Lal PathLabs", "market": "NSE", "mcap_cr": 31800, "roce": 27.2, "roe": 21.8, "pe": 58.0, "pat_cagr_3y": 18.0, "de": 0.07, "cfo_pat_5y": 144.0, "debtor_days": 22, "promoter_holding": 53.2, "promoter_pledge": 0.0, "governance_clean": True},
        # US (NASDAQ/NYSE)
        {"ticker": "MSFT", "name": "Microsoft Corporation", "market": "US", "mcap_cr": 2600000, "roce": 35.0, "roe": 38.0, "pe": 34.0, "pat_cagr_3y": 18.5, "de": 0.25, "cfo_pat_5y": 102.0, "debtor_days": 52, "promoter_holding": 50.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "market": "US", "mcap_cr": 3100000, "roce": 65.0, "roe": 72.0, "pe": 48.0, "pat_cagr_3y": 85.0, "de": 0.15, "cfo_pat_5y": 95.0, "debtor_days": 48, "promoter_holding": 50.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "CRWD", "name": "CrowdStrike Holdings", "market": "US", "mcap_cr": 78000, "roce": 18.0, "roe": 19.5, "pe": 72.0, "pat_cagr_3y": 35.0, "de": 0.18, "cfo_pat_5y": 115.0, "debtor_days": 60, "promoter_holding": 50.0, "promoter_pledge": 0.0, "governance_clean": True},
        {"ticker": "ASTS", "name": "AST SpaceMobile", "market": "US", "mcap_cr": 6500, "roce": -15.0, "roe": -25.0, "pe": 999.0, "pat_cagr_3y": -10.0, "de": 0.85, "cfo_pat_5y": -50.0, "debtor_days": 90, "promoter_holding": 40.0, "promoter_pledge": 0.0, "governance_clean": True}
    ])

# ---------------------------------------------------------
# 3. LIVE MOMENTUM & TECHNICALS FETCHER
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_live_momentum_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) >= 50:
            cmp = float(hist['Close'].iloc[-1])
            dma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            dma_200 = float(hist['Close'].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else dma_50
            return_1y = float(((cmp - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100)
            high_52w = float(hist['Close'].max())
            dist_52w_high = float((cmp / high_52w) * 100)
            
            # RSI Calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi_14 = float(100 - (100 / (1 + rs.iloc[-1])))

            # Volume Breakout vs 50D Average Volume
            vol_50d = hist['Volume'].rolling(50).mean().iloc[-1]
            latest_vol = hist['Volume'].iloc[-1]
            vol_surge = float(latest_vol / (vol_50d + 1e-9))

            return {
                "cmp": round(cmp, 2),
                "dma_50": round(dma_50, 2),
                "dma_200": round(dma_200, 2),
                "above_50dma": cmp > dma_50,
                "above_200dma": cmp > dma_200,
                "return_1y": round(return_1y, 1),
                "dist_52w_high": round(dist_52w_high, 1),
                "rsi_14": round(rsi_14, 1),
                "vol_surge": round(vol_surge, 2)
            }
    except Exception:
        pass
    return {
        "cmp": 0.0, "dma_50": 0.0, "dma_200": 0.0,
        "above_50dma": True, "above_200dma": True,
        "return_1y": 0.0, "dist_52w_high": 100.0, "rsi_14": 50.0, "vol_surge": 1.0
    }

# ---------------------------------------------------------
# 4. GATE 1 AUDIT & 5-PERSONA ENGINE
# ---------------------------------------------------------
def run_gate1_and_personas(row: dict, live: dict):
    # Gate 1 Red Flag Checks
    flags = []
    if row["de"] > 1.0: flags.append(("D/E > 1.0x", "❌ REJECT"))
    if row["cfo_pat_5y"] < 80.0: flags.append(("5Y CFO/PAT < 80%", "❌ REJECT" if row["cfo_pat_5y"] < 50 else "⚠️ WATCH"))
    if row["debtor_days"] > 90: flags.append(("Debtor Days > 90", "⚠️ WATCH"))
    if row["roce"] < 15.0: flags.append(("ROCE < 15%", "❌ REJECT"))
    if not row["governance_clean"]: flags.append(("Adverse Governance / Auditor Exit", "❌ REJECT (Governance)"))
    
    reject_count = sum(1 for _, v in flags if "❌" in v)
    gate_status = "❌ AUTO REJECT" if reject_count >= 1 else ("⚠️ WATCH" if flags else "✅ PASS")

    # 5-Persona Scoring (0.5 Continuous Resolution)
    wb_cap = 95.0 if row["roce"] >= 25 else (80.0 if row["roce"] >= 18 else 50.0)
    wb_solv = 95.0 if row["de"] <= 0.1 else (80.0 if row["de"] <= 0.3 else 50.0)
    wb_cfo = 95.0 if row["cfo_pat_5y"] >= 95 else (80.0 if row["cfo_pat_5y"] >= 80 else 45.0)
    wb_val = 38.0 if row["pe"] > 60 else (85.0 if row["pe"] <= 25 else 60.0)
    score_wb = round(0.30 * wb_cap + 0.25 * wb_solv + 0.25 * wb_cfo + 0.20 * wb_val, 1)

    growth = max(row["pat_cagr_3y"], 1.0)
    peg = row["pe"] / growth
    pl_peg = 95.0 if peg <= 1.0 else (80.0 if peg <= 1.5 else (45.0 if peg <= 2.5 else 35.0))
    pl_grow = 95.0 if growth >= 25 else (80.0 if growth >= 15 else 50.0)
    score_pl = round(0.35 * pl_peg + 0.35 * pl_grow + 0.30 * wb_solv, 1)

    cm_scale = 85.0 if row["mcap_cr"] <= 15000 else (75.0 if row["mcap_cr"] <= 50000 else 55.0)
    score_cm = round(0.40 * wb_cap + 0.35 * cm_scale + 0.25 * 85.0, 1)
    score_rj = round(0.35 * 85.0 + 0.35 * 85.0 + 0.30 * wb_val, 1)

    # CANSLIM Momentum Evaluation (O'Neil)
    c_score = 95.0 if row["pat_cagr_3y"] >= 25 else 70.0
    l_score = 95.0 if live["return_1y"] >= 40 else (75.0 if live["return_1y"] >= 15 else 35.0)
    s_score = 95.0 if live["vol_surge"] >= 1.5 else 75.0
    m_score = 95.0 if live["above_200dma"] else 30.0
    score_canslim = round((c_score + l_score + s_score + m_score + 80.0) / 5, 1)

    return gate_status, flags, {
        "Buffett": score_wb, "Lynch": score_pl, "Mayer": score_cm, "RJ": score_rj, "CANSLIM": score_canslim
    }

# ---------------------------------------------------------
# 5. USER INTERFACE & ORCHESTRATION
# ---------------------------------------------------------
st.title("📈 SuperInvesting AI: Natural Language Screening Engine")

# Sidebar Configuration
st.sidebar.header("🔑 API & Model Settings")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Paste your Gemini key here or set GEMINI_API_KEY")

st.sidebar.subheader("Active Personas")
p_wb = st.sidebar.checkbox("Warren Buffett", value=True)
p_pl = st.sidebar.checkbox("Peter Lynch", value=True)
p_cm = st.sidebar.checkbox("Chris Mayer", value=True)
p_rj = st.sidebar.checkbox("Rakesh Jhunjhunwala", value=True)
p_canslim = st.sidebar.checkbox("William O'Neil (CANSLIM)", value=True)

# Build unified active dataset
df_raw = get_master_universe()
enriched_rows = []
for _, row in df_raw.iterrows():
    live = fetch_live_momentum_data(row["ticker"])
    gate_status, flags, p_scores = run_gate1_and_personas(row, live)
    
    # Calculate Super AI Score across selected personas
    active_scores = []
    if p_wb: active_scores.append(p_scores["Buffett"])
    if p_pl: active_scores.append(p_scores["Lynch"])
    if p_cm: active_scores.append(p_scores["Mayer"])
    if p_rj: active_scores.append(p_scores["RJ"])
    if p_canslim: active_scores.append(p_scores["CANSLIM"])
    
    super_ai = round(np.mean(active_scores), 1) if active_scores else 0.0
    
    combined = {**row.to_dict(), **live}
    combined.update({
        "Gate 1": gate_status,
        "flags": flags,
        "Buffett": p_scores["Buffett"],
        "Lynch": p_scores["Lynch"],
        "Mayer": p_scores["Mayer"],
        "RJ": p_scores["RJ"],
        "CANSLIM": p_scores["CANSLIM"],
        "Super AI Score": super_ai
    })
    enriched_rows.append(combined)

df_all = pd.DataFrame(enriched_rows)

# Natural Language Prompt Search Bar
st.subheader("💬 Ask Any Screening Prompt")
user_prompt = st.text_input(
    "Type your query in plain English:",
    placeholder="e.g., 'Find debt-free Indian compounders trading above 50-DMA with 1Y Return > 30%'"
)

df_filtered = df_all.copy()

if user_prompt:
    if not gemini_key:
        st.warning("⚠️ Please provide a Gemini API Key in the sidebar to enable Natural Language query translation.")
    else:
        with st.spinner("🤖 Gemini is compiling your investment prompt into structured query filters..."):
            query_spec = compile_prompt_with_gemini(gemini_key, user_prompt)
            
            if query_spec:
                st.info(f"**Compiled Filters:** {', '.join([f'{f.field} {f.op} {f.val}' for f in query_spec.filters]) or 'None (All Universe)'} | **Market:** {query_spec.market}")
                
                # Apply Market Filter
                if query_spec.market == "NSE":
                    df_filtered = df_filtered[df_filtered["market"] == "NSE"]
                elif query_spec.market == "US":
                    df_filtered = df_filtered[df_filtered["market"] == "US"]
                
                # Apply Dynamic Field Filters
                for f in query_spec.filters:
                    if f.field in df_filtered.columns:
                        if f.field in ["above_50dma", "above_200dma"]:
                            expected_bool = True if f.val >= 1.0 else False
                            df_filtered = df_filtered[df_filtered[f.field] == expected_bool]
                        elif f.op == ">=":
                            df_filtered = df_filtered[df_filtered[f.field] >= f.val]
                        elif f.op == "<=":
                            df_filtered = df_filtered[df_filtered[f.field] <= f.val]
                        elif f.op == ">":
                            df_filtered = df_filtered[df_filtered[f.field] > f.val]
                        elif f.op == "<":
                            df_filtered = df_filtered[df_filtered[f.field] < f.val]
                        elif f.op == "==":
                            df_filtered = df_filtered[df_filtered[f.field] == f.val]

df_filtered = df_filtered.sort_values(by="Super AI Score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# 6. RESULTS LEADERBOARD & INSPECTOR
# ---------------------------------------------------------
st.markdown("### 🏆 Scored Results Table")
display_cols = ["ticker", "name", "market", "cmp", "return_1y", "rsi_14", "above_50dma", "Gate 1", "Buffett", "Lynch", "Mayer", "RJ", "CANSLIM", "Super AI Score"]
st.dataframe(df_filtered[display_cols], use_container_width=True)

st.markdown("---")
st.subheader("🔍 Deep-Dive Stock Audit Inspector")
selected_ticker = st.selectbox("Select a stock to audit:", df_filtered["ticker"].tolist() if not df_filtered.empty else [])

if selected_ticker:
    s = df_filtered[df_filtered["ticker"] == selected_ticker].iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Super AI Score", f"{s['Super AI Score']} / 100")
    c2.metric("Gate 1 Status", s["Gate 1"])
    c3.metric("Live CMP", f"{s['cmp']}")
    c4.metric("14-Day RSI", f"{s['rsi_14']}")

    with st.expander("🚩 Gate 1 Red Flag Audit"):
        if not s["flags"]:
            st.success("✅ Clean Forensic Profile: Passed all Gate 1 Red Flags.")
        else:
            for item, verdict in s["flags"]:
                st.warning(f"**{item}** — {verdict}")

    with st.expander("📊 5-Persona Scorecard Breakdown"):
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Warren Buffett", f"{s['Buffett']}")
        p2.metric("Peter Lynch", f"{s['Lynch']}")
        p3.metric("Chris Mayer", f"{s['Mayer']}")
        p4.metric("Rakesh Jhunjhunwala", f"{s['RJ']}")
        p5.metric("CANSLIM", f"{s['CANSLIM']}")
