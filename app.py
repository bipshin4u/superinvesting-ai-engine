import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List, Optional

st.set_page_config(page_title="SuperInvesting AI", layout="wide", page_icon="📈")

# ---------------------------------------------------------
# 1. NLP LAYER: GEMINI QUERY COMPILER 
# ---------------------------------------------------------
class FilterCriterion(BaseModel):
    field: str = Field(description="Fields: roce, roe, pat_cagr_3y, de, pe_ttm, return_1y, above_50dma")
    op: str = Field(description="Operators: '>=', '<=', '==', '>', '<'")
    val: float = Field(description="Numerical value")

class QuerySpecification(BaseModel):
    market: str = Field(default="ALL", description="'NSE', 'US', or 'ALL'")
    filters: List[FilterCriterion] = Field(default=[])
    active_personas: List[str] = Field(default=["Buffett", "Lynch", "Mayer", "RJ", "CANSLIM"])
    top_n: int = Field(default=10)
    # NEW: AI Synthesis Narrative for conviction
    synthesis_narrative: str = Field(default="", description="A 2-3 sentence fundamental analysis theory explaining why this specific screen identifies strong investments and creates conviction.")

def compile_prompt(api_key: str, prompt: str) -> Optional[QuerySpecification]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        sys_inst = "Translate the financial prompt into a structured JSON filter schema and provide a compelling investment thesis."
        response = client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=f"Prompt: {prompt}",
            config={"system_instruction": sys_inst, "response_mime_type": "application/json", "response_schema": QuerySpecification, "temperature": 0.2}
        )
        return QuerySpecification.model_validate_json(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# ---------------------------------------------------------
# 2. TICKER FORMATTING & MARKET SWITCH LOGIC
# ---------------------------------------------------------
def format_ticker(symbol: str, market_mode: str) -> str:
    """Normalizes symbols based on active market toggle, auto-appending .NS for India."""
    clean_sym = symbol.strip().upper()
    if market_mode == "India (NSE)":
        if not (clean_sym.endswith(".NS") or clean_sym.endswith(".BO")):
            return f"{clean_sym}.NS"
        return clean_sym
    else:
        return clean_sym.replace(".NS", "").replace(".BO", "")

def clean_display_ticker(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".BO", "")

# ---------------------------------------------------------
# 3. YFINANCE LIVE FETCHER
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_live_data(ticker: str) -> dict:
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) >= 50:
            cmp = float(hist['Close'].iloc[-1])
            dma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            dma_200 = float(hist['Close'].rolling(200).mean().iloc[-1]) if len(hist)>=200 else dma_50
            ret_1y = float(((cmp - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100)
            return {"cmp": round(cmp, 2), "return_1y": round(ret_1y, 1), "above_50dma": cmp > dma_50, "above_200dma": cmp > dma_200}
    except Exception:
        pass
    return {"cmp": 0.0, "return_1y": 0.0, "above_50dma": False, "above_200dma": False}

# ---------------------------------------------------------
# 4. SCORING ENGINE (GATE 1 + 5 PERSONAS)
# ---------------------------------------------------------
def score_stock(row: dict, live: dict):
    flags = []
    if row.get("market") == "NSE":
        if row.get("tax_rate", 25) < 25.0: flags.append(("Tax < 25%", "❌ REJECT"))
        if row.get("promoter_pledge", 0) > 5.0: flags.append(("Pledge > 5%", "❌ AUTO REJECT"))
    else:
        if row.get("tax_rate", 21) < 15.0: flags.append(("Tax < 15%", "❌ REJECT"))
        if row.get("sbc_dilution", 0) > 3.0: flags.append(("SBC Dilution > 3%", "❌ AUTO REJECT"))
    
    if row.get("de", 0) > 1.0: flags.append(("D/E > 1.0x", "❌ REJECT"))
    if row.get("roce", 15) < 15.0: flags.append(("ROCE < 15%", "❌ REJECT"))
    if row.get("cfo_pat_5y", 100) < 80.0: flags.append(("CFO/PAT < 80%", "❌ REJECT"))
    
    rjct = sum(1 for _, v in flags if "❌" in v)
    gate = "❌ AUTO REJECT" if any("AUTO" in v for _, v in flags) else ("❌ REJECT" if rjct >= 2 else ("⚠️ WATCH" if flags else "✅ PASS"))

    wb = round(0.30*(95 if row.get("roce",0)>=25 else 50) + 0.25*(95 if row.get("de",1)<=0.1 else 50) + 0.25*(95 if row.get("cfo_pat_5y",0)>=95 else 45) + 0.20*(85 if row.get("pe_ttm",100)<=25 else 38), 1)
    lynch = round(0.35*(95 if (row.get("pe_ttm",100)/max(row.get("pat_cagr_3y",1),1))<=1.0 else 40) + 0.35*(95 if row.get("pat_cagr_3y",0)>=25 else 75) + 0.30*80, 1)
    mayer = round(0.40*(95 if row.get("roce",0)>=25 else 75) + 0.35*(85 if row.get("mcap",50000)<15000 else 55) + 0.25*80, 1)
    
    rj_gov = 85.0
    if row.get("market") == "US" and row.get("sbc_dilution",0) > 3.0: rj_gov = 45.0
    rj = round(0.35*85 + 0.35*85 + 0.30*rj_gov, 1)
    
    canslim = round(( (95 if row.get("pat_cagr_3y",0)>=30 else 75) + (95 if live.get("return_1y",0)>=50 else 35) + 80 + 80 + 80)/5, 1)

    return gate, flags, {"Buffett": wb, "Lynch": lynch, "Mayer": mayer, "RJ": rj, "CANSLIM": canslim}

# ---------------------------------------------------------
# 5. UI ORCHESTRATOR & TAB LAYOUT
# ---------------------------------------------------------
st.title("🏛️ Multibagger Investing & Scoring Engine")

# Securely load Gemini API Key
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("⚠️ Gemini API Key not found. Please add it to Streamlit's Advanced Settings > Secrets.")
    gemini_key = None

st.sidebar.header("🌐 Market & Personas")
market_choice = st.sidebar.radio("Select Active Market", ["India (NSE)", "US (NASDAQ/NYSE)"], horizontal=True)

act_wb = st.sidebar.checkbox("Warren Buffett", True)
act_pl = st.sidebar.checkbox("Peter Lynch", True)
act_cm = st.sidebar.checkbox("Chris Mayer", True)
act_rj = st.sidebar.checkbox("Rakesh Jhunjhunwala", True)
act_cs = st.sidebar.checkbox("CANSLIM", True)

# --- SECTION A: LIVE SINGLE STOCK AUDITOR ---
st.markdown("### 🔍 Live Single Stock Auditor")
user_input = st.text_input("Enter Ticker to audit instantly (e.g., RELIANCE, ZOMATO, MSFT):")

if user_input:
    query_ticker = format_ticker(user_input, market_choice)
    display_ticker = clean_display_ticker(user_input)
    st.info(f"Fetching live data for: **{display_ticker}** (Querying: `{query_ticker}`)")
    
    live_data = fetch_live_data(query_ticker)
    if live_data["cmp"] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live CMP", live_data["cmp"])
        c2.metric("1Y Return", f"{live_data['return_1y']}%")
        c3.metric("Above 50-DMA", "Yes" if live_data["above_50dma"] else "No")
        c4.metric("Above 200-DMA", "Yes" if live_data["above_200dma"] else "No")
    else:
        st.warning("Could not fetch data. Check if the ticker is correct for the selected market.")

st.markdown("---")

# --- SECTION B: NATURAL LANGUAGE SCREENER & TABS ---
st.markdown("### 💬 Natural Language AI Screener")
prompt = st.text_input("Ask any query (e.g., 'Find Indian stocks with ROCE > 25% and zero debt'):")

# Create Two Tabs for clean layout
tab_results, tab_master = st.tabs(["✨ AI Screening Results", "📚 Master Universe Data"])

@st.cache_data
def get_master_universe():
    return pd.DataFrame([
        {"ticker": "BSE.NS", "name": "BSE Limited", "market": "NSE", "mcap": 134000, "roce": 59.9, "roe": 46.0, "pe_ttm": 47.5, "pat_cagr_3y": 126.0, "de": 0.0, "cfo_pat_5y": 105.0, "tax_rate": 26.0, "sbc_dilution": 0.5, "promoter_holding": 0.0, "promoter_pledge": 0.0, "gov_clean": True},
        {"ticker": "PERSISTENT.NS", "name": "Persistent Systems", "market": "NSE", "mcap": 88000, "roce": 28.4, "roe": 27.2, "pe_ttm": 44.3, "pat_cagr_3y": 27.5, "de": 0.05, "cfo_pat_5y": 88.0, "tax_rate": 25.2, "sbc_dilution": 1.0, "promoter_holding": 31.0, "promoter_pledge": 0.0, "gov_clean": True},
        {"ticker": "MSFT", "name": "Microsoft", "market": "US", "mcap": 3100000, "roce": 35.0, "roe": 38.0, "pe_ttm": 34.0, "pat_cagr_3y": 18.5, "de": 0.25, "cfo_pat_5y": 102.0, "tax_rate": 18.5, "sbc_dilution": 1.2, "promoter_holding": 50.0, "promoter_pledge": 0.0, "gov_clean": True},
        {"ticker": "NVDA", "name": "NVIDIA", "market": "US", "mcap": 2200000, "roce": 65.0, "roe": 72.0, "pe_ttm": 48.0, "pat_cagr_3y": 85.0, "de": 0.15, "cfo_pat_5y": 95.0, "tax_rate": 16.0, "sbc_dilution": 1.5, "promoter_holding": 50.0, "promoter_pledge": 0.0, "gov_clean": True}
    ])

# Render the raw Master List silently in the background tab
with tab_master:
    st.markdown("#### Master Database (Pre-Screening)")
    df_univ = get_master_universe()
    st.dataframe(df_univ, use_container_width=True)

# Process and display AI results in the primary active tab
with tab_results:
    if prompt and gemini_key:
        with st.spinner("Compiling prompt and generating investment thesis..."):
            spec = compile_prompt(gemini_key, prompt)
            
            if spec:
                # 1. Display the Investment Conviction Narrative exactly below the prompt
                st.success(f"**Investment Thesis:** {spec.synthesis_narrative}")
                st.caption(f"**Filters Applied:** {', '.join([f'{f.field} {f.op} {f.val}' for f in spec.filters])}")
                
                # 2. Filter the Data
                if spec.market in ["NSE", "US"]:
                    df_univ = df_univ[df_univ["market"] == spec.market]
                for f in spec.filters:
                    if f.field in df_univ.columns:
                        if f.op == ">=": df_univ = df_univ[df_univ[f.field] >= f.val]
                        elif f.op == "<=": df_univ = df_univ[df_univ[f.field] <= f.val]
                        elif f.op == "==": df_univ = df_univ[df_univ[f.field] == f.val]
                        elif f.op == ">": df_univ = df_univ[df_univ[f.field] > f.val]
                        elif f.op == "<": df_univ = df_univ[df_univ[f.field] < f.val]

                # 3. Score the Surviving Stocks
                results = []
                for _, r in df_univ.iterrows():
                    lv = fetch_live_data(r["ticker"])
                    gt, flg, p_scores = score_stock(r.to_dict(), lv)
                    
                    a_scores = []
                    if act_wb: a_scores.append(p_scores["Buffett"])
                    if act_pl: a_scores.append(p_scores["Lynch"])
                    if act_cm: a_scores.append(p_scores["Mayer"])
                    if act_rj: a_scores.append(p_scores["RJ"])
                    if act_cs: a_scores.append(p_scores["CANSLIM"])
                    
                    super_ai = round(np.mean(a_scores), 1) if a_scores else 0.0
                    
                    res = {
                        "Ticker": clean_display_ticker(r["ticker"]), 
                        "Name": r["name"], 
                        "Market": r["market"], 
                        "CMP": lv["cmp"], 
                        "1Y Ret": lv["return_1y"], 
                        "Gate 1": gt, 
                        "Super AI": super_ai
                    }
                    results.append(res)

                # 4. Display Results Table
                if results:
                    st.dataframe(pd.DataFrame(results).sort_values("Super AI", ascending=False), use_container_width=True)
                else:
                    st.warning("No stocks in the master universe matched these criteria.")
    elif not prompt:
        st.info("👈 Enter your screening prompt above to generate an investment thesis and view scored results.")
