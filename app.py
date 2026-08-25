"""
Multibagger Investing & Scoring Engine — single-file Streamlit app.

WHAT THIS FILE IS: a full port of the confirmed Multibagger Investing
Framework (12-point Gate 1, market-bifurcated India/US rules, 5-persona
weighted scoring, Perfection Zone valuation cap, Super AI Score) into one
copy-pasteable app.py, matching your existing repo's minimal dependencies
(pandas, numpy, yfinance, google-genai, pydantic — no fastapi/uvicorn).

READ THIS BEFORE TRUSTING ANY RESULT:
  - FUNDAMENTALS (ROCE, promoter holding/pledge, CFO/PAT, tax rate, etc.)
    come from whatever universe source you configure below — a Google
    Sheet you maintain, or the tiny built-in demo list. This file does
    NOT invent fundamentals for real companies. If a field isn't in your
    source data, it's filled with a conservative, gate-neutral default
    and flagged in "estimated_fields" — never a value chosen to
    manufacture a false PASS or a false AUTO_REJECT.
  - TECHNICALS (current price, 50-DMA, volume, proximity to 52-wk high,
    relative strength) ARE fetched live via yfinance on every query,
    since these genuinely change daily and yfinance handles them well.
  - The built-in DEMO_UNIVERSE below is a tiny illustrative starter set,
    not a real screening universe. Point GOOGLE_SHEET_CSV_URL (in the
    sidebar or st.secrets) at your own maintained data for real use.

HOW TO POINT THIS AT A GOOGLE SHEET (no API key, no auth needed):
  1. In Google Sheets: File > Share > "Anyone with the link" > Viewer.
  2. Grab the sheet ID from the URL and the gid of the specific tab.
  3. Build this URL: https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>
  4. Paste that URL into the sidebar's "Google Sheet CSV URL" field (or
     set it once via Streamlit Cloud's Secrets as GOOGLE_SHEET_CSV_URL).
  Required columns: ticker, name, market ("NSE" or "US"), currency
  ("INR"/"USD"), market_cap_native. Every other column is optional —
  anything missing gets a safe default (see fill_defaults() below).
"""
from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

st.set_page_config(page_title="Multibagger Investing AI", layout="wide", page_icon="📈")

# ===========================================================================
# 1. CONFIG — thresholds pulled straight from the confirmed framework spec
# ===========================================================================

FX_RATE_INR_PER_USD = 87.0
INR_PER_CRORE = 10_000_000.0

TAX_RATE_FLOOR = {"NSE": 25.0, "US": 15.0}          # Gate 1 flag #6
US_DILUTION_WATCH_LOW = 1.5                           # Gate 1 flag #9 (US)
US_DILUTION_REJECT_HIGH = 3.0
US_SBC_DILUTION_REJECT_CEILING = 3.0                  # Gate 1 flag #10 (US)
US_INSIDER_TAX_DEFAULT = 20.0
NSE_TAX_DEFAULT = 30.0

MAYER_RUNWAY_CAP_NSE_CR = 15_000.0                    # ₹15,000 Cr
MAYER_RUNWAY_CAP_US_M = 15_000.0                       # $15,000 M ( = $15B )

PERFECTION_ZONE_PE_LOW = 60.0
PERFECTION_ZONE_PE_HIGH = 80.0
PERFECTION_ZONE_CAP_LOW = 28.0
PERFECTION_ZONE_CAP_HIGH = 38.0

PERSONA_HURDLES = {"buffett": 65.0, "lynch": 70.0, "jhunjhunwala": 70.0, "mayer": 70.0, "oneil": 70.0}
SCORE_RESOLUTION = 0.5

PERSONA_LABELS = {
    "buffett": "Warren Buffett", "lynch": "Peter Lynch", "jhunjhunwala": "Rakesh Jhunjhunwala",
    "mayer": "Chris Mayer", "oneil": "CANSLIM (O'Neil)",
}

# Conservative, gate-neutral defaults for any field missing from your data
# source. Booleans that could imply a governance scandal ALWAYS default
# False — this file never manufactures a red flag about a real company.
DEFAULT_VALUES = {
    "is_nbfc": False, "interest_coverage": 10.0, "contingent_liabilities_pct_net_worth": 5.0,
    "debt_to_equity": 0.4,
    "promoter_pledge_pct": 0.0, "promoter_holding_pct": 60.0, "promoter_holding_falling": False,
    "auditor_exit_flag": False, "qualified_opinion_flag": False, "adverse_rpt_flag": False,
    "cfo_pat_5yr_pct": 100.0, "receivable_days": 30.0, "receivable_days_rising": False,
    "roce_10yr_avg_pct": None,  # falls back to roce_pct if present, else 15.0
    "opm_erratic_or_declining": False, "share_dilution_yoy_pct": 0.0,
    "dilution_justified_by_acquisition": False,
    "roce_pct": 15.0, "roe_pct": 15.0, "net_cash": False, "pe_ratio": 25.0, "peg_ratio": 1.5,
    "sales_eps_cagr_pct": 10.0, "debtor_days": 30.0, "promoter_led": True,
    "market_share_rank": 5, "industry_penetration_pct": 30.0, "opm_expansion_yoy_bps": 0.0,
    "reinvestment_rate_pct": 50.0, "sbc_dilution_pct": 0.0, "gross_margin_pct": 30.0,
    "gross_margin_expanding": False, "insider_net_selling_flag": False, "csuite_exit_flag": False,
    "eps_growth_qoq_pct": 10.0, "eps_growth_annual_avg_pct": 10.0,
    "institutional_ownership_pct": 20.0, "market_direction_score": 60.0,
}
# Fields that come from live yfinance data, not the fundamentals source.
TECHNICAL_FIELDS = ["current_price", "dma_50", "pct_off_52wk_high", "volume_vs_50d_avg_pct", "relative_strength_rating"]
DEFAULT_TECHNICALS = {
    "current_price": 0.0, "dma_50": 0.0, "pct_off_52wk_high": 10.0,
    "volume_vs_50d_avg_pct": 100.0, "relative_strength_rating": 50.0,
}


# ===========================================================================
# 2. SCORING HELPERS — continuous 0-100 scaling, never round mid-pipeline
# ===========================================================================

def round_half(x: float) -> float:
    return round(x / SCORE_RESOLUTION) * SCORE_RESOLUTION


def scale_up(value: float, target: float, floor: float = 0.0) -> float:
    """Higher is better: target (or above) -> 100, floor (or below) -> 0."""
    if value <= floor:
        return 0.0
    if value >= target:
        return 100.0
    return (value - floor) / (target - floor) * 100.0


def scale_down(value: float, target: float, ceiling: float) -> float:
    """Lower is better: target (or below) -> 100, ceiling (or above) -> 0."""
    if value <= target:
        return 100.0
    if value >= ceiling:
        return 0.0
    return (ceiling - value) / (ceiling - target) * 100.0


def band_score(value: float, low: float, high: float, ideal_low: float, ideal_high: float) -> float:
    """Highest inside [ideal_low, ideal_high], tapering to 0 at [low, high]."""
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return scale_up(value, ideal_low, floor=low)
    return scale_down(value, ideal_high, ceiling=high)


def apply_perfection_zone_cap(valuation_subscore: float, pe_ratio: float) -> tuple[float, Optional[str]]:
    """Reverse-DCF cap: P/E 60-80x hard-caps a valuation-margin subscore."""
    if pe_ratio < PERFECTION_ZONE_PE_LOW:
        return valuation_subscore, None
    if pe_ratio >= PERFECTION_ZONE_PE_HIGH:
        cap = PERFECTION_ZONE_CAP_LOW
    else:
        progress = (pe_ratio - PERFECTION_ZONE_PE_LOW) / (PERFECTION_ZONE_PE_HIGH - PERFECTION_ZONE_PE_LOW)
        cap = PERFECTION_ZONE_CAP_HIGH - progress * (PERFECTION_ZONE_CAP_HIGH - PERFECTION_ZONE_CAP_LOW)
    if valuation_subscore <= cap:
        return valuation_subscore, None
    return cap, f"Perfection Zone: P/E {pe_ratio:.1f}x caps this subscore at {cap:.1f}"


def crore_to_usd_millions(cr: float) -> float:
    return (cr * INR_PER_CRORE) / FX_RATE_INR_PER_USD / 1_000_000.0


def to_usd_millions(market_cap_native: float, currency: str) -> float:
    return crore_to_usd_millions(market_cap_native) if currency == "INR" else market_cap_native


# ===========================================================================
# 3. GATE 1 — the 12 red-flag rejection filters, market-bifurcated,
#    3-tier severity (none/watch/reject). Confirmed design:
#      Flag #6  ETR floor:      NSE 25% / US 15%
#      Flag #9  Dilution:       NSE binary reject-unless-justified /
#                                US 3-tier <=1.5 none / 1.5-3.0 watch / >3.0 reject
#      Flags 10/11 Promoter:    NSE pledge(#10)+holding(#11) unchanged /
#                                US #10 = insider dump OR C-suite exit OR
#                                raw SBC dilution >3% -> hard AUTO_REJECT; #11 retired
#      Flag #12 Forensic:       universal both markets, always reject-severity
# ===========================================================================

def evaluate_gate1(row: dict) -> dict:
    market = row["market"]
    flags = []

    def add(flag_id, category, name, severity, detail):
        flags.append({"id": flag_id, "category": category, "name": name, "severity": severity, "detail": detail})

    de_ceiling = 4.0 if row.get("is_nbfc") else 1.0
    add(1, "balance_sheet", "Debt/Equity above ceiling",
        "reject" if row["debt_to_equity"] > de_ceiling else "none",
        f"D/E {row['debt_to_equity']:.2f}x vs ceiling {de_ceiling:.1f}x")
    add(2, "balance_sheet", "Interest coverage < 4.0x",
        "reject" if row["interest_coverage"] < 4.0 else "none",
        f"Interest coverage {row['interest_coverage']:.2f}x")
    add(3, "balance_sheet", "Contingent liabilities > 25% of net worth",
        "reject" if row["contingent_liabilities_pct_net_worth"] > 25.0 else "none",
        f"{row['contingent_liabilities_pct_net_worth']:.1f}% of net worth")

    add(4, "earnings_quality", "5-Yr cumulative CFO/PAT < 80%",
        "reject" if row["cfo_pat_5yr_pct"] < 80.0 else "none", f"CFO/PAT {row['cfo_pat_5yr_pct']:.1f}%")
    add(5, "earnings_quality", "Receivable days > 90 or rising",
        "reject" if (row["receivable_days"] > 90.0 or row["receivable_days_rising"]) else "none",
        f"Receivable days {row['receivable_days']:.0f}")
    tax_floor = TAX_RATE_FLOOR.get(market, 20.0)
    add(6, "earnings_quality", "Effective tax rate below market floor",
        "reject" if row["effective_tax_rate_5yr_avg_pct"] < tax_floor else "none",
        f"ETR {row['effective_tax_rate_5yr_avg_pct']:.1f}% vs {market} floor {tax_floor:.1f}%")

    add(7, "business_quality", "10-Yr avg ROCE < 15%",
        "reject" if row["roce_10yr_avg_pct"] < 15.0 else "none", f"10-yr avg ROCE {row['roce_10yr_avg_pct']:.1f}%")
    add(8, "business_quality", "OPM erratic or declining",
        "reject" if row["opm_erratic_or_declining"] else "none", "OPM flagged erratic/declining")

    dilution = row["share_dilution_yoy_pct"]
    if market == "NSE":
        if dilution <= 0.0:
            d_sev, d_detail = "none", f"YoY dilution {dilution:.1f}%"
        elif row.get("dilution_justified_by_acquisition"):
            d_sev, d_detail = "watch", f"YoY dilution {dilution:.1f}%, justified by acquisition"
        else:
            d_sev, d_detail = "reject", f"YoY dilution {dilution:.1f}%, not justified"
    else:
        if dilution <= US_DILUTION_WATCH_LOW:
            d_sev, d_detail = "none", f"Net dilution {dilution:.1f}% (<= {US_DILUTION_WATCH_LOW}%)"
        elif dilution <= US_DILUTION_REJECT_HIGH:
            d_sev, d_detail = "watch", f"Net dilution {dilution:.1f}% (watch band)"
        else:
            d_sev, d_detail = "reject", f"Net dilution {dilution:.1f}% > {US_DILUTION_REJECT_HIGH}%"
    add(9, "business_quality", "Share count dilution / net SBC dilution", d_sev, d_detail)

    if market == "NSE":
        promoter_led = row.get("promoter_led", True)
        if promoter_led:
            add(10, "governance", "Promoter pledging > 5%",
                "reject" if row["promoter_pledge_pct"] > 5.0 else "none",
                f"Promoter pledge {row['promoter_pledge_pct']:.1f}%")
            holding_bad = row["promoter_holding_pct"] < 40.0 or row["promoter_holding_falling"]
            add(11, "governance", "Promoter holding < 40% or falling",
                "reject" if holding_bad else "none", f"Promoter holding {row['promoter_holding_pct']:.1f}%")
        else:
            # Company has no traditional promoter/founder-controlling-shareholder
            # structure at all (e.g. a demutualized exchange) — the pledge/holding
            # concept doesn't apply, same treatment as the US market gets.
            add(10, "governance", "Promoter pledging (N/A — not a promoter-led company)",
                "none", "promoter_led=False on this record; pledge concept doesn't apply")
            add(11, "governance", "Promoter holding (N/A — not a promoter-led company)",
                "none", "promoter_led=False on this record; holding-floor concept doesn't apply")
    else:
        insider_issue = row.get("insider_net_selling_flag") or row.get("csuite_exit_flag")
        sbc_over = row.get("sbc_dilution_pct", 0.0) > US_SBC_DILUTION_REJECT_CEILING
        add(10, "governance", "Insider dumping / C-suite exit / SBC dilution ceiling",
            "reject" if (insider_issue or sbc_over) else "none",
            f"insider_dump={row.get('insider_net_selling_flag')} csuite_exit={row.get('csuite_exit_flag')} "
            f"sbc={row.get('sbc_dilution_pct', 0.0):.1f}%")
        add(11, "governance", "Promoter holding (N/A — US market)", "none", "Folded into flag #10 for US")

    forensic = row["auditor_exit_flag"] or row["qualified_opinion_flag"] or row["adverse_rpt_flag"]
    add(12, "governance", "Auditor exits / qualified opinions / forensic red flags",
        "reject" if forensic else "none", "Forensic issue flagged" if forensic else "No forensic issues")

    reject_flags = [f for f in flags if f["severity"] == "reject"]
    watch_flags = [f for f in flags if f["severity"] == "watch"]
    gov_reject = [f for f in reject_flags if f["category"] == "governance"]
    non_gov_reject = [f for f in reject_flags if f["category"] != "governance"]

    if gov_reject:
        verdict, reason = "AUTO_REJECT", f"{len(gov_reject)} governance red flag(s) at reject severity"
    elif len(non_gov_reject) >= 2:
        verdict, reason = "REJECT", f"{len(non_gov_reject)} non-governance red flags (threshold: 2+)"
    else:
        verdict = "PASS"
        watch_items = reject_flags + watch_flags
        reason = "Clean" if not watch_items else f"{len(watch_items)} item(s) below rejection threshold"

    return {
        "verdict": verdict, "reason": reason, "flags": flags,
        "watch_items": (reject_flags + watch_flags) if verdict == "PASS" else [],
    }


# ===========================================================================
# 4. TIER 3 — the 5-persona scoring engine (Section 3 of the spec)
# ===========================================================================

def score_buffett(row: dict) -> dict:
    ce = scale_up(max(row["roce_pct"], row["roe_pct"]), 25.0)
    solv = 100.0 if row["net_cash"] else scale_down(row["debt_to_equity"], 0.1, 1.0)
    cf = scale_up(row["cfo_pat_5yr_pct"], 95.0)
    val, note = apply_perfection_zone_cap(scale_down(row["pe_ratio"], 25.0, 80.0), row["pe_ratio"])
    subs = [("Capital Efficiency (ROCE/ROE)", 30, ce), ("Balance Sheet Solvency", 25, solv),
            ("Cash Flow Quality", 25, cf), ("Valuation Margin of Safety", 20, val)]
    total = round_half(sum(s * w for _, w, s in subs) / 100.0)
    return _persona_result("buffett", total, subs, [note] if note else [])


def score_lynch(row: dict) -> dict:
    peg, note = apply_perfection_zone_cap(scale_down(row["peg_ratio"], 1.0, 3.0), row["pe_ratio"])
    growth = scale_up(row["sales_eps_cagr_pct"], 25.0)
    wc = scale_down(row["debtor_days"], 45.0, 120.0)
    safety = 100.0 if row["net_cash"] else scale_down(row["debt_to_equity"], 0.1, 1.0)
    subs = [("PEG Ratio", 35, peg), ("Growth Velocity", 30, growth),
            ("Working Capital Speed", 20, wc), ("Balance Sheet Safety", 15, safety)]
    total = round_half(sum(s * w for _, w, s in subs) / 100.0)
    return _persona_result("lynch", total, subs, [note] if note else [])


def score_jhunjhunwala(row: dict) -> dict:
    rank = row["market_share_rank"]
    dominance = 100.0 if rank <= 2 else scale_down(float(rank), 2.0, 8.0)
    runway = scale_down(row["industry_penetration_pct"], 15.0, 70.0)
    opm_infl = scale_up(row["opm_expansion_yoy_bps"], 150.0, floor=-100.0)

    if row["market"] == "US":
        insider_issue = row.get("insider_net_selling_flag") or row.get("csuite_exit_flag")
        dilution = row.get("share_dilution_yoy_pct", 0.0)
        if insider_issue or dilution > US_DILUTION_REJECT_HIGH:
            conviction = 45.0  # extrapolated: Gate 1 would already reject here; treated as equally severe
        elif US_DILUTION_WATCH_LOW <= dilution <= US_DILUTION_REJECT_HIGH:
            conviction = 70.0
        else:
            conviction = 85.0
    else:
        pledge_pen = scale_down(row["promoter_pledge_pct"], 0.0, 10.0)
        holding_sc = scale_up(row["promoter_holding_pct"], 60.0, floor=30.0)
        conviction = (pledge_pen + holding_sc) / 2.0
        if row["promoter_holding_falling"]:
            conviction *= 0.5

    subs = [("Sector Dominance & Brand Power", 30, dominance), ("Pond Dynamics & Runway", 25, runway),
            ("Operating Leverage Inflection", 25, opm_infl), ("Promoter Conviction & Governance", 20, conviction)]
    total = round_half(sum(s * w for _, w, s in subs) / 100.0)
    return _persona_result("jhunjhunwala", total, subs, [])


def score_mayer(row: dict) -> dict:
    reinvest_sc = scale_up(row["reinvestment_rate_pct"], 75.0)
    roce_sc = scale_up(row["roce_pct"], 20.0)
    compounding = min(reinvest_sc, roce_sc)  # needs BOTH high, not averaged

    runway_cap = MAYER_RUNWAY_CAP_NSE_CR if row["market"] == "NSE" else MAYER_RUNWAY_CAP_US_M
    runway = scale_down(row["market_cap_native"], runway_cap, runway_cap * 10)

    notes = []
    if row["market"] == "US":
        alignment = scale_down(row["share_dilution_yoy_pct"], US_DILUTION_WATCH_LOW, US_DILUTION_REJECT_HIGH)
        if row["share_dilution_yoy_pct"] > US_DILUTION_REJECT_HIGH:
            notes.append("SBC dilution veto tripped")
    else:
        alignment = scale_down(row["share_dilution_yoy_pct"], 0.0, 5.0)

    gm = scale_up(row["gross_margin_pct"], 60.0)
    if row["gross_margin_expanding"]:
        gm = min(100.0, gm + 10.0)

    subs = [("Compounding Engine", 35, compounding), ("Market-Cap Runway Headroom", 30, runway),
            ("Owner-Operator Alignment / Dilution Veto", 20, alignment), ("Gross Margin & Pricing Power", 15, gm)]
    total = round_half(sum(s * w for _, w, s in subs) / 100.0)
    return _persona_result("mayer", total, subs, notes)


def score_oneil(row: dict) -> dict:
    c = scale_up(row["eps_growth_qoq_pct"], 25.0, floor=-20.0)
    a = scale_up(row["eps_growth_annual_avg_pct"], 25.0, floor=-10.0)
    n = scale_down(row["pct_off_52wk_high"], 0.0, 30.0)
    s = scale_up(row["volume_vs_50d_avg_pct"], 150.0, floor=50.0)
    l = scale_up(row["relative_strength_rating"], 90.0, floor=20.0)
    i = band_score(row["institutional_ownership_pct"], 0.0, 95.0, 20.0, 70.0)
    m = max(0.0, min(100.0, row["market_direction_score"]))
    each = 100.0 / 7.0
    subs = [("C - Current Quarterly EPS", each, c), ("A - Annual Earnings Growth", each, a),
            ("N - New Highs", each, n), ("S - Supply & Demand (Volume)", each, s),
            ("L - Leader / Relative Strength", each, l), ("I - Institutional Sponsorship", each, i),
            ("M - Market Direction", each, m)]
    total = round_half(sum(v for _, _, v in subs) / 7.0)
    notes = []
    if row["relative_strength_rating"] < 80:
        notes.append("RS below the canonical CANSLIM 80 bar — scored proportionally, not zeroed")
    return _persona_result("oneil", total, subs, notes)


def _persona_result(key: str, total: float, subs: list, notes: list) -> dict:
    hurdle = PERSONA_HURDLES[key]
    return {"persona": key, "total": total, "hurdle": hurdle, "passed": total > hurdle,
            "subscores": [{"label": lbl, "weight": w, "score": round(sc, 1)} for lbl, w, sc in subs], "notes": notes}


PERSONA_SCORERS = {"buffett": score_buffett, "lynch": score_lynch, "jhunjhunwala": score_jhunjhunwala,
                   "mayer": score_mayer, "oneil": score_oneil}


def score_all_personas(row: dict, active_keys: Optional[list[str]] = None) -> dict:
    keys = active_keys or list(PERSONA_SCORERS)
    return {k: PERSONA_SCORERS[k](row) for k in keys if k in PERSONA_SCORERS}


def compute_super_score(persona_results: dict) -> float:
    if not persona_results:
        return 0.0
    return round_half(sum(r["total"] for r in persona_results.values()) / len(persona_results))


# ===========================================================================
# 5. ORCHESTRATION — Gate 1 -> Tier 3 -> Super Score for one stock row
# ===========================================================================

def fill_defaults(row: dict) -> tuple[dict, list[str]]:
    """Fill any missing field with a conservative default; track what was estimated."""
    filled, estimated = dict(row), []
    for field, default in DEFAULT_VALUES.items():
        if field not in filled or pd.isna(filled.get(field)):
            if field == "roce_10yr_avg_pct":
                filled[field] = filled.get("roce_pct", 15.0)
            elif field == "effective_tax_rate_5yr_avg_pct":
                continue  # handled below, market-dependent
            else:
                filled[field] = default
            estimated.append(field)
    if "effective_tax_rate_5yr_avg_pct" not in filled or pd.isna(filled.get("effective_tax_rate_5yr_avg_pct")):
        filled["effective_tax_rate_5yr_avg_pct"] = NSE_TAX_DEFAULT if filled["market"] == "NSE" else US_INSIDER_TAX_DEFAULT
        estimated.append("effective_tax_rate_5yr_avg_pct")
    for field, default in DEFAULT_TECHNICALS.items():
        if field not in filled or pd.isna(filled.get(field)):
            filled[field] = default
            estimated.append(field)
    return filled, estimated


def score_stock(row: dict, active_personas: Optional[list[str]] = None, apply_gate: bool = True) -> dict:
    filled, estimated = fill_defaults(row)
    gate = evaluate_gate1(filled)

    result = {
        "ticker": filled["ticker"], "name": filled.get("name", filled["ticker"]), "market": filled["market"],
        "gate": gate, "personas": {}, "super_score": None, "estimated_fields": estimated,
    }
    if apply_gate and gate["verdict"] in ("REJECT", "AUTO_REJECT"):
        return result

    personas = score_all_personas(filled, active_personas)
    result["personas"] = personas
    result["super_score"] = compute_super_score(personas)
    return result


# ===========================================================================
# 6. DATA LAYER — universe (Google Sheet or demo fallback) + live technicals
# ===========================================================================

# A tiny illustrative starter set — NOT a real screening universe. Replace
# via the Google Sheet URL in the sidebar for anything real. (Kept small
# and unchanged from the original prototype rather than expanded further —
# see the accompanying chat message for why this project doesn't hand-add
# more real-company rows here.)
DEMO_UNIVERSE = pd.DataFrame([
    {"ticker": "BSE.NS", "name": "BSE Limited", "market": "NSE", "currency": "INR", "market_cap_native": 134000,
     "roce_pct": 59.9, "roe_pct": 46.0, "pe_ratio": 47.5, "sales_eps_cagr_pct": 126.0, "debt_to_equity": 0.0,
     "cfo_pat_5yr_pct": 105.0, "effective_tax_rate_5yr_avg_pct": 26.0, "sbc_dilution_pct": 0.5,
     "promoter_holding_pct": 0.0, "promoter_pledge_pct": 0.0, "promoter_led": False},
    {"ticker": "PERSISTENT.NS", "name": "Persistent Systems", "market": "NSE", "currency": "INR",
     "market_cap_native": 88000, "roce_pct": 28.4, "roe_pct": 27.2, "pe_ratio": 44.3, "sales_eps_cagr_pct": 27.5,
     "debt_to_equity": 0.05, "cfo_pat_5yr_pct": 88.0, "effective_tax_rate_5yr_avg_pct": 25.2,
     "sbc_dilution_pct": 1.0, "promoter_holding_pct": 31.0, "promoter_pledge_pct": 0.0, "promoter_led": True},
    {"ticker": "MSFT", "name": "Microsoft", "market": "US", "currency": "USD", "market_cap_native": 3_100_000,
     "roce_pct": 35.0, "roe_pct": 38.0, "pe_ratio": 34.0, "sales_eps_cagr_pct": 18.5, "debt_to_equity": 0.25,
     "cfo_pat_5yr_pct": 102.0, "effective_tax_rate_5yr_avg_pct": 18.5, "sbc_dilution_pct": 1.2},
    {"ticker": "NVDA", "name": "NVIDIA", "market": "US", "currency": "USD", "market_cap_native": 2_200_000,
     "roce_pct": 65.0, "roe_pct": 72.0, "pe_ratio": 48.0, "sales_eps_cagr_pct": 85.0, "debt_to_equity": 0.15,
     "cfo_pat_5yr_pct": 95.0, "effective_tax_rate_5yr_avg_pct": 16.0, "sbc_dilution_pct": 1.5},
])


@st.cache_data(ttl=3600)
def load_universe(sheet_csv_url: Optional[str]) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label). Falls back to the demo set on any failure."""
    if sheet_csv_url:
        try:
            df = pd.read_csv(sheet_csv_url)
            required = {"ticker", "name", "market", "currency", "market_cap_native"}
            missing = required - set(df.columns)
            if missing:
                return DEMO_UNIVERSE.copy(), f"⚠️ Sheet missing columns {sorted(missing)} — using demo fallback"
            if df.empty:
                return DEMO_UNIVERSE.copy(), "⚠️ Sheet loaded but empty — using demo fallback"
            return df, f"✅ Google Sheet ({len(df)} rows)"
        except Exception as exc:
            return DEMO_UNIVERSE.copy(), f"⚠️ Could not load sheet ({exc}) — using demo fallback"
    return DEMO_UNIVERSE.copy(), f"ℹ️ Demo universe ({len(DEMO_UNIVERSE)} rows) — no Google Sheet configured"


def format_ticker(symbol: str, market: str) -> str:
    clean = symbol.strip().upper()
    if market == "NSE":
        return clean if clean.endswith((".NS", ".BO")) else f"{clean}.NS"
    return clean.replace(".NS", "").replace(".BO", "")


def clean_display_ticker(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".BO", "")


@st.cache_data(ttl=1800)
def fetch_live_technicals(ticker: str) -> dict:
    """Live price/DMA/volume/52wk-high data via yfinance. RS rating is computed
    separately as a peer-relative percentile (see rank_relative_strength)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 50:
            return dict(DEFAULT_TECHNICALS)
        close = hist["Close"]
        current = float(close.iloc[-1])
        dma_50 = float(close.rolling(50).mean().iloc[-1])
        high_52wk = float(close.max())
        pct_off_high = max(0.0, (high_52wk - current) / high_52wk * 100.0) if high_52wk else 10.0
        avg_vol_50 = float(hist["Volume"].rolling(50).mean().iloc[-1])
        latest_vol = float(hist["Volume"].iloc[-1])
        vol_vs_avg = (latest_vol / avg_vol_50 * 100.0) if avg_vol_50 else 100.0
        return_1y = ((current - close.iloc[0]) / close.iloc[0]) * 100.0
        return {"current_price": round(current, 2), "dma_50": round(dma_50, 2),
                "pct_off_52wk_high": round(pct_off_high, 1), "volume_vs_50d_avg_pct": round(vol_vs_avg, 1),
                "_return_1y": round(return_1y, 1)}
    except Exception:
        return dict(DEFAULT_TECHNICALS)


def rank_relative_strength(technicals_by_ticker: dict) -> dict:
    """Simplified RS proxy: percentile rank of 1-yr return WITHIN the current
    screened set (not true IBD methodology, which ranks against the whole
    market — that needs a much bigger universe than most Sheet setups will have)."""
    returns = {t: v.get("_return_1y", 0.0) for t, v in technicals_by_ticker.items()}
    if len(returns) <= 1:
        return {t: 50.0 for t in returns}
    sorted_tickers = sorted(returns, key=lambda t: returns[t])
    n = len(sorted_tickers)
    return {t: round((i / max(n - 1, 1)) * 100.0, 1) for i, t in enumerate(sorted_tickers)}


# ===========================================================================
# 7. TIER 1 — Gemini query compiler
# ===========================================================================

SCREENABLE_FIELDS = {
    "roce_pct": "Current ROCE %", "roce_10yr_avg_pct": "10-yr avg ROCE %", "roe_pct": "Current ROE %",
    "pe_ratio": "P/E ratio", "peg_ratio": "PEG ratio", "debt_to_equity": "Debt/Equity ratio",
    "cfo_pat_5yr_pct": "5-yr cumulative CFO/PAT %", "sales_eps_cagr_pct": "Sales & EPS CAGR %",
    "promoter_holding_pct": "Promoter holding % (NSE only)", "promoter_pledge_pct": "Promoter pledge % (NSE only)",
    "gross_margin_pct": "Gross margin %", "institutional_ownership_pct": "Institutional ownership %",
    "market_cap_native": "Market cap, native units (₹ crore NSE / $ million US)",
    "current_price": "Current price", "dma_50": "50-day moving average",
    "relative_strength_rating": "Relative strength rating, 0-100",
}


class FilterCriterion(BaseModel):
    field: str = Field(description=f"One of: {', '.join(SCREENABLE_FIELDS)}")
    op: Literal["gt", "gte", "lt", "lte", "eq"]
    val: float


class QuerySpecification(BaseModel):
    market: Literal["NSE", "US", "ALL"] = "ALL"
    filters: list[FilterCriterion] = Field(default_factory=list)
    active_personas: Optional[list[Literal["buffett", "lynch", "jhunjhunwala", "mayer", "oneil"]]] = None
    top_n: int = 10
    synthesis_narrative: str = Field(default="", description="2-3 sentence investment thesis for this screen")


def _system_instructions() -> str:
    field_lines = "\n".join(f"  - {k}: {v}" for k, v in SCREENABLE_FIELDS.items())
    return f"""Translate a stock-screening prompt into structured JSON.
market: "NSE" (India), "US", or "ALL" (unscoped).
filters: one entry per metric threshold, using ONLY these fields:
{field_lines}
Do not invent a field not in this list — omit anything that doesn't map cleanly.
active_personas: which of buffett/lynch/jhunjhunwala/mayer/oneil the prompt names, else null for all five.
top_n: requested result count, else 10.
synthesis_narrative: a short, honest 2-3 sentence investment thesis for why this screen finds quality names.
Return only the JSON."""


def compile_prompt(api_key: str, prompt: str) -> Optional[QuerySpecification]:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            contents=f"{_system_instructions()}\n\nPrompt: {prompt}",
            config={"response_mime_type": "application/json", "response_schema": QuerySpecification, "temperature": 0.2},
        )
        return QuerySpecification.model_validate_json(response.text)
    except Exception as exc:
        st.error(f"Gemini query compiler failed ({exc}) — falling back to an unfiltered top-{10} scan.")
        return QuerySpecification(market="ALL", filters=[], top_n=10,
                                   synthesis_narrative="(Gemini unavailable — showing an unfiltered scan instead.)")


# ===========================================================================
# 8. UI
# ===========================================================================

def apply_filters(df: pd.DataFrame, spec: QuerySpecification) -> pd.DataFrame:
    out = df
    if spec.market != "ALL":
        out = out[out["market"] == spec.market]
    ops = {"gt": lambda s, v: s > v, "gte": lambda s, v: s >= v, "lt": lambda s, v: s < v,
           "lte": lambda s, v: s <= v, "eq": lambda s, v: s == v}
    for f in spec.filters:
        if f.field in out.columns:
            out = out[ops[f.op](out[f.field], f.val)]
    return out


def main() -> None:
    st.title("🏛️ Multibagger Investing & Scoring Engine")

    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        st.warning("⚠️ No GEMINI_API_KEY found (Secrets or env var) — the NL screener will fall back to an unfiltered scan.")

    st.sidebar.header("🌐 Data source")
    default_sheet_url = st.secrets.get("GOOGLE_SHEET_CSV_URL", "") if hasattr(st, "secrets") else ""
    sheet_url = st.sidebar.text_input(
        "Google Sheet CSV URL (optional)", value=default_sheet_url,
        help="File > Share > Anyone with link > Viewer, then use the .../export?format=csv&gid=... URL. "
             "Leave blank to use the small built-in demo universe.",
    )
    universe_df, source_label = load_universe(sheet_url or None)
    st.sidebar.caption(source_label)

    st.sidebar.header("🎯 Active personas")
    persona_toggles = {k: st.sidebar.checkbox(v, value=True) for k, v in PERSONA_LABELS.items()}
    override_personas = st.sidebar.checkbox("Override query with these toggles", value=False)
    apply_gate = st.sidebar.checkbox("Apply Red Flag Gate", value=True)

    st.markdown("### 🔍 Live Single Stock Auditor")
    market_for_audit = st.radio("Market for ticker lookup", ["NSE", "US"], horizontal=True)
    user_input = st.text_input("Enter ticker to audit instantly (e.g., RELIANCE, MSFT):")
    if user_input:
        query_ticker = format_ticker(user_input, market_for_audit)
        tech = fetch_live_technicals(query_ticker)
        if tech["current_price"] > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Live Price", tech["current_price"])
            c2.metric("50-DMA", tech["dma_50"])
            c3.metric("Above 50-DMA", "Yes" if tech["current_price"] > tech["dma_50"] else "No")
            c4.metric("Off 52-wk high", f"{tech['pct_off_52wk_high']:.1f}%")
        else:
            st.warning("Could not fetch live data for this ticker.")

    st.markdown("---")
    st.markdown("### 💬 Natural Language AI Screener")
    prompt = st.text_input("Ask any query (e.g., 'NSE stocks with ROCE above 25% and low debt'):")

    tab_results, tab_master = st.tabs(["✨ AI Screening Results", "📚 Universe Data"])

    with tab_master:
        st.caption(source_label)
        st.dataframe(universe_df, use_container_width=True)

    with tab_results:
        if not prompt:
            st.info("👈 Enter a screening prompt above.")
            return

        with st.spinner("Compiling query, fetching live technicals, and scoring..."):
            spec = compile_prompt(gemini_key, prompt) if gemini_key else QuerySpecification(
                market="ALL", top_n=10, synthesis_narrative="(No Gemini key configured — showing an unfiltered scan.)")
            if spec is None:
                return

            st.success(f"**Investment Thesis:** {spec.synthesis_narrative}")
            if spec.filters:
                st.caption("**Filters applied:** " + ", ".join(f"{f.field} {f.op} {f.val}" for f in spec.filters))

            matched = apply_filters(universe_df, spec)
            if matched.empty:
                st.warning("No stocks in the current universe matched these criteria. "
                           "If you expected more, check whether your Google Sheet actually covers this segment — "
                           "this screener only ever searches what's loaded in the Universe Data tab.")
                return

            active_personas = None
            if override_personas:
                active_personas = [k for k, v in persona_toggles.items() if v] or None
            else:
                active_personas = spec.active_personas

            tech_by_ticker = {}
            scored = []
            for _, r in matched.iterrows():
                row = r.to_dict()
                q_ticker = format_ticker(row["ticker"], row["market"])
                tech = fetch_live_technicals(q_ticker)
                tech_by_ticker[row["ticker"]] = tech
                row.update({k: v for k, v in tech.items() if not k.startswith("_")})
                result = score_stock(row, active_personas=active_personas, apply_gate=apply_gate)
                result["_row"] = row
                scored.append(result)

            rs_by_ticker = rank_relative_strength(tech_by_ticker)
            for result in scored:
                rs = rs_by_ticker.get(result["ticker"], 50.0)
                result["_row"]["relative_strength_rating"] = rs
                # Re-score if O'Neil is active, since RS wasn't known until now.
                active = active_personas or list(PERSONA_SCORERS)
                if "oneil" in active:
                    fresh = score_stock(result["_row"], active_personas=active_personas, apply_gate=apply_gate)
                    result.update(fresh)

            scored.sort(key=lambda r: (r["super_score"] if r["super_score"] is not None else -1), reverse=True)
            scored = scored[: spec.top_n]

            table_rows = []
            for r in scored:
                row_out = {
                    "Ticker": clean_display_ticker(r["ticker"]), "Name": r["name"], "Market": r["market"],
                    "Price": r["_row"].get("current_price", 0.0), "Gate": r["gate"]["verdict"],
                    "Super AI Score": r["super_score"],
                }
                for pk, pr in r["personas"].items():
                    row_out[PERSONA_LABELS[pk]] = pr["total"]
                if r["estimated_fields"]:
                    row_out["⚠️ Estimated fields"] = len(r["estimated_fields"])
                table_rows.append(row_out)

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            st.markdown("#### Drill down")
            options = [clean_display_ticker(r["ticker"]) for r in scored]
            if options:
                chosen = st.selectbox("Select a stock for the full breakdown", options)
                chosen_result = next(r for r in scored if clean_display_ticker(r["ticker"]) == chosen)
                gate_tab, persona_tab = st.tabs(["Red Flag Gate (12-point)", "Persona scoring"])
                with gate_tab:
                    st.write(f"**Verdict: {chosen_result['gate']['verdict']}** — {chosen_result['gate']['reason']}")
                    if chosen_result["estimated_fields"]:
                        st.warning(f"{len(chosen_result['estimated_fields'])} field(s) were estimated/defaulted "
                                   f"(not in your data source) — treat this as illustrative only: "
                                   f"{', '.join(chosen_result['estimated_fields'])}")
                    icon = {"reject": "❌", "watch": "⚠️", "none": "✅"}
                    for f in chosen_result["gate"]["flags"]:
                        st.markdown(f"{icon[f['severity']]} **#{f['id']} — {f['name']}**: {f['detail']}")
                with persona_tab:
                    if not chosen_result["personas"]:
                        st.info("No persona scores — this stock didn't clear the Red Flag Gate.")
                    for pk, pr in chosen_result["personas"].items():
                        label = PERSONA_LABELS[pk]
                        verdict = "✅ PASS" if pr["passed"] else "❌ below hurdle"
                        with st.expander(f"{label}: {pr['total']} / 100 (hurdle {pr['hurdle']}) — {verdict}"):
                            for sub in pr["subscores"]:
                                st.markdown(f"- **{sub['label']}** ({sub['weight']:.0f}%): {sub['score']} / 100")
                            for note in pr["notes"]:
                                st.caption(f"ℹ️ {note}")
                    if chosen_result["super_score"] is not None:
                        st.metric("Super AI Score", chosen_result["super_score"])


if __name__ == "__main__":
    main()
