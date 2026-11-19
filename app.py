"""PhishGuard — Simple Automata-Based Phishing URL Detection Website.

College Project for Theory of Computation / Automata Theory Course.
Demonstrates practical application of Deterministic Finite Automata (DFA),
Regular Languages, and State Transitions in Phishing URL Pattern Recognition.
"""

from __future__ import annotations

import traceback
from typing import Any, List, Optional
import pandas as pd
import streamlit as st

from automata import (
    DFA,
    DFAResult,
    DFAVisualizer,
    PhishGuardAssessment,
    PhishGuardAutomataSuite,
    RiskScorer,
    RuleEngine,
    RuleResult,
    ThreatLevel,
    URLParser,
)


# =============================================================================
# CSS STYLING & THEMING
# =============================================================================

def get_custom_css() -> str:
    """Returns clean, minimal CSS focused on readability, clear status colors, and automata clarity."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-primary: #0d1117;
            --bg-card: #161b22;
            --bg-card-hover: #1c2128;
            --border: #30363d;
            --text-primary: #f0f6fc;
            --text-muted: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-amber: #d29922;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
        }

        .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Clean header */
        .project-header {
            text-align: center;
            padding: 24px 0 16px 0;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }
        .project-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            color: #58a6ff;
            letter-spacing: -0.5px;
            margin: 0;
        }
        .project-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .toc-badge {
            display: inline-block;
            background: rgba(88, 166, 255, 0.12);
            color: #58a6ff;
            border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            margin-top: 10px;
        }

        /* Cards */
        .toc-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .toc-section-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-blue);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 14px;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .status-safe {
            background: rgba(63, 185, 80, 0.15);
            color: #3fb950;
            border: 1px solid #3fb950;
        }
        .status-suspicious {
            background: rgba(210, 153, 34, 0.15);
            color: #d29922;
            border: 1px solid #d29922;
        }
        .status-malicious {
            background: rgba(248, 81, 73, 0.15);
            color: #f85149;
            border: 1px solid #f85149;
        }

        /* Compact Score Indicator */
        .score-box {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1;
        }
        .score-bar-bg {
            background-color: #21262d;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
            margin-top: 8px;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.4s ease;
        }

        /* Reason Bullets */
        .reason-bullet {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 12px;
            background: #11161d;
            border-left: 3px solid #f85149;
            border-radius: 0 4px 4px 0;
            margin-bottom: 6px;
            font-size: 0.88rem;
            color: var(--text-primary);
        }
        .reason-bullet-safe {
            border-left-color: #3fb950;
        }

        /* Educational State Transition Sequence Ribbon */
        .state-ribbon {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            overflow-x: auto;
        }
        .state-node {
            background: #21262d;
            color: #c9d1d9;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #30363d;
        }
        .state-node-start {
            background: rgba(88, 166, 255, 0.15);
            color: #58a6ff;
            border: 1px solid #58a6ff;
            font-weight: 600;
        }
        .state-node-accept {
            background: rgba(63, 185, 80, 0.2);
            color: #3fb950;
            border: 1px solid #3fb950;
            font-weight: 600;
        }
        .state-arrow {
            color: #8b949e;
            font-size: 0.75rem;
        }

        /* Formal flow container */
        .flow-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            background: #11161d;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            gap: 8px;
        }
        .flow-step {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .flow-arrow {
            color: var(--text-muted);
            font-weight: bold;
        }

        /* Concept chips for Viva Reference */
        .concept-chip {
            display: inline-block;
            background: #21262d;
            color: #79c0ff;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            margin-right: 6px;
            margin-bottom: 4px;
            border: 1px solid #30363d;
        }

        /* Streamlit widget tweaks */
        div.stButton > button:first-child {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        div.stButton > button:first-child:hover {
            background-color: #2ea043;
            border: none;
        }
    </style>
    """


# =============================================================================
# REUSABLE UI RENDERING COMPONENTS
# =============================================================================

def render_header() -> None:
    """Renders the top title and project objective banner."""
    st.markdown(
        """
        <div class="project-header">
            <h1 class="project-title">🛡️ PHISHGUARD</h1>
            <div class="project-subtitle">Automata-Based Phishing URL Detection System</div>
            <div class="toc-badge">Theory of Computation / Automata Theory Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_result(assessment: PhishGuardAssessment, rule_report: Any) -> None:
    """Renders Section 2: Clean Detection Result with score and concise reasons."""
    if assessment.threat_level == ThreatLevel.MALICIOUS:
        status_label = "HIGH RISK"
        badge_class = "status-malicious"
        bar_color = "#f85149"
        icon = "🛑"
    elif assessment.threat_level == ThreatLevel.SUSPICIOUS:
        status_label = "SUSPICIOUS"
        badge_class = "status-suspicious"
        bar_color = "#d29922"
        icon = "⚠️"
    else:
        status_label = "SAFE"
        badge_class = "status-safe"
        bar_color = "#3fb950"
        icon = "✅"

    score_val = round(assessment.risk_score, 1)

    st.markdown(
        f"""
        <div class="toc-card" style="border-left: 4px solid {bar_color};">
            <div class="toc-section-title">DETECTION RESULT</div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;">
                <div>
                    <span class="status-badge {badge_class}">
                        {icon} Result: {status_label}
                    </span>
                </div>
                <div style="text-align: right;">
                    <div class="score-box" style="color: {bar_color};">
                        {score_val} <span style="font-size: 1rem; color: #8b949e; font-weight: normal;">/ 100</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #8b949e; font-family: 'JetBrains Mono', monospace;">Risk Score</div>
                </div>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: {score_val}%; background-color: {bar_color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="toc-section-title">WHY WAS THIS URL FLAGGED?</div>', unsafe_allow_html=True)
    triggered: List[RuleResult] = getattr(rule_report, "triggered_rules", [])

    if not triggered:
        st.markdown(
            """
            <div class="reason-bullet reason-bullet-safe">
                <span>✅</span>
                <div>
                    <strong>Clean Structural Baseline:</strong> No suspicious automata patterns, brand impersonation, or obfuscated delimiters were detected in this URL.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        top_reasons = triggered[:4]
        for r in top_reasons:
            st.markdown(
                f"""
                <div class="reason-bullet">
                    <span>⚠️</span>
                    <div>
                        <strong>{r.rule_name}:</strong> {r.evidence or r.explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_state_sequence_ribbon(trace_result: DFAResult) -> None:
    """Renders the step-by-step state transition chain as a clean visual ribbon."""
    if not trace_result.path:
        st.info("No transitions taken.")
        return

    chips = []
    steps = trace_result.path[:12]
    for i, step in enumerate(steps):
        is_start = (i == 0)
        bubble_cls = "state-node-start" if is_start else ("state-node-accept" if step.is_accept else "state-node")
        sym = repr(step.symbol) if step.symbol != " " else "' '"
        chips.append(f'<span class="{bubble_cls}">{step.from_state}</span>')
        chips.append(f'<span class="state-arrow">--{sym}--></span>')

    last_step = trace_result.path[-1]
    final_cls = "state-node-accept" if trace_result.accepted else "state-node"
    chips.append(f'<span class="{final_cls}">{last_step.to_state}</span>')

    if len(trace_result.path) > 12:
        chips.append(f'<span style="color: #8b949e; font-size: 0.75rem;">... (+{len(trace_result.path)-12} more steps)</span>')

    st.markdown(
        f"""
        <div class="state-ribbon">
            {' '.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_automata_working(
    active_dfa: DFA,
    trace_result: DFAResult,
    checked_pattern: str,
    explanation: str,
) -> None:
    """Renders Section 3: Educational Automata Working demonstration."""
    verdict_text = "ACCEPTED (w ∈ L(M)) — Pattern Recognized" if trace_result.accepted else "REJECTED (w ∉ L(M)) — Pattern Not Recognized"
    verdict_color = "#f85149" if trace_result.accepted else "#3fb950"
    verdict_icon = "🎯" if trace_result.accepted else "✅"

    st.markdown(
        f"""
        <div class="flow-container">
            <div class="flow-step">
                <span style="color: #8b949e;">Pattern:</span>
                <span style="color: #79c0ff;">"{checked_pattern}"</span>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <span style="color: #8b949e;">Automaton:</span>
                <span style="color: #bc8cff;">{active_dfa.name}</span>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <span style="color: #8b949e;">Language:</span>
                <span style="color: {verdict_color}; font-weight: 600;">{verdict_icon} {verdict_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="font-size: 0.9rem; color: #c9d1d9; margin-bottom: 14px; line-height: 1.5;">
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="font-family: JetBrains Mono; font-size: 0.8rem; color: #8b949e; margin-bottom: 6px;">STATE TRANSITION SEQUENCE:</div>', unsafe_allow_html=True)
    render_state_sequence_ribbon(trace_result)

    st.markdown('<div style="font-family: JetBrains Mono; font-size: 0.8rem; color: #8b949e; margin-top: 14px; margin-bottom: 6px;">STATE TRANSITION DIAGRAM (Active Path Highlighted):</div>', unsafe_allow_html=True)
    mermaid_code = active_dfa.to_mermaid(highlight_path=trace_result.path)
    st.markdown(f"```mermaid\n{mermaid_code}\n```")


def render_how_it_works_section() -> None:
    """Renders Section 4: Educational overview for college presentation and viva."""
    st.markdown(
        """
        <div class="toc-card">
            <div class="toc-section-title">HOW PHISHGUARD WORKS (AUTOMATA PIPELINE)</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 16px;">
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="color: #58a6ff; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">1. URL Input String</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 4px;">User submits URL string <em>w</em> over alphabet &Sigma; (ASCII characters).</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="color: #79c0ff; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">2. String Tokenization</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 4px;">Decomposes <em>w</em> into protocol, authority/host, and path substrings without execution.</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="color: #bc8cff; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">3. DFA Pattern Recognition</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 4px;">Specialized DFAs process substrings symbol-by-symbol via transition function &delta;(q, &sigma;).</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="color: #d29922; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">4. Risk Evaluation</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 4px;">Accepted adversarial patterns trigger calibrated risk rules with transparent point attribution.</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="color: #3fb950; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">5. Final Classification</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 4px;">Deterministic mapping categorizes URL as Safe, Suspicious, or High Risk.</div>
                </div>
            </div>

            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8b949e; border-top: 1px solid #30363d; padding-top: 10px;">
                <span class="concept-chip">Alphabet &Sigma;</span>
                <span class="concept-chip">DFA: M = (Q, &Sigma;, &delta;, q₀, F)</span>
                <span class="concept-chip">Transition Table &delta;(q, &sigma;)</span>
                <span class="concept-chip">Regular Languages L(M)</span>
                <span class="concept-chip">O(n) Time Complexity (ReDoS-Free)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def get_engine():
    """Initializes and caches the DFA suite and rule engine."""
    suite = PhishGuardAutomataSuite()
    engine = RuleEngine()
    return suite, engine


def get_token_for_dfa(dfa_name: str, parsed: Any) -> str:
    """Extracts the exact component string w ∈ Σ* tested by a specific automaton."""
    name_lower = dfa_name.lower()
    if "ip" in name_lower:
        return parsed.host or ""
    elif "scheme" in name_lower and "spoof" not in name_lower:
        return (parsed.scheme + ":") if parsed.scheme else ""
    elif "tld" in name_lower:
        return ("." + parsed.tld.lstrip(".")) if parsed.tld else ""
    elif "credential" in name_lower:
        return parsed.netloc or ""
    elif "percent" in name_lower:
        return parsed.raw_url or ""
    elif "subdomain" in name_lower:
        return parsed.host or ""
    elif "brand" in name_lower:
        return parsed.host or ""
    elif "action" in name_lower or "intent" in name_lower:
        return parsed.path or parsed.host or ""
    return parsed.host or ""


# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================

def main() -> None:
    """Main application loop executing the clean 4-section layout."""
    try:
        st.set_page_config(
            page_title="PhishGuard — Automata-Based Phishing Detection",
            page_icon="🛡️",
            layout="centered",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        pass

    st.markdown(get_custom_css(), unsafe_allow_html=True)

    suite, rule_engine = get_engine()
    DEFAULT_URL = "http://paypal.com.account-verification.login-portal.top/cgi-bin/auth"

    if "url_input_val" not in st.session_state:
        st.session_state["url_input_val"] = DEFAULT_URL
    if "analyzed_url" not in st.session_state:
        st.session_state["analyzed_url"] = DEFAULT_URL

    render_header()

    # -------------------------------------------------------------------------
    # SECTION 1: URL SCANNER
    # -------------------------------------------------------------------------
    st.markdown('<div class="toc-section-title">SECTION 1: URL SCANNER</div>', unsafe_allow_html=True)

    flat_presets = [
        {"name": "— Or choose an educational demo URL —", "url": ""},
        {"name": "⚠️ PayPal Brand Stacking (.top TLD)", "url": "http://paypal.com.account-verification.login-portal.top/cgi-bin/auth"},
        {"name": "⚠️ Raw IPv4 Address with Custom Port (:8080)", "url": "http://185.220.101.5:8080/banking/chase/login.php"},
        {"name": "⚠️ HTTP Basic Auth '@' Symbol Spoofing", "url": "https://paypal.com:security-update@attacker-controlled-host.com/login"},
        {"name": "⚠️ Cyrillic Unicode Homoglyph Attack (pаypal)", "url": "http://p\u0430yp\u0430l.com/verify-identity"},
        {"name": "⚠️ Dangerous 'javascript:' Scheme Injection", "url": "javascript:alert('Credentials Harvested');"},
        {"name": "✅ Legitimate Wikipedia Article (Clean)", "url": "https://en.wikipedia.org/wiki/Deterministic_finite_automaton"},
        {"name": "✅ Legitimate Official PayPal Portal (Clean)", "url": "https://www.paypal.com/signin"},
    ]

    selected_demo = st.selectbox(
        "Quick Test Presets (for college presentation):",
        flat_presets,
        format_func=lambda x: x["name"],
        key="demo_preset_select",
        label_visibility="collapsed",
    )

    if selected_demo["url"] and selected_demo["url"] != st.session_state.get("url_input_val"):
        st.session_state["url_input_val"] = selected_demo["url"]
        st.session_state["analyzed_url"] = selected_demo["url"]
        st.rerun()

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        target_url = st.text_input(
            "Enter URL string w ∈ Σ* to scan",
            value=st.session_state.get("url_input_val", DEFAULT_URL),
            placeholder="e.g. http://paypal.com.verify.top/login or https://example.com",
            label_visibility="collapsed",
            key="url_input_field",
        )
    with col_btn:
        scan_clicked = st.button("Scan URL", use_container_width=True, type="primary", key="btn_scan")

    if scan_clicked:
        st.session_state["url_input_val"] = target_url
        st.session_state["analyzed_url"] = target_url
    elif target_url != st.session_state.get("analyzed_url"):
        st.session_state["url_input_val"] = target_url
        st.session_state["analyzed_url"] = target_url

    curr_url = st.session_state.get("analyzed_url", target_url).strip()
    st.markdown("<hr style='border: none; border-top: 1px solid #30363d; margin: 24px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN EVALUATION PIPELINE
    # -------------------------------------------------------------------------
    if curr_url:
        try:
            parsed = URLParser.parse(curr_url)
            dfa_suite_res = suite.analyze_parsed_url(parsed)
            rule_report = rule_engine.evaluate(parsed, dfa_suite_res)
            assessment = RiskScorer.calculate_assessment(parsed, dfa_suite_res, rule_report)

            # -----------------------------------------------------------------
            # SECTION 2: DETECTION RESULT
            # -----------------------------------------------------------------
            st.markdown('<div class="toc-section-title">SECTION 2: DETECTION RESULT</div>', unsafe_allow_html=True)
            render_detection_result(assessment, rule_report)

            st.markdown("<hr style='border: none; border-top: 1px solid #30363d; margin: 24px 0;'>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # SECTION 3: AUTOMATA WORKING (CORE EDUCATIONAL SECTION)
            # -----------------------------------------------------------------
            st.markdown('<div class="toc-section-title">SECTION 3: AUTOMATA WORKING (TOC DEMONSTRATION)</div>', unsafe_allow_html=True)

            triggered_dfa_names = dfa_suite_res.triggered_automata
            all_dfa_names = list(suite.dfa_registry.keys())
            default_dfa_name = triggered_dfa_names[0] if triggered_dfa_names else "IPv4-Host-Automaton"

            def format_dfa_option(name: str) -> str:
                if name in triggered_dfa_names:
                    return f"⚡ {name} (Pattern Accepted / Triggered)"
                return f"⚪ {name} (Clean / Rejected)"

            selected_dfa_name = st.selectbox(
                "Select Detection Automaton to Inspect:",
                all_dfa_names,
                index=all_dfa_names.index(default_dfa_name),
                format_func=format_dfa_option,
                key="active_dfa_selector",
            )

            active_dfa = suite.dfa_registry[selected_dfa_name]
            token_tested = get_token_for_dfa(selected_dfa_name, parsed)
            if not token_tested:
                token_tested = parsed.host or curr_url

            trace_result = active_dfa.simulate(token_tested)

            if trace_result.accepted:
                explanation = (
                    f"The automaton <strong>{active_dfa.name}</strong> processed the input token "
                    f"<code>\"{token_tested}\"</code> character-by-character and successfully reached the terminal accepting state "
                    f"<code>{trace_result.final_state}</code>. In formal language theory, this proves that "
                    f"<strong><em>w &isin; L(M)</em></strong> (the string belongs to the language recognized by this DFA), "
                    f"confirming the presence of a suspicious pattern."
                )
            else:
                explanation = (
                    f"The automaton <strong>{active_dfa.name}</strong> processed the input token "
                    f"<code>\"{token_tested}\"</code> from initial state <code>{active_dfa.start_state}</code> and halted in "
                    f"<code>{trace_result.final_state}</code> without reaching any accepting state. "
                    f"Therefore, <strong><em>w &notin; L(M)</em></strong> (the string is rejected by this automaton), "
                    f"meaning no adversarial signature for this specific rule was present."
                )

            render_automata_working(active_dfa, trace_result, token_tested, explanation)

            with st.expander("📐 View Automata Details (Formal 5-Tuple & Transition Table)"):
                st.markdown(
                    f"""
                    **Formal 5-Tuple Specification:**
                    $$M = (Q, \\Sigma, \\delta, q_0, F)$$
                    - **States Set $Q$:** {len(active_dfa.states)} discrete states
                    - **Alphabet $\\Sigma$:** ASCII characters relevant to URL syntax
                    - **Initial State $q_0$:** `{active_dfa.start_state}`
                    - **Accepting States $F$:** {len(active_dfa.accept_states)} states (`{", ".join(list(active_dfa.accept_states)[:4])}...`)
                    - **Transition Function $\\delta$:** $\\delta: Q \\times \\Sigma \\to Q$ (Deterministic $O(1)$ lookup per symbol)
                    """
                )

                tab_tt, tab_trace = st.tabs(["Transition Table δ(q, σ)", "Step-by-Step State Trace"])
                with tab_tt:
                    st.caption("Formal state transition mapping for this automaton:")
                    tt_df = DFAVisualizer.get_transition_table(active_dfa)
                    st.dataframe(tt_df, use_container_width=True, hide_index=True)

                with tab_trace:
                    st.caption(f"Exact character execution steps for token '{token_tested}':")
                    trace_df = DFAVisualizer.create_trace_dataframe(trace_result)
                    st.dataframe(trace_df, use_container_width=True, hide_index=True)

        except Exception as ex:
            st.error(f"❌ Error during URL automata analysis: {str(ex)}")
            with st.expander("🔍 Error Traceback Details"):
                st.code(traceback.format_exc(), language="python")
    else:
        st.info("ℹ️ Enter a target URL in the box above or pick a demo preset, then click **Scan URL**.")

    st.markdown("<hr style='border: none; border-top: 1px solid #30363d; margin: 24px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 4: HOW PHISHGUARD WORKS
    # -------------------------------------------------------------------------
    st.markdown('<div class="toc-section-title">SECTION 4: HOW IT WORKS & SYLLABUS MAPPING</div>', unsafe_allow_html=True)
    render_how_it_works_section()

    # Viva Sandbox utility
    with st.expander("🧪 Automata Sandbox: Test Any Custom String (Viva Live Experimentation)"):
        st.caption("Type any custom string symbol-by-symbol to observe DFA state transitions in real time:")
        col_sb_dfa, col_sb_in = st.columns([1, 2])
        with col_sb_dfa:
            sb_dfa_name = st.selectbox("Choose DFA:", list(suite.dfa_registry.keys()), key="sandbox_dfa_sel")
        with col_sb_in:
            sb_input = st.text_input("Custom input string w:", value="192.168.1.1", key="sandbox_input_str")

        if sb_input:
            sb_dfa = suite.dfa_registry[sb_dfa_name]
            sb_res = sb_dfa.simulate(sb_input)
            col_sb1, col_sb2, col_sb3 = st.columns(3)
            with col_sb1:
                st.metric("Status", "ACCEPTED 🎯" if sb_res.accepted else "REJECTED ❌")
            with col_sb2:
                st.metric("Final State", sb_res.final_state)
            with col_sb3:
                st.metric("Length |w|", len(sb_input))
            sb_mermaid = sb_dfa.to_mermaid(highlight_path=sb_res.path)
            st.markdown(f"```mermaid\n{sb_mermaid}\n```")

    with st.expander("📂 Batch URL Scanner (Optional Bulk Inspection)"):
        sample_batch = "https://en.wikipedia.org/wiki/DFA\nhttp://paypal.com.verify.top/login\nhttp://185.220.101.5:8080/bank\nhttps://google.com"
        batch_text = st.text_area("Paste URLs (one per line):", value=sample_batch, height=120, key="batch_ta")
        if st.button("Run Batch Scan", key="btn_batch_run"):
            batch_lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            rows = []
            for line in batch_lines:
                p = URLParser.parse(line)
                d = suite.analyze_parsed_url(p)
                r = rule_engine.evaluate(p, d)
                a = RiskScorer.calculate_assessment(p, d, r)
                rows.append({
                    "URL": line,
                    "Risk Score": f"{a.risk_score:.0f}/100",
                    "Classification": a.threat_level.value,
                    "Automata Triggered": len(d.triggered_automata),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style="text-align: center; color: #8b949e; font-size: 0.75rem; margin-top: 30px; border-top: 1px solid #30363d; padding-top: 14px; font-family: 'JetBrains Mono', monospace;">
            PhishGuard — Automata-Based Phishing URL Detection | Theory of Computation Course Project
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
