"""PhishGuard — Automata-Based Phishing URL Detection Backend.

Theory of Computation / Formal Language Theory Implementation.
Implements:
1. Formal Deterministic Finite Automata (DFA) engine (5-tuple M = (Q, Σ, δ, q0, F))
2. Specialized Detection DFAs (IP, schemes, abusive TLDs, keywords, brands, auth abuse, encodings, subdomains)
3. Zero-execution static URL parser and Shannon entropy analyzer
4. Calibrated cybersecurity heuristic rules and explainable risk scorer
5. Educational benchmark datasets for viva demonstration
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd


# =============================================================================
# 1. FORMAL AUTOMATA CORE (Theory of Computation 5-Tuple Engine)
# =============================================================================

@dataclass
class DFAState:
    """Represents a discrete state q ∈ Q in a Deterministic Finite Automaton."""
    name: str
    is_start: bool = False
    is_accept: bool = False
    description: str = ""
    label: Optional[str] = None


@dataclass
class DFAStep:
    """Represents a single deterministic transition: δ(from_state, symbol) = to_state."""
    step_number: int
    from_state: str
    symbol: str
    to_state: str
    is_accept: bool = False
    description: str = ""


@dataclass
class DFAMatch:
    """Represents an accepted substring pattern w ∈ L(M) identified in an input string."""
    pattern_name: str
    matched_value: str
    start_pos: int
    end_pos: int
    final_state: str
    risk_weight: float = 0.0
    details: str = ""


@dataclass
class DFAResult:
    """Complete execution trace and forensic result of running a string through a DFA."""
    automaton_name: str
    input_string: str
    accepted: bool
    final_state: str
    path: List[DFAStep] = field(default_factory=list)
    state_sequence: List[str] = field(default_factory=list)
    matches: List[DFAMatch] = field(default_factory=list)
    halt_reason: str = "END_OF_INPUT"


class DFA:
    """Formal Deterministic Finite Automaton (DFA).

    Mathematically defined as the 5-tuple:
        M = (Q, Σ, δ, q_0, F)
    Where:
        Q:   Finite set of states
        Σ:   Input alphabet
        δ:   State transition function Q × Σ → Q
        q_0: Initial / start state (q_0 ∈ Q)
        F:   Set of accepting / terminal states (F ⊆ Q)
    """

    def __init__(
        self,
        name: str,
        description: str,
        start_state: str,
        accept_states: Optional[Set[str]] = None,
        dead_state: str = "q_trap",
        category: str = "Automata Detection",
        risk_weight: float = 20.0,
    ) -> None:
        self.name = name
        self.description = description
        self.start_state = start_state
        self.accept_states: Set[str] = set(accept_states or [])
        self.dead_state = dead_state
        self.category = category
        self.risk_weight = risk_weight

        self.states: Dict[str, DFAState] = {}
        self.alphabet: Set[str] = set()
        # transitions: δ(q, σ) -> q'  structured as {state: {symbol: next_state}}
        self.transitions: Dict[str, Dict[str, str]] = {}

        # Register start state
        self.add_state(start_state, is_start=True, description="Initial State")
        for acc in self.accept_states:
            self.add_state(acc, is_accept=True, description="Terminal Accepting State")

    def add_state(
        self,
        name: str,
        is_start: bool = False,
        is_accept: bool = False,
        description: str = "",
        label: Optional[str] = None,
    ) -> None:
        """Registers a state q ∈ Q into the automaton."""
        if is_accept:
            self.accept_states.add(name)
        self.states[name] = DFAState(
            name=name,
            is_start=is_start,
            is_accept=is_accept,
            description=description,
            label=label or name,
        )
        if name not in self.transitions:
            self.transitions[name] = {}

    def add_transition(self, from_state: str, symbol: str, to_state: str) -> None:
        """Defines a deterministic transition: δ(from_state, symbol) = to_state."""
        if from_state not in self.states:
            self.add_state(from_state)
        if to_state not in self.states:
            self.add_state(to_state)

        self.transitions[from_state][symbol] = to_state
        self.alphabet.add(symbol)

    def add_character_class_transitions(self, from_state: str, characters: List[str], to_state: str) -> None:
        """Defines transitions for a set of characters to a target state."""
        for ch in characters:
            self.add_transition(from_state, ch, to_state)

    def transition(self, current_state: str, symbol: str) -> str:
        """Evaluates δ(q, σ). Returns dead state if undefined."""
        state_transitions = self.transitions.get(current_state, {})
        if symbol in state_transitions:
            return state_transitions[symbol]
        if "*" in state_transitions:
            return state_transitions["*"]
        return self.dead_state

    def simulate(self, input_string: str) -> DFAResult:
        """Simulates the DFA on string w ∈ Σ* step-by-step and records state execution path."""
        curr = self.start_state
        path: List[DFAStep] = []
        state_seq = [curr]
        halt_reason = "END_OF_INPUT"

        for step_idx, sym in enumerate(input_string, start=1):
            next_state = self.transition(curr, sym)
            is_acc = next_state in self.accept_states

            step_desc = self.states[next_state].description if next_state in self.states else ""
            if next_state == self.dead_state:
                step_desc = "Trapped in Dead/Reject State"

            path.append(
                DFAStep(
                    step_number=step_idx,
                    from_state=curr,
                    symbol=sym,
                    to_state=next_state,
                    is_accept=is_acc,
                    description=step_desc,
                )
            )
            curr = next_state
            state_seq.append(curr)

            if curr == self.dead_state:
                halt_reason = "TRAP_STATE_REACHED"
                break

        accepted = curr in self.accept_states
        matches: List[DFAMatch] = []
        if accepted:
            matches.append(
                DFAMatch(
                    pattern_name=self.name,
                    matched_value=input_string,
                    start_pos=0,
                    end_pos=len(input_string),
                    final_state=curr,
                    risk_weight=self.risk_weight,
                    details=f"Entire input accepted in state '{curr}'",
                )
            )

        return DFAResult(
            automaton_name=self.name,
            input_string=input_string,
            accepted=accepted,
            final_state=curr,
            path=path,
            state_sequence=state_seq,
            matches=matches,
            halt_reason=halt_reason,
        )

    def scan_substrings(self, text: str) -> List[DFAMatch]:
        """Scans all substring tokens to find matches w ∈ L(M)."""
        matches: List[DFAMatch] = []
        n = len(text)
        if n == 0:
            return matches

        for i in range(n):
            curr = self.start_state
            for j in range(i, n):
                char = text[j]
                nxt = self.transition(curr, char)
                if nxt == self.dead_state:
                    break
                curr = nxt
                if curr in self.accept_states:
                    matched_text = text[i : j + 1]
                    matches.append(
                        DFAMatch(
                            pattern_name=self.name,
                            matched_value=matched_text,
                            start_pos=i,
                            end_pos=j + 1,
                            final_state=curr,
                            risk_weight=self.risk_weight,
                            details=f"Accepted token '{matched_text}' in state '{curr}'",
                        )
                    )
        return matches

    def to_mermaid(self, highlight_path: Optional[List[DFAStep]] = None) -> str:
        """Generates clean Mermaid state diagram code with the active execution path highlighted."""
        lines = ["flowchart LR"]
        lines.append("    classDef default fill:#161b22,stroke:#30363d,stroke-width:1px,color:#e6edf3;")
        lines.append("    classDef startState fill:#1f3d59,stroke:#58a6ff,stroke-width:2px,color:#79c0ff;")
        lines.append("    classDef acceptState fill:#1c3b2b,stroke:#3fb950,stroke-width:2px,color:#56d364;")
        lines.append("    classDef deadState fill:#3d1d24,stroke:#f85149,stroke-width:1px,color:#ff7b72;")
        lines.append("    classDef pathActive fill:#382c16,stroke:#d29922,stroke-width:2px,color:#e3b341;")

        def _clean_id(raw_id: str) -> str:
            return "".join(c if c.isalnum() or c == "_" else "_" for c in raw_id)

        def _clean_label(lbl: str) -> str:
            return lbl.replace('"', "'").replace("\n", " ").strip()

        # Build active path edges set
        path_edges: Set[Tuple[str, str, str]] = set()
        visited_states: Set[str] = set()
        if highlight_path:
            for s in highlight_path:
                path_edges.add((s.from_state, s.symbol, s.to_state))
                visited_states.add(s.from_state)
                visited_states.add(s.to_state)

        # Include either all states or first 16 to keep diagram clean and readable
        state_list = list(self.states.values())
        if len(state_list) > 16:
            visible_states = [st for st in state_list if st.name in visited_states or st.is_start or st.is_accept][:16]
        else:
            visible_states = state_list
        visible_names = {st.name for st in visible_states}

        for st in visible_states:
            cid = _clean_id(st.name)
            desc = _clean_label(st.label or st.name)
            if st.is_accept:
                lines.append(f'    {cid}((["🎯 {desc}"])):::acceptState')
            elif st.is_start:
                lines.append(f'    {cid}["▶ {desc}"]:::startState')
            elif "trap" in st.name.lower() or "dead" in st.name.lower():
                lines.append(f'    {cid}["❌ {desc}"]:::deadState')
            else:
                lines.append(f'    {cid}["{desc}"]')

        # Add transitions
        edge_count = 0
        for src, sym_map in self.transitions.items():
            if src not in visible_names:
                continue
            grouped: Dict[str, List[str]] = {}
            for sym, dst in sym_map.items():
                if dst in visible_names:
                    grouped.setdefault(dst, []).append(sym)

            for dst, syms in grouped.items():
                if edge_count >= 24:
                    break
                cid_src = _clean_id(src)
                cid_dst = _clean_id(dst)
                label_str = ", ".join(syms[:3])
                if len(syms) > 3:
                    label_str += f" (+{len(syms)-3})"
                label_str = _clean_label(label_str)

                is_active = any((src, s, dst) in path_edges for s in syms)
                if is_active:
                    lines.append(f"    {cid_src} == {label_str} ==> {cid_dst}")
                else:
                    lines.append(f"    {cid_src} -- {label_str} --> {cid_dst}")
                edge_count += 1

        return "\n".join(lines)


class DFAVisualizer:
    """Extracts formal transition tables and execution step trace tables for Streamlit."""

    @staticmethod
    def get_transition_table(dfa: DFA) -> pd.DataFrame:
        """Extracts complete transition matrix δ(q, σ) as a structured DataFrame."""
        rows = []
        for src, sym_map in dfa.transitions.items():
            st_obj = dfa.states.get(src)
            is_start = st_obj.is_start if st_obj else False
            is_acc = st_obj.is_accept if st_obj else False

            state_type = "Normal"
            if is_start and is_acc:
                state_type = "Start & Accept"
            elif is_start:
                state_type = "Start (q0)"
            elif is_acc:
                state_type = "Accept (F)"

            for sym, dst in sym_map.items():
                rows.append({
                    "Current State (q)": src,
                    "Type": state_type,
                    "Input Symbol (σ)": sym if sym != " " else "[SPACE]",
                    "Next State δ(q, σ)": dst,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def create_trace_dataframe(dfa_result: DFAResult) -> pd.DataFrame:
        """Converts step-by-step state simulation path into an educational trace table."""
        rows = []
        for step in dfa_result.path:
            status = "ACCEPT STATE 🎯" if step.is_accept else "Normal State"
            if "trap" in step.to_state.lower() or "dead" in step.to_state.lower():
                status = "TRAP / DEAD STATE ❌"

            rows.append({
                "Step": step.step_number,
                "From State": step.from_state,
                "Symbol (σ)": step.symbol if step.symbol != " " else "[SPACE]",
                "To State": step.to_state,
                "Status": status,
                "State Description": step.description or "-",
            })
        return pd.DataFrame(rows)


# =============================================================================
# 2. SPECIALIZED DETECTION AUTOMATA
# =============================================================================

def build_ip_dfa() -> DFA:
    """Constructs the IPv4 and Hexadecimal IP Recognition Automaton."""
    start_state = "q_start"
    accept_states: Set[str] = {
        "q_oct4_d1", "q_oct4_d2", "q_oct4_d3",
        "q_port_1", "q_port_2", "q_port_3", "q_port_4", "q_port_5",
        "q_hex_full",
    }
    dfa = DFA(
        name="IPv4-Host-Automaton",
        description="Formal DFA recognizing IPv4 dotted-decimal addresses, optional ports, and 32-bit hex IPs in hostnames.",
        start_state=start_state,
        accept_states=accept_states,
        dead_state="q_trap",
        category="Network Addressing",
        risk_weight=35.0,
    )
    digits = [str(d) for d in range(10)]
    hex_chars = digits + ["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"]

    dfa.add_state("q_start", is_start=True, description="Start: Awaiting digit or 0x")
    dfa.add_state("q_oct1_d1", description="Octet 1: 1st digit")
    dfa.add_state("q_oct1_d2", description="Octet 1: 2nd digit")
    dfa.add_state("q_oct1_d3", description="Octet 1: 3rd digit")
    dfa.add_state("q_dot1", description="Delimiter: Dot after Octet 1")

    dfa.add_state("q_oct2_d1", description="Octet 2: 1st digit")
    dfa.add_state("q_oct2_d2", description="Octet 2: 2nd digit")
    dfa.add_state("q_oct2_d3", description="Octet 2: 3rd digit")
    dfa.add_state("q_dot2", description="Delimiter: Dot after Octet 2")

    dfa.add_state("q_oct3_d1", description="Octet 3: 1st digit")
    dfa.add_state("q_oct3_d2", description="Octet 3: 2nd digit")
    dfa.add_state("q_oct3_d3", description="Octet 3: 3rd digit")
    dfa.add_state("q_dot3", description="Delimiter: Dot after Octet 3")

    dfa.add_state("q_oct4_d1", is_accept=True, description="Accept: Octet 4 (1 digit)")
    dfa.add_state("q_oct4_d2", is_accept=True, description="Accept: Octet 4 (2 digits)")
    dfa.add_state("q_oct4_d3", is_accept=True, description="Accept: Octet 4 (3 digits)")

    dfa.add_state("q_port_colon", description="Port delimiter ':'")
    for p in range(1, 6):
        dfa.add_state(f"q_port_{p}", is_accept=True, description=f"Accept: IPv4 with port ({p} digits)")

    dfa.add_state("q_hex_lead0", description="Leading '0'")
    dfa.add_state("q_hex_x", description="'0x' Hex prefix")
    for i in range(1, 8):
        dfa.add_state(f"q_hex_{i}", description=f"Hex digit {i}/8")
    dfa.add_state("q_hex_full", is_accept=True, description="Accept: Full 32-bit Hexadecimal IP (e.g. 0x7f000001)")

    # Octet 1 transitions
    for d in digits:
        dfa.add_transition("q_start", d, "q_hex_lead0" if d == "0" else "q_oct1_d1")
    dfa.add_transition("q_hex_lead0", ".", "q_dot1")
    dfa.add_transition("q_hex_lead0", "x", "q_hex_x")
    dfa.add_transition("q_hex_lead0", "X", "q_hex_x")
    for d in digits:
        dfa.add_transition("q_hex_lead0", d, "q_oct1_d2")
        dfa.add_transition("q_oct1_d1", d, "q_oct1_d2")
        dfa.add_transition("q_oct1_d2", d, "q_oct1_d3")
    dfa.add_transition("q_oct1_d1", ".", "q_dot1")
    dfa.add_transition("q_oct1_d2", ".", "q_dot1")
    dfa.add_transition("q_oct1_d3", ".", "q_dot1")

    # Octet 2 transitions
    for d in digits:
        dfa.add_transition("q_dot1", d, "q_oct2_d1")
        dfa.add_transition("q_oct2_d1", d, "q_oct2_d2")
        dfa.add_transition("q_oct2_d2", d, "q_oct2_d3")
    dfa.add_transition("q_oct2_d1", ".", "q_dot2")
    dfa.add_transition("q_oct2_d2", ".", "q_dot2")
    dfa.add_transition("q_oct2_d3", ".", "q_dot2")

    # Octet 3 transitions
    for d in digits:
        dfa.add_transition("q_dot2", d, "q_oct3_d1")
        dfa.add_transition("q_oct3_d1", d, "q_oct3_d2")
        dfa.add_transition("q_oct3_d2", d, "q_oct3_d3")
    dfa.add_transition("q_oct3_d1", ".", "q_dot3")
    dfa.add_transition("q_oct3_d2", ".", "q_dot3")
    dfa.add_transition("q_oct3_d3", ".", "q_dot3")

    # Octet 4 transitions
    for d in digits:
        dfa.add_transition("q_dot3", d, "q_oct4_d1")
        dfa.add_transition("q_oct4_d1", d, "q_oct4_d2")
        dfa.add_transition("q_oct4_d2", d, "q_oct4_d3")

    # Port transitions
    for oct4_s in ["q_oct4_d1", "q_oct4_d2", "q_oct4_d3"]:
        dfa.add_transition(oct4_s, ":", "q_port_colon")
    for d in digits:
        dfa.add_transition("q_port_colon", d, "q_port_1")
        dfa.add_transition("q_port_1", d, "q_port_2")
        dfa.add_transition("q_port_2", d, "q_port_3")
        dfa.add_transition("q_port_3", d, "q_port_4")
        dfa.add_transition("q_port_4", d, "q_port_5")

    # Hex IP transitions
    for h in hex_chars:
        dfa.add_transition("q_hex_x", h, "q_hex_1")
        dfa.add_transition("q_hex_1", h, "q_hex_2")
        dfa.add_transition("q_hex_2", h, "q_hex_3")
        dfa.add_transition("q_hex_3", h, "q_hex_4")
        dfa.add_transition("q_hex_4", h, "q_hex_5")
        dfa.add_transition("q_hex_5", h, "q_hex_6")
        dfa.add_transition("q_hex_6", h, "q_hex_7")
        dfa.add_transition("q_hex_7", h, "q_hex_full")

    return dfa


def _build_trie_dfa(name: str, desc: str, word_map: Dict[str, str], risk_weight: float) -> DFA:
    """Helper constructing a deterministic prefix-tree DFA from a word dictionary."""
    start_state = "q0"
    accept_states: Set[str] = set()
    dfa = DFA(name=name, description=desc, start_state=start_state, accept_states=accept_states, risk_weight=risk_weight)

    trie: Dict[str, Dict[str, str]] = {start_state: {}}
    state_desc: Dict[str, str] = {start_state: "Root"}
    accept_info: Dict[str, str] = {}
    counter = 0

    for word, meaning in word_map.items():
        curr = start_state
        for ch in word:
            c_low = ch.lower()
            if c_low not in trie[curr]:
                counter += 1
                nxt = f"q_{counter}"
                trie[curr][c_low] = nxt
                trie[curr][c_low.upper()] = nxt
                trie[nxt] = {}
                state_desc[nxt] = f"Prefix: '{word[:counter]}'"
            curr = trie[curr][c_low]
        accept_info[curr] = meaning
        accept_states.add(curr)

    for st, d in state_desc.items():
        is_acc = st in accept_info
        dfa.add_state(name=st, is_accept=is_acc, is_start=(st == start_state), description=accept_info.get(st, d))

    for src, smap in trie.items():
        for sym, dst in smap.items():
            dfa.add_transition(src, sym, dst)

    return dfa


def build_dangerous_scheme_dfa() -> DFA:
    """Constructs DFA recognizing dangerous executable URI schemes."""
    schemes = {
        "javascript:": "Inline script execution scheme",
        "data:": "Data URI payload embedding",
        "file:": "Local system file access",
        "blob:": "In-memory blob payload execution",
        "ftp:": "Insecure plain legacy protocol",
    }
    return _build_trie_dfa("Dangerous-Scheme-Automaton", "Formal DFA recognizing dangerous URI schemes.", schemes, 40.0)


def build_abusive_tld_dfa() -> DFA:
    """Constructs DFA recognizing high-abuse Top-Level Domains."""
    abusive_tlds = {
        ".tk": "High-risk free registrar TLD",
        ".ml": "High-risk free registrar TLD",
        ".ga": "High-risk free registrar TLD",
        ".cf": "High-risk free registrar TLD",
        ".gq": "High-risk free registrar TLD",
        ".xyz": "Heavily abused low-cost TLD",
        ".top": "Heavily abused low-cost TLD",
        ".buzz": "High spam correlation TLD",
        ".club": "Commonly abused generic TLD",
        ".work": "Commonly abused generic TLD",
        ".icu": "High spam correlation TLD",
        ".loan": "High financial phishing TLD",
        ".fit": "Commonly abused spam TLD",
        ".surf": "Commonly abused spam TLD",
        ".zip": "Deceptive file-extension TLD",
        ".mov": "Deceptive video-extension TLD",
        ".kim": "High abuse score TLD",
        ".country": "High spam correlation TLD",
    }
    return _build_trie_dfa("Abusive-TLD-Automaton", "Formal DFA recognizing top-level domains associated with disposable spam registrars.", abusive_tlds, 25.0)


def build_scheme_spoofing_dfa() -> DFA:
    """Constructs DFA recognizing deceptive protocol prefixes in hostname labels."""
    prefixes = {
        "https-": "Deceptive https- prefix in host",
        "http-": "Deceptive http- prefix in host",
        "https.": "Deceptive https. subdomain prefix",
        "http.": "Deceptive http. subdomain prefix",
        "ssl-": "Deceptive ssl- prefix in host",
        "secure-https": "Deceptive secure-https prefix",
    }
    return _build_trie_dfa("Scheme-Spoofing-Automaton", "Formal DFA recognizing fake protocol prefixes embedded in hostnames.", prefixes, 30.0)


def build_keyword_automaton(name: str, desc: str, words: List[str], risk_weight: float) -> DFA:
    """Constructs a DFA recognizing keywords with leetspeak character substitutions."""
    start_state = "kw_0"
    accept_states: Set[str] = set()
    dfa = DFA(name=name, description=desc, start_state=start_state, accept_states=accept_states, risk_weight=risk_weight)

    subs = {
        "a": ["a", "4", "@"],
        "b": ["b", "8"],
        "e": ["e", "3"],
        "i": ["i", "1", "!"],
        "l": ["l", "1", "|"],
        "o": ["o", "0"],
        "s": ["s", "5", "$"],
        "t": ["t", "7"],
        "g": ["g", "9"],
    }
    trie: Dict[str, Dict[str, str]] = {start_state: {}}
    state_desc: Dict[str, str] = {start_state: "Root"}
    accepted_words: Dict[str, str] = {}
    counter = 0

    for word in words:
        curr = start_state
        matched_prefix = ""
        for ch in word.lower():
            matched_prefix += ch
            chars = subs.get(ch, [ch])
            existing = None
            for c in chars:
                if c in trie[curr]:
                    existing = trie[curr][c]
                    break
            if not existing:
                counter += 1
                new_state = f"kw_s{counter}"
                trie[new_state] = {}
                state_desc[new_state] = f"Prefix: '{matched_prefix}'"
                existing = new_state
            for c in chars:
                if c not in trie[curr]:
                    trie[curr][c] = existing
                if c.isalpha():
                    trie[curr][c.upper()] = existing
            curr = existing
        accepted_words[curr] = word
        accept_states.add(curr)

    for st, d in state_desc.items():
        is_acc = st in accepted_words
        dfa.add_state(name=st, is_accept=is_acc, is_start=(st == start_state), description=f"Matched: '{accepted_words[st]}'" if is_acc else d)

    for src, smap in trie.items():
        for sym, dst in smap.items():
            dfa.add_transition(src, sym, dst)

    return dfa


def build_brand_dfa() -> DFA:
    """Builds Brand Impersonation Recognition Automaton."""
    brands = [
        "paypal", "apple", "microsoft", "google", "netflix", "amazon",
        "facebook", "instagram", "binance", "coinbase", "metamask",
        "chase", "wellsfargo", "bankofamerica", "outlook", "office365",
    ]
    return build_keyword_automaton(
        "Brand-Impersonation-Automaton",
        "Formal DFA recognizing targeted high-profile brands with leetspeak tolerance.",
        brands,
        30.0,
    )


def build_action_keyword_dfa() -> DFA:
    """Builds Credential Harvesting & Urgency Action Keywords Automaton."""
    keywords = [
        "login", "signin", "verify", "verification", "security", "account",
        "banking", "secure", "update", "wallet", "support", "billing",
        "confirm", "suspended", "recovery", "password", "reset", "authenticate",
    ]
    return build_keyword_automaton(
        "Phishing-Intent-Automaton",
        "Formal DFA recognizing urgency and credential harvesting keywords.",
        keywords,
        20.0,
    )


def build_percent_encoding_dfa() -> DFA:
    """Constructs DFA recognizing percent-encoded hex sequences (%XX)."""
    start_state = "pct_0"
    accept_states = {
        "pct_accept_generic", "pct_accept_critical_dot", "pct_accept_critical_slash",
        "pct_accept_critical_at", "pct_accept_critical_null", "pct_accept_double_enc",
    }
    dfa = DFA(
        name="Percent-Encoding-Automaton",
        description="Formal DFA recognizing hex percent-encoding and critical delimiter obfuscation (%2e, %2f, %40).",
        start_state=start_state,
        accept_states=accept_states,
        risk_weight=20.0,
    )
    hex_digits = [str(d) for d in range(10)] + ["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"]

    dfa.add_state("pct_0", is_start=True, description="Start: Awaiting '%'")
    dfa.add_state("pct_pct", description="Read '%'")
    dfa.add_state("pct_hex_2", description="Read '%2'")
    dfa.add_state("pct_hex_4", description="Read '%4'")
    dfa.add_state("pct_hex_0", description="Read '%0'")
    dfa.add_state("pct_hex_other", description="Read generic first hex")

    dfa.add_state("pct_accept_critical_dot", is_accept=True, description="Accept: '%2e' (Dot '.')")
    dfa.add_state("pct_accept_critical_slash", is_accept=True, description="Accept: '%2f' (Slash '/')")
    dfa.add_state("pct_accept_critical_at", is_accept=True, description="Accept: '%40' ('@' symbol)")
    dfa.add_state("pct_accept_critical_null", is_accept=True, description="Accept: '%00' (Null byte)")
    dfa.add_state("pct_accept_double_enc", is_accept=True, description="Accept: '%25' (Double percent '%')")
    dfa.add_state("pct_accept_generic", is_accept=True, description="Accept: Generic '%XX'")

    dfa.add_transition("pct_0", "%", "pct_pct")
    dfa.add_transition("pct_pct", "2", "pct_hex_2")
    dfa.add_transition("pct_pct", "4", "pct_hex_4")
    dfa.add_transition("pct_pct", "0", "pct_hex_0")
    for h in [h for h in hex_digits if h not in ("2", "4", "0")]:
        dfa.add_transition("pct_pct", h, "pct_hex_other")

    dfa.add_transition("pct_hex_2", "e", "pct_accept_critical_dot")
    dfa.add_transition("pct_hex_2", "E", "pct_accept_critical_dot")
    dfa.add_transition("pct_hex_2", "f", "pct_accept_critical_slash")
    dfa.add_transition("pct_hex_2", "F", "pct_accept_critical_slash")
    dfa.add_transition("pct_hex_2", "5", "pct_accept_double_enc")
    for h in [h for h in hex_digits if h not in ("e", "E", "f", "F", "5")]:
        dfa.add_transition("pct_hex_2", h, "pct_accept_generic")

    dfa.add_transition("pct_hex_4", "0", "pct_accept_critical_at")
    for h in [h for h in hex_digits if h != "0"]:
        dfa.add_transition("pct_hex_4", h, "pct_accept_generic")

    dfa.add_transition("pct_hex_0", "0", "pct_accept_critical_null")
    for h in [h for h in hex_digits if h != "0"]:
        dfa.add_transition("pct_hex_0", h, "pct_accept_generic")

    for h in hex_digits:
        dfa.add_transition("pct_hex_other", h, "pct_accept_generic")

    return dfa


def build_credential_abuse_dfa() -> DFA:
    """Constructs DFA recognizing the '@' symbol in URL authorities."""
    start_state = "cred_0"
    accept_states = {"cred_accept_at", "cred_accept_user_host"}
    dfa = DFA(
        name="Credential-Abuse-Automaton",
        description="Formal DFA recognizing '@' symbol credential stuffing and hostname spoofing in authority strings.",
        start_state=start_state,
        accept_states=accept_states,
        risk_weight=35.0,
    )
    dfa.add_state("cred_0", is_start=True, description="Start: Scanning authority")
    dfa.add_state("cred_char", description="Scanning userinfo or fake brand prefix")
    dfa.add_state("cred_colon", description="Encountered ':' (possible fake port/password)")
    dfa.add_state("cred_pass", description="Scanning post-colon password segment")
    dfa.add_state("cred_accept_at", is_accept=True, description="Accept: Encountered '@' authority boundary delimiter")
    dfa.add_state("cred_accept_user_host", is_accept=True, description="Accept: Destination host after '@'")

    chars = [chr(c) for c in range(32, 127) if chr(c) not in ("@", "/", "\\")]
    for c in chars:
        if c == ":":
            dfa.add_transition("cred_0", c, "cred_colon")
            dfa.add_transition("cred_char", c, "cred_colon")
        else:
            dfa.add_transition("cred_0", c, "cred_char")
            dfa.add_transition("cred_char", c, "cred_char")
            dfa.add_transition("cred_colon", c, "cred_pass")
            dfa.add_transition("cred_pass", c, "cred_pass")

    dfa.add_transition("cred_0", "@", "cred_accept_at")
    dfa.add_transition("cred_char", "@", "cred_accept_at")
    dfa.add_transition("cred_colon", "@", "cred_accept_at")
    dfa.add_transition("cred_pass", "@", "cred_accept_at")

    for c in chars:
        dfa.add_transition("cred_accept_at", c, "cred_accept_user_host")
        dfa.add_transition("cred_accept_user_host", c, "cred_accept_user_host")

    return dfa


def build_subdomain_depth_dfa() -> DFA:
    """Constructs DFA tracking subdomain depth and detecting subdomain flooding."""
    start_state = "sub_start"
    accept_states = {"sub_level_4_accept", "sub_level_5_flood"}
    dfa = DFA(
        name="Subdomain-Flooding-Automaton",
        description="Formal DFA tracking dot transitions to detect excessive subdomain nesting and brand stacking.",
        start_state=start_state,
        accept_states=accept_states,
        risk_weight=20.0,
    )
    dfa.add_state("sub_start", is_start=True, description="Start: Label 1")
    dfa.add_state("sub_label_1", description="Reading 1st domain label")
    dfa.add_state("sub_dot_1", description="Dot 1: Transition to Level 2 (SLD)")
    dfa.add_state("sub_label_2", description="Reading 2nd domain label")
    dfa.add_state("sub_dot_2", description="Dot 2: Transition to Level 3 (Subdomain 1)")
    dfa.add_state("sub_label_3", description="Reading 3rd domain label")
    dfa.add_state("sub_dot_3", description="Dot 3: Transition to Level 4")
    dfa.add_state("sub_level_4_accept", is_accept=True, description="Accept: 4 Hierarchy Levels (High Subdomain Depth)")
    dfa.add_state("sub_dot_4", is_accept=True, description="Dot 4+: Excessive Delimiters")
    dfa.add_state("sub_level_5_flood", is_accept=True, description="Accept: 5+ Hierarchy Levels (Subdomain Flooding Attack)")

    chars = [chr(c) for c in range(ord("a"), ord("z") + 1)] + \
            [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
            [str(d) for d in range(10)] + ["-", "_"]

    for c in chars:
        dfa.add_transition("sub_start", c, "sub_label_1")
        dfa.add_transition("sub_label_1", c, "sub_label_1")
    dfa.add_transition("sub_label_1", ".", "sub_dot_1")

    for c in chars:
        dfa.add_transition("sub_dot_1", c, "sub_label_2")
        dfa.add_transition("sub_label_2", c, "sub_label_2")
    dfa.add_transition("sub_label_2", ".", "sub_dot_2")

    for c in chars:
        dfa.add_transition("sub_dot_2", c, "sub_label_3")
        dfa.add_transition("sub_label_3", c, "sub_label_3")
    dfa.add_transition("sub_label_3", ".", "sub_dot_3")

    for c in chars:
        dfa.add_transition("sub_dot_3", c, "sub_level_4_accept")
        dfa.add_transition("sub_level_4_accept", c, "sub_level_4_accept")
    dfa.add_transition("sub_level_4_accept", ".", "sub_dot_4")

    for c in chars:
        dfa.add_transition("sub_dot_4", c, "sub_level_5_flood")
        dfa.add_transition("sub_level_5_flood", c, "sub_level_5_flood")
    dfa.add_transition("sub_level_5_flood", ".", "sub_dot_4")

    return dfa


# =============================================================================
# 3. AUTOMATA SUITE COORDINATOR
# =============================================================================

@dataclass
class SuiteAnalysisResult:
    """Aggregated output from running the automata suite."""
    dfa_results: Dict[str, DFAResult] = field(default_factory=dict)
    all_matches: List[DFAMatch] = field(default_factory=list)
    triggered_automata: List[str] = field(default_factory=list)
    automata_risk_score: float = 0.0

    def get_result(self, automaton_name: str) -> Optional[DFAResult]:
        return self.dfa_results.get(automaton_name)


class PhishGuardAutomataSuite:
    """Coordinates parallel evaluation across all 9 specialized DFAs."""

    def __init__(self) -> None:
        self.ip_dfa = build_ip_dfa()
        self.dangerous_scheme_dfa = build_dangerous_scheme_dfa()
        self.abusive_tld_dfa = build_abusive_tld_dfa()
        self.scheme_spoofing_dfa = build_scheme_spoofing_dfa()
        self.brand_dfa = build_brand_dfa()
        self.action_keyword_dfa = build_action_keyword_dfa()
        self.percent_encoding_dfa = build_percent_encoding_dfa()
        self.credential_abuse_dfa = build_credential_abuse_dfa()
        self.subdomain_depth_dfa = build_subdomain_depth_dfa()

        self.dfa_registry: Dict[str, DFA] = {
            self.ip_dfa.name: self.ip_dfa,
            self.dangerous_scheme_dfa.name: self.dangerous_scheme_dfa,
            self.abusive_tld_dfa.name: self.abusive_tld_dfa,
            self.scheme_spoofing_dfa.name: self.scheme_spoofing_dfa,
            self.brand_dfa.name: self.brand_dfa,
            self.action_keyword_dfa.name: self.action_keyword_dfa,
            self.percent_encoding_dfa.name: self.percent_encoding_dfa,
            self.credential_abuse_dfa.name: self.credential_abuse_dfa,
            self.subdomain_depth_dfa.name: self.subdomain_depth_dfa,
        }

    def analyze_parsed_url(self, parsed: ParsedURL) -> SuiteAnalysisResult:
        """Runs each specialized DFA on its respective URL token substring."""
        results: Dict[str, DFAResult] = {}
        all_matches: List[DFAMatch] = []
        triggered: List[str] = []
        total_risk = 0.0

        # 1. IP DFA on host
        ip_res = self.ip_dfa.simulate(parsed.host or "")
        results[self.ip_dfa.name] = ip_res
        if ip_res.accepted:
            triggered.append(self.ip_dfa.name)
            all_matches.extend(ip_res.matches)
            total_risk += self.ip_dfa.risk_weight

        # 2. Dangerous Scheme DFA on scheme
        scheme_str = (parsed.scheme + ":") if parsed.scheme else ""
        scheme_res = self.dangerous_scheme_dfa.simulate(scheme_str)
        results[self.dangerous_scheme_dfa.name] = scheme_res
        if scheme_res.accepted:
            triggered.append(self.dangerous_scheme_dfa.name)
            all_matches.extend(scheme_res.matches)
            total_risk += self.dangerous_scheme_dfa.risk_weight

        # 3. Abusive TLD DFA on TLD
        tld_str = ("." + parsed.tld.lstrip(".")) if parsed.tld else ""
        tld_res = self.abusive_tld_dfa.simulate(tld_str)
        results[self.abusive_tld_dfa.name] = tld_res
        if tld_res.accepted:
            triggered.append(self.abusive_tld_dfa.name)
            all_matches.extend(tld_res.matches)
            total_risk += self.abusive_tld_dfa.risk_weight

        # 4. Subdomain Depth DFA on host
        sub_res = self.subdomain_depth_dfa.simulate(parsed.host or "")
        results[self.subdomain_depth_dfa.name] = sub_res
        if sub_res.accepted:
            triggered.append(self.subdomain_depth_dfa.name)
            all_matches.extend(sub_res.matches)
            total_risk += self.subdomain_depth_dfa.risk_weight

        # 5. Credential Abuse DFA on authority netloc
        cred_res = self.credential_abuse_dfa.simulate(parsed.netloc or "")
        results[self.credential_abuse_dfa.name] = cred_res
        if cred_res.accepted:
            triggered.append(self.credential_abuse_dfa.name)
            all_matches.extend(cred_res.matches)
            total_risk += self.credential_abuse_dfa.risk_weight

        # 6. Scheme Spoofing DFA on host & path
        sp_matches = self.scheme_spoofing_dfa.scan_substrings(parsed.host or "") + \
                     self.scheme_spoofing_dfa.scan_substrings(parsed.path or "")
        sp_res = self.scheme_spoofing_dfa.simulate(parsed.host or "")
        if sp_matches:
            sp_res.accepted = True
            sp_res.matches = sp_matches
            triggered.append(self.scheme_spoofing_dfa.name)
            all_matches.extend(sp_matches)
            total_risk += self.scheme_spoofing_dfa.risk_weight
        results[self.scheme_spoofing_dfa.name] = sp_res

        # 7. Brand Impersonation DFA on host & path
        brand_matches = self.brand_dfa.scan_substrings(parsed.host or "") + \
                        self.brand_dfa.scan_substrings(parsed.path or "")
        brand_res = self.brand_dfa.simulate(parsed.host or "")
        if brand_matches:
            brand_res.accepted = True
            brand_res.matches = brand_matches
            triggered.append(self.brand_dfa.name)
            all_matches.extend(brand_matches)
            total_risk += self.brand_dfa.risk_weight
        results[self.brand_dfa.name] = brand_res

        # 8. Action Keyword DFA on path & host
        act_matches = self.action_keyword_dfa.scan_substrings(parsed.path or "") + \
                      self.action_keyword_dfa.scan_substrings(parsed.host or "")
        act_res = self.action_keyword_dfa.simulate(parsed.path or "")
        if act_matches:
            act_res.accepted = True
            act_res.matches = act_matches
            triggered.append(self.action_keyword_dfa.name)
            all_matches.extend(act_matches)
            total_risk += self.action_keyword_dfa.risk_weight
        results[self.action_keyword_dfa.name] = act_res

        # 9. Percent-Encoding DFA on raw URL
        pct_matches = self.percent_encoding_dfa.scan_substrings(parsed.raw_url)
        pct_res = self.percent_encoding_dfa.simulate(parsed.raw_url)
        if pct_matches:
            pct_res.accepted = True
            pct_res.matches = pct_matches
            triggered.append(self.percent_encoding_dfa.name)
            all_matches.extend(pct_matches)
            total_risk += self.percent_encoding_dfa.risk_weight
        results[self.percent_encoding_dfa.name] = pct_res

        return SuiteAnalysisResult(
            dfa_results=results,
            all_matches=all_matches,
            triggered_automata=triggered,
            automata_risk_score=min(100.0, total_risk),
        )


# =============================================================================
# 4. STATIC URL PARSER & SHANNON ENTROPY (Zero-Execution)
# =============================================================================

KNOWN_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cli.gs", "tiny.cc",
    "snipurl.com", "short.to", "wp.me", "bit.do", "t.ly", "lnkd.in", "cutt.ly", "rb.gy",
}

TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au",
    "co.nz", "co.jp", "com.br", "co.in", "net.in", "org.in", "com.cn", "com.ru",
}

HOMOGLYPH_MAP: Dict[str, str] = {
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u0456": "i", "\u043e": "o",
    "\u0440": "p", "\u0443": "y", "\u0445": "x", "\u03bf": "o", "\u03bd": "v",
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H",
    "\u041e": "O", "\u0420": "P", "\u0422": "T", "\u0425": "X",
}


@dataclass
class ParsedURL:
    """Detailed structural decomposition of a URL string w."""
    raw_url: str
    normalized_url: str
    scheme: str
    netloc: str
    userinfo: str
    host: str
    port: Optional[int]
    path: str
    params: str
    query: str
    fragment: str
    subdomain: str
    registered_domain: str
    tld: str

    length_url: int
    length_host: int
    length_path: int
    length_query: int

    dot_count_url: int
    dot_count_host: int
    hyphen_count_host: int
    hyphen_count_url: int
    slash_count: int
    digit_count_url: int
    digit_count_host: int
    digit_ratio_host: float

    has_at_symbol: bool
    is_shortened: bool
    has_punycode: bool
    has_homoglyphs: bool
    homoglyphs_found: List[Tuple[str, str, str]] = field(default_factory=list)
    shannon_entropy_host: float = 0.0
    shannon_entropy_path: float = 0.0
    vowel_consonant_ratio_host: float = 0.0


class URLParser:
    """100% offline lexical decomposition and structural metrics calculator."""

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculates Shannon information entropy H(X) = -sum(p(x) * log2(p(x)))."""
        if not text:
            return 0.0
        length = len(text)
        counts = Counter(text)
        entropy = 0.0
        for count in counts.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        return float(entropy)

    @classmethod
    def decompose_domain(cls, host: str) -> Tuple[str, str, str]:
        """Separates host into (subdomain, registered_domain, tld)."""
        if not host:
            return "", "", ""
        if ":" in host:
            host = host.split(":")[0]
        parts = host.lower().split(".")
        if len(parts) == 1:
            return "", parts[0], ""

        if len(parts) >= 3:
            potential_tld = f"{parts[-2]}.{parts[-1]}"
            if potential_tld in TWO_LEVEL_TLDS:
                tld = potential_tld
                reg_domain = f"{parts[-3]}.{tld}"
                subdomain = ".".join(parts[:-3])
                return subdomain, reg_domain, tld

        tld = parts[-1]
        reg_domain = f"{parts[-2]}.{tld}"
        subdomain = ".".join(parts[:-2])
        return subdomain, reg_domain, tld

    @classmethod
    def detect_homoglyphs(cls, text: str) -> List[Tuple[str, str, str]]:
        """Identifies Cyrillic/Greek homoglyphs spoofing Latin characters."""
        findings = []
        for char in text:
            if char in HOMOGLYPH_MAP:
                char_name = unicodedata.name(char, "UNKNOWN")
                latin_equiv = HOMOGLYPH_MAP[char]
                findings.append((char, latin_equiv, char_name))
        return findings

    @classmethod
    def parse(cls, url: str) -> ParsedURL:
        """Parses URL string w ∈ Σ* into structured tokens and metrics."""
        cleaned = url.strip()
        working = cleaned
        has_scheme = "://" in working or working.startswith(("javascript:", "data:", "file:", "blob:", "ftp:"))
        if not has_scheme:
            working = "http://" + working

        try:
            parsed = urlparse(working)
        except Exception:
            working = "http://malformed.invalid/" + working
            parsed = urlparse(working)

        scheme = parsed.scheme if has_scheme else ""
        netloc = parsed.netloc or ""
        path = parsed.path or ""
        params = parsed.params or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        userinfo = ""
        host_port = netloc
        if "@" in netloc:
            parts = netloc.split("@", 1)
            userinfo = parts[0]
            host_port = parts[1]

        host = host_port
        port: Optional[int] = None
        if host_port.startswith("[") and "]" in host_port:
            end_bracket = host_port.find("]")
            host = host_port[1:end_bracket]
            after = host_port[end_bracket + 1 :]
            if after.startswith(":"):
                try:
                    port = int(after[1:])
                except (ValueError, OverflowError):
                    port = None
        elif ":" in host_port:
            hp_parts = host_port.rsplit(":", 1)
            host = hp_parts[0]
            try:
                port = int(hp_parts[1])
            except (ValueError, OverflowError):
                port = None

        subdomain, registered_domain, tld = cls.decompose_domain(host)

        length_url = len(cleaned)
        length_host = len(host)
        length_path = len(path)
        length_query = len(query)

        dot_count_url = cleaned.count(".")
        dot_count_host = host.count(".")
        hyphen_count_host = host.count("-")
        hyphen_count_url = cleaned.count("-")
        slash_count = cleaned.count("/")

        digit_count_url = sum(c.isdigit() for c in cleaned)
        digit_count_host = sum(c.isdigit() for c in host)
        digit_ratio_host = (digit_count_host / length_host) if length_host > 0 else 0.0

        vowels = set("aeiouAEIOU")
        letters = [c for c in host if c.isalpha()]
        vowel_count = sum(1 for c in letters if c in vowels)
        consonant_count = len(letters) - vowel_count
        vowel_consonant_ratio = (vowel_count / consonant_count) if consonant_count > 0 else 0.0

        has_at_symbol = "@" in netloc or "@" in cleaned
        is_shortened = registered_domain.lower() in KNOWN_URL_SHORTENERS or host.lower() in KNOWN_URL_SHORTENERS
        has_punycode = "xn--" in host.lower()
        homoglyphs = cls.detect_homoglyphs(host)
        has_homoglyphs = len(homoglyphs) > 0

        entropy_host = cls.calculate_shannon_entropy(host)
        entropy_path = cls.calculate_shannon_entropy(path)

        normalized_url = f"{scheme}://{host_port}{path}"
        if query:
            normalized_url += f"?{query}"
        if fragment:
            normalized_url += f"#{fragment}"

        return ParsedURL(
            raw_url=cleaned,
            normalized_url=normalized_url,
            scheme=scheme,
            netloc=netloc,
            userinfo=userinfo,
            host=host,
            port=port,
            path=path,
            params=params,
            query=query,
            fragment=fragment,
            subdomain=subdomain,
            registered_domain=registered_domain,
            tld=tld,
            length_url=length_url,
            length_host=length_host,
            length_path=length_path,
            length_query=length_query,
            dot_count_url=dot_count_url,
            dot_count_host=dot_count_host,
            hyphen_count_host=hyphen_count_host,
            hyphen_count_url=hyphen_count_url,
            slash_count=slash_count,
            digit_count_url=digit_count_url,
            digit_count_host=digit_count_host,
            digit_ratio_host=digit_ratio_host,
            has_at_symbol=has_at_symbol,
            is_shortened=is_shortened,
            has_punycode=has_punycode,
            has_homoglyphs=has_homoglyphs,
            homoglyphs_found=homoglyphs,
            shannon_entropy_host=entropy_host,
            shannon_entropy_path=entropy_path,
            vowel_consonant_ratio_host=vowel_consonant_ratio,
        )


# =============================================================================
# 5. DETECTION RULES & CALIBRATED RISK SCORER
# =============================================================================

class RuleSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleCategory(str, Enum):
    AUTOMATA = "Automata Recognition"
    STRUCTURAL = "Structural Analysis"
    LEXICAL = "Lexical & Homoglyphs"
    PROTOCOL = "Protocol & Transport"
    OBFUSCATION = "Evasion & Obfuscation"


@dataclass
class RuleResult:
    """Forensic result produced by evaluating a detection rule."""
    rule_id: str
    rule_name: str
    category: RuleCategory
    severity: RuleSeverity
    triggered: bool
    score_impact: float
    evidence: str
    explanation: str
    remediation: str


class BaseRule:
    """Base class for detection rules."""
    rule_id: str = "RULE-BASE"
    rule_name: str = "Base Rule"
    category: RuleCategory = RuleCategory.STRUCTURAL
    severity: RuleSeverity = RuleSeverity.MEDIUM
    base_weight: float = 15.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        raise NotImplementedError


# --- Automata Rules ---

class IPHostRule(BaseRule):
    rule_id = "RULE-DFA-IP-01"
    rule_name = "Raw / Obfuscated IP Address in Hostname"
    category = RuleCategory.AUTOMATA
    severity = RuleSeverity.CRITICAL
    base_weight = 35.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("IPv4-Host-Automaton")
        triggered = res is not None and res.accepted
        evidence = f"Host '{parsed.host}' accepted by IPv4-Host-Automaton in state '{res.final_state if res else ''}'" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Direct IP addresses in URLs bypass DNS reputation tracking and strongly correlate with phishing and C2 servers.",
            "Block access immediately unless this is a verified internal RFC 1918 management endpoint.",
        )


class DangerousSchemeRule(BaseRule):
    rule_id = "RULE-DFA-SCHEME-02"
    rule_name = "Dangerous / Executable URI Scheme"
    category = RuleCategory.PROTOCOL
    severity = RuleSeverity.CRITICAL
    base_weight = 40.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Dangerous-Scheme-Automaton")
        triggered = res is not None and res.accepted
        scheme_found = parsed.scheme + ":" if parsed.scheme else ""
        evidence = f"Scheme '{scheme_found}' reached accept state '{res.final_state if res else ''}'" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Schemes like 'javascript:', 'data:', and 'file:' allow direct inline script execution or local file access without network transport.",
            "Do not render or evaluate script payloads from this URL.",
        )


class AbusiveTLDRule(BaseRule):
    rule_id = "RULE-DFA-TLD-03"
    rule_name = "High-Risk / Abusive Top-Level Domain"
    category = RuleCategory.AUTOMATA
    severity = RuleSeverity.HIGH
    base_weight = 25.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Abusive-TLD-Automaton")
        triggered = res is not None and res.accepted
        evidence = f"TLD '.{parsed.tld}' matched abusive TLD automaton" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Certain TLDs (.xyz, .top, .tk, .zip, .buzz) feature minimal registration verification and are disproportionately abused in mass phishing.",
            "Exercise extreme caution before submitting credentials on domains registered under high-abuse TLDs.",
        )


class CredentialStuffingAtRule(BaseRule):
    rule_id = "RULE-DFA-AUTH-04"
    rule_name = "HTTP Basic Auth '@' Symbol Hostname Spoofing"
    category = RuleCategory.OBFUSCATION
    severity = RuleSeverity.CRITICAL
    base_weight = 35.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Credential-Abuse-Automaton")
        triggered = res is not None and res.accepted
        evidence = f"Authority contains '@' symbol delimiter: netloc='{parsed.netloc}'" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "In RFC 3986 URL syntax, text before '@' is userinfo, not the host. Attackers exploit this to spoof reputable domains (e.g. paypal.com@attacker.com).",
            "Filter URLs containing userinfo credentials in HTTP/HTTPS requests at web proxies.",
        )


class PercentEncodingObfuscationRule(BaseRule):
    rule_id = "RULE-DFA-OBF-05"
    rule_name = "Percent-Encoded Delimiter Obfuscation"
    category = RuleCategory.OBFUSCATION
    severity = RuleSeverity.HIGH
    base_weight = 20.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Percent-Encoding-Automaton")
        triggered = False
        evidence = ""
        if res and res.matches:
            critical = [m for m in res.matches if "critical" in m.final_state or "double" in m.final_state]
            if critical:
                triggered = True
                evidence = f"Detected {len(critical)} obfuscated delimiter(s): {', '.join(m.matched_value for m in critical)}"
            elif len(res.matches) >= 3:
                triggered = True
                evidence = f"High density of percent-encoding: {len(res.matches)} '%XX' tokens detected"
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Attackers encode dots (%2e), slashes (%2f), or null bytes (%00) to evade security perimeter pattern filters.",
            "Normalize and fully decode URLs before security analysis.",
        )


class BrandImpersonationRule(BaseRule):
    rule_id = "RULE-DFA-BRAND-06"
    rule_name = "Brand Trademark Impersonation in Host / Subdomain"
    category = RuleCategory.AUTOMATA
    severity = RuleSeverity.HIGH
    base_weight = 30.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Brand-Impersonation-Automaton")
        triggered = False
        evidence = ""
        if res and res.matches:
            matched_brands = {m.matched_value.lower() for m in res.matches}
            reg_domain = parsed.registered_domain.lower()
            suspicious = [b for b in matched_brands if not (reg_domain == f"{b}.com" or reg_domain.startswith(f"{b}."))]
            if suspicious:
                triggered = True
                evidence = f"Brand trademark(s) {suspicious} detected in hostile context (Registered domain is '{parsed.registered_domain}')"
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Embedding reputable brand names into subdomains or paths is the primary signature of credential harvesting phishing.",
            "Verify the root registered domain against authorized corporate asset records.",
        )


class PhishingIntentRule(BaseRule):
    rule_id = "RULE-DFA-ACTION-07"
    rule_name = "Credential Harvesting / Urgency Action Keywords"
    category = RuleCategory.AUTOMATA
    severity = RuleSeverity.MEDIUM
    base_weight = 20.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Phishing-Intent-Automaton")
        triggered = False
        evidence = ""
        if res and res.matches:
            kw = list({m.matched_value.lower() for m in res.matches})
            triggered = True
            evidence = f"Detected {len(kw)} action keyword(s): {', '.join(kw[:4])}"
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Keywords like 'login', 'verify', 'banking', and 'suspended' create artificial urgency to harvest user credentials.",
            "Verify whether the destination is an authentic authentication portal.",
        )


class SubdomainFloodingRule(BaseRule):
    rule_id = "RULE-DFA-SUB-08"
    rule_name = "Excessive Subdomain Hierarchy (Flooding)"
    category = RuleCategory.AUTOMATA
    severity = RuleSeverity.MEDIUM
    base_weight = 20.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        res = dfa_res.get_result("Subdomain-Flooding-Automaton")
        triggered = res is not None and res.accepted
        evidence = f"Hostname '{parsed.host}' has {parsed.dot_count_host} dots" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Deeply nested subdomains (3+ dots in hostname) are used to push the real root domain out of view on address bars.",
            "Inspect the rightmost registered domain name before trusting any subdomain prefix.",
        )


# --- Structural & Lexical Rules ---

class ExcessiveURLLengthRule(BaseRule):
    rule_id = "RULE-STR-LEN-01"
    rule_name = "Abnormally Long URL Structure"
    category = RuleCategory.STRUCTURAL
    severity = RuleSeverity.MEDIUM
    base_weight = 15.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        triggered = parsed.length_url > 75
        impact = 25.0 if parsed.length_url > 120 else (15.0 if triggered else 0.0)
        evidence = f"Total URL length: {parsed.length_url} chars (baseline < 75)" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category,
            RuleSeverity.HIGH if parsed.length_url > 120 else self.severity,
            triggered, impact, evidence,
            "Long URLs are crafted to push the genuine malicious domain out of view on browser address bars.",
            "Examine the full decoded URL hierarchy.",
        )


class ExcessiveHyphensRule(BaseRule):
    rule_id = "RULE-STR-HYP-02"
    rule_name = "Multiple Hyphens in Hostname"
    category = RuleCategory.STRUCTURAL
    severity = RuleSeverity.MEDIUM
    base_weight = 15.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        count = parsed.hyphen_count_host
        triggered = count >= 2
        impact = 25.0 if count >= 4 else (15.0 if triggered else 0.0)
        evidence = f"Host '{parsed.host}' contains {count} hyphen(s)" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category,
            RuleSeverity.HIGH if count >= 4 else self.severity,
            triggered, impact, evidence,
            "Legitimate domains rarely contain multiple hyphens. Attackers chain hyphens to fabricate legitimate-looking service names.",
            "Do not trust hyphenated service strings; check WHOIS records.",
        )


class HighShannonEntropyRule(BaseRule):
    rule_id = "RULE-STR-ENT-03"
    rule_name = "High Shannon Entropy in Hostname"
    category = RuleCategory.STRUCTURAL
    severity = RuleSeverity.HIGH
    base_weight = 20.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        entropy = parsed.shannon_entropy_host
        triggered = entropy > 4.1 and parsed.length_host > 12
        evidence = f"Host entropy: {entropy:.3f} bits/symbol (Threshold > 4.1)" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "High Shannon entropy indicates pseudo-random character distributions generated by algorithmic malware (DGAs).",
            "Query threat feeds to check if this domain belongs to an active botnet.",
        )


class HighDigitRatioRule(BaseRule):
    rule_id = "RULE-STR-DIG-04"
    rule_name = "High Numeric Digit Ratio in Hostname"
    category = RuleCategory.STRUCTURAL
    severity = RuleSeverity.MEDIUM
    base_weight = 15.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        ratio = parsed.digit_ratio_host
        triggered = ratio > 0.25 and parsed.digit_count_host >= 3
        evidence = f"Host '{parsed.host}' has {parsed.digit_count_host} digits ({ratio*100:.1f}% numeric density)" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Legitimate brand hostnames are mnemonic words. Excessive digits indicate disposable cloud staging or compromised hosts.",
            "Verify whether the host corresponds to a valid enterprise corporate namespace.",
        )


class URLShortenerRule(BaseRule):
    rule_id = "RULE-STR-SHR-05"
    rule_name = "URL Shortener Masking"
    category = RuleCategory.STRUCTURAL
    severity = RuleSeverity.MEDIUM
    base_weight = 20.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        triggered = parsed.is_shortened
        evidence = f"Registered domain '{parsed.registered_domain}' is a recognized URL shortener" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "URL shorteners obscure the ultimate destination, IP reputation, and path parameters prior to resolution.",
            "Expand shortened links via preview tools before navigating.",
        )


class NonStandardPortRule(BaseRule):
    rule_id = "RULE-STR-PRT-06"
    rule_name = "Non-Standard Network Port Binding"
    category = RuleCategory.PROTOCOL
    severity = RuleSeverity.HIGH
    base_weight = 25.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        port = parsed.port
        triggered = port is not None and port not in (80, 443, 8080)
        evidence = f"URL explicitly targets non-standard port :{port}" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Standard consumer web applications operate on ports 80 or 443. High/obscure ports indicate ad-hoc proxy tunnels or compromised hosts.",
            "Investigate whether this port binding is authorized by network infrastructure.",
        )


class HomoglyphAttackRule(BaseRule):
    rule_id = "RULE-LEX-HOMO-01"
    rule_name = "Visual Homoglyph / Script Confusion Spoofing"
    category = RuleCategory.LEXICAL
    severity = RuleSeverity.CRITICAL
    base_weight = 40.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        triggered = parsed.has_homoglyphs
        evidence = ""
        if triggered and parsed.homoglyphs_found:
            details = [f"'{char}' ({name} -> Latin '{latin}')" for char, latin, name in parsed.homoglyphs_found[:3]]
            evidence = f"Detected {len(parsed.homoglyphs_found)} homoglyph character(s): {', '.join(details)}"
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Homoglyph attacks exploit Unicode characters from foreign alphabets (e.g. Cyrillic 'а') that look identical to Latin letters.",
            "Never trust visual domain appearance in emails; inspect the Punycode (xn--) equivalent.",
        )


class PunycodeEncodingRule(BaseRule):
    rule_id = "RULE-LEX-PUNY-02"
    rule_name = "Punycode (xn--) IDN Domain Representation"
    category = RuleCategory.LEXICAL
    severity = RuleSeverity.HIGH
    base_weight = 25.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        triggered = parsed.has_punycode
        evidence = f"Hostname '{parsed.host}' utilizes Punycode ('xn--') encoding" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "Attackers abuse Internationalized Domain Names (IDNs) in Punycode to register lookalike domains of major brands.",
            "Review the decoded Unicode glyphs and confirm the registrar identity.",
        )


class MissingHTTPSRule(BaseRule):
    rule_id = "RULE-PRT-HTTP-03"
    rule_name = "Insecure Plain HTTP Transport on Sensitive Endpoint"
    category = RuleCategory.PROTOCOL
    severity = RuleSeverity.MEDIUM
    base_weight = 15.0

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleResult:
        is_http = parsed.scheme.lower() == "http"
        act_res = dfa_res.get_result("Phishing-Intent-Automaton")
        brand_res = dfa_res.get_result("Brand-Impersonation-Automaton")
        has_sensitive = bool((act_res and act_res.accepted) or (brand_res and brand_res.accepted))
        triggered = is_http and has_sensitive
        evidence = "Plain HTTP scheme used in conjunction with credential/brand endpoint" if triggered else ""
        return RuleResult(
            self.rule_id, self.rule_name, self.category, self.severity, triggered,
            self.base_weight if triggered else 0.0, evidence,
            "All legitimate authentication gateways enforce HTTPS/TLS encryption. Unencrypted HTTP login endpoints are either fraudulent or insecure.",
            "Do not submit credentials over unencrypted HTTP channels.",
        )


@dataclass
class RuleEvaluationReport:
    """Consolidated report of all rule evaluations."""
    all_results: List[RuleResult] = field(default_factory=list)
    triggered_rules: List[RuleResult] = field(default_factory=list)
    total_rule_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0


class RuleEngine:
    """Evaluates all 17 cybersecurity detection rules."""

    def __init__(self) -> None:
        self.rules: List[BaseRule] = [
            IPHostRule(),
            DangerousSchemeRule(),
            AbusiveTLDRule(),
            CredentialStuffingAtRule(),
            PercentEncodingObfuscationRule(),
            BrandImpersonationRule(),
            PhishingIntentRule(),
            SubdomainFloodingRule(),
            ExcessiveURLLengthRule(),
            ExcessiveHyphensRule(),
            HighShannonEntropyRule(),
            HighDigitRatioRule(),
            URLShortenerRule(),
            NonStandardPortRule(),
            HomoglyphAttackRule(),
            PunycodeEncodingRule(),
            MissingHTTPSRule(),
        ]

    def evaluate(self, parsed: ParsedURL, dfa_res: SuiteAnalysisResult) -> RuleEvaluationReport:
        all_res: List[RuleResult] = []
        triggered: List[RuleResult] = []
        total_score = 0.0
        cat_scores: Dict[str, float] = {cat.value: 0.0 for cat in RuleCategory}

        critical_c = 0
        high_c = 0
        med_c = 0
        low_c = 0

        for rule in self.rules:
            res = rule.evaluate(parsed, dfa_res)
            all_res.append(res)
            if res.triggered:
                triggered.append(res)
                total_score += res.score_impact
                cat_scores[res.category.value] = cat_scores.get(res.category.value, 0.0) + res.score_impact
                if res.severity == RuleSeverity.CRITICAL:
                    critical_c += 1
                elif res.severity == RuleSeverity.HIGH:
                    high_c += 1
                elif res.severity == RuleSeverity.MEDIUM:
                    med_c += 1
                elif res.severity == RuleSeverity.LOW:
                    low_c += 1

        return RuleEvaluationReport(
            all_results=all_res,
            triggered_rules=triggered,
            total_rule_score=total_score,
            category_scores=cat_scores,
            critical_count=critical_c,
            high_count=high_c,
            medium_count=med_c,
            low_count=low_c,
        )


class ThreatLevel(str, Enum):
    BENIGN = "BENIGN / SAFE"
    LOW = "LOW SUSPICION"
    SUSPICIOUS = "SUSPICIOUS / ELEVATED"
    MALICIOUS = "HIGH RISK / MALICIOUS"


@dataclass
class ScoreFactor:
    name: str
    impact: float
    category: str
    reason: str
    is_mitigation: bool = False


@dataclass
class PhishGuardAssessment:
    """Composite explainable threat assessment."""
    url: str
    risk_score: float
    threat_level: ThreatLevel
    confidence: float
    status_color: str
    status_icon: str
    executive_summary: str
    primary_threat_drivers: List[str] = field(default_factory=list)
    contributing_factors: List[ScoreFactor] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    category_breakdown: Dict[str, float] = field(default_factory=dict)


REPUTABLE_ROOT_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "github.com",
    "wikipedia.org", "cloudflare.com", "paypal.com", "linkedin.com", "youtube.com",
}


class RiskScorer:
    """Calculates normalized, explainable composite threat score (0-100)."""

    @classmethod
    def calculate_assessment(
        cls,
        parsed: ParsedURL,
        dfa_res: SuiteAnalysisResult,
        rule_report: RuleEvaluationReport,
    ) -> PhishGuardAssessment:
        raw_score = 0.0
        factors: List[ScoreFactor] = []

        for r in rule_report.triggered_rules:
            raw_score += r.score_impact
            factors.append(
                ScoreFactor(
                    name=r.rule_name,
                    impact=r.score_impact,
                    category=r.category.value,
                    reason=r.evidence or r.explanation,
                    is_mitigation=False,
                )
            )

        mitigation_points = 0.0
        is_reputable = parsed.registered_domain.lower() in REPUTABLE_ROOT_DOMAINS
        no_critical = rule_report.critical_count == 0

        if is_reputable and no_critical and not parsed.has_homoglyphs and not parsed.has_at_symbol:
            if parsed.dot_count_host <= 2:
                credit = 25.0
                mitigation_points += credit
                factors.append(
                    ScoreFactor(
                        name="Reputable Domain Standing",
                        impact=-credit,
                        category="Reputation Baseline",
                        reason=f"Domain '{parsed.registered_domain}' is an authoritative primary namespace with clean structure.",
                        is_mitigation=True,
                    )
                )

        if parsed.scheme.lower() == "https" and no_critical:
            credit = 5.0
            mitigation_points += credit
            factors.append(
                ScoreFactor(
                    name="Standard TLS Encryption (HTTPS)",
                    impact=-credit,
                    category="Transport Security",
                    reason="Standard HTTPS transport present without protocol spoofing.",
                    is_mitigation=True,
                )
            )

        if rule_report.critical_count >= 1 and rule_report.high_count >= 1:
            penalty = 15.0
            raw_score += penalty
            factors.append(
                ScoreFactor(
                    name="Compound Attack Vector Penalty",
                    impact=penalty,
                    category="Threat Correlation",
                    reason="Simultaneous occurrence of critical protocol abuse and high-risk automata detection patterns.",
                    is_mitigation=False,
                )
            )

        final_score = max(0.0, min(100.0, raw_score - mitigation_points))

        if final_score <= 25.0:
            threat_level = ThreatLevel.BENIGN
            status_color = "#3fb950"
            status_icon = "✅"
        elif final_score <= 50.0:
            threat_level = ThreatLevel.LOW
            status_color = "#d29922"
            status_icon = "⚠️"
        elif final_score <= 75.0:
            threat_level = ThreatLevel.SUSPICIOUS
            status_color = "#d29922"
            status_icon = "⚠️"
        else:
            threat_level = ThreatLevel.MALICIOUS
            status_color = "#f85149"
            status_icon = "🛑"

        confidence = min(0.99, max(0.60, 0.60 + (len(rule_report.triggered_rules) * 0.08) + (len(dfa_res.all_matches) * 0.05)))

        pos_factors = [f for f in factors if not f.is_mitigation]
        pos_factors.sort(key=lambda x: x.impact, reverse=True)
        primary_drivers = [f"{f.name} (+{f.impact:.0f} pts)" for f in pos_factors[:3]]

        if threat_level == ThreatLevel.MALICIOUS:
            summary = (
                f"HIGH-CONFIDENCE PHISHING DETECTED: The target URL exhibited severe adversarial characteristics, "
                f"accumulating a risk score of {final_score:.1f}/100. "
                f"Automata analysis identified {len(rule_report.triggered_rules)} active threat indicators, "
                f"dominated by {pos_factors[0].name if pos_factors else 'structural anomalies'}."
            )
        elif threat_level == ThreatLevel.SUSPICIOUS:
            summary = (
                f"ELEVATED RISK DETECTED: The target URL exhibits {len(rule_report.triggered_rules)} suspicious traits "
                f"with an overall risk score of {final_score:.1f}/100. Proceed with elevated caution."
            )
        elif threat_level == ThreatLevel.LOW:
            summary = f"LOW SUSPICION: Minor non-standard structural elements observed (Score: {final_score:.1f}/100)."
        else:
            summary = f"BENIGN / SAFE: URL demonstrates standard architectural conformity and clean lexical structure (Score: {final_score:.1f}/100)."

        remediation = [
            "Block traffic to this URL at enterprise DNS firewalls and web proxies.",
            "Do not enter credentials, session cookies, or MFA tokens on this host.",
        ] if threat_level in (ThreatLevel.MALICIOUS, ThreatLevel.SUSPICIOUS) else [
            "URL passed static automata heuristics. Standard browsing vigilance remains recommended.",
        ]

        return PhishGuardAssessment(
            url=parsed.raw_url,
            risk_score=final_score,
            threat_level=threat_level,
            confidence=confidence,
            status_color=status_color,
            status_icon=status_icon,
            executive_summary=summary,
            primary_threat_drivers=primary_drivers,
            contributing_factors=factors,
            remediation_steps=remediation,
            category_breakdown=rule_report.category_scores,
        )


# =============================================================================
# 6. EDUCATIONAL BENCHMARK DATASETS (For Viva Demonstration)
# =============================================================================

SAMPLE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "🎯 Targeted Brand Phishing": [
        {
            "name": "Subdomain Brand Stacking (PayPal)",
            "url": "http://paypal.com.account-verification.login-portal.top/cgi-bin/auth",
            "category": "Subdomain Stacking & Abusive TLD",
            "description": "Places reputable brand 'paypal.com' into deeply nested subdomains under a .top abusive TLD.",
        },
        {
            "name": "Apple ID Credential Harvest",
            "url": "http://appleid.apple.com.verify-device-security.xyz/update",
            "category": "Brand Impersonation",
            "description": "Impersonates Apple ID verification with hyphenated action words on .xyz.",
        },
        {
            "name": "Microsoft 365 Password Reset",
            "url": "http://login.microsoftonline.com-auth-token-session.club/login.srf",
            "category": "Typosquatting & Action Keywords",
            "description": "Fakes Microsoft Online login via brand-stuffing with action keywords.",
        },
        {
            "name": "Binance Crypto Wallet Drainer",
            "url": "http://binance-wallet-recovery-verification.buzz/secure",
            "category": "Crypto Phishing",
            "description": "Cryptocurrency wallet recovery lure on .buzz registrar.",
        },
    ],
    "🌐 Raw IP & Port Attacks": [
        {
            "name": "Raw Dotted IPv4 with Custom Port",
            "url": "http://185.220.101.5:8080/banking/chase/login.php",
            "category": "IP-in-URL",
            "description": "Direct routable IPv4 host avoiding DNS registration, running on non-standard port 8080.",
        },
        {
            "name": "Hexadecimal Encoded IP",
            "url": "http://0x7f000001/account/update-billing",
            "category": "Hex IP Obfuscation",
            "description": "Uses 32-bit hexadecimal IP notation (0x7f000001 = 127.0.0.1) to evade simple IP string checks.",
        },
        {
            "name": "Private RFC 1918 Staging Address",
            "url": "http://192.168.1.105:1337/payload.exe",
            "category": "Port & Binary Payload",
            "description": "Local subnet host targeting non-standard high port 1337 with binary file extension.",
        },
    ],
    "🎭 Deceptive Authority & Obfuscation": [
        {
            "name": "HTTP Basic Auth '@' Symbol Spoofing",
            "url": "https://paypal.com:security-update@attacker-controlled-host.com/login",
            "category": "Authority Boundary Confusion",
            "description": "Exploits '@' delimiter where browsers treat 'paypal.com' as user credentials and route to 'attacker-controlled-host.com'.",
        },
        {
            "name": "Percent-Encoded Critical Delimiters (%2e, %2f)",
            "url": "http://secure%2epaypal%2ecom%2flogin.html@evil-phish.net/",
            "category": "Hex Delimiter Encoding",
            "description": "Conceals dot (%2e) and slash (%2f) to bypass naive perimeter tokenizers.",
        },
        {
            "name": "URL Shortener Masking",
            "url": "https://bit.ly/3xSecUr3L0g1n",
            "category": "Link Shortener",
            "description": "Masks ultimate target domain through a public redirect service.",
        },
    ],
    "🔤 Homoglyph & Unicode Deception": [
        {
            "name": "Cyrillic Homoglyph Attack (pаypal)",
            "url": "http://p\u0430yp\u0430l.com/verify-identity",
            "category": "Cyrillic Homoglyphs",
            "description": "Substitutes Latin 'a' with Cyrillic 'а' (U+0430). Looks visually identical to genuine paypal.com.",
        },
        {
            "name": "Punycode (xn--) Internationalized Domain",
            "url": "http://xn--pypal-4ve.com/account/security",
            "category": "Punycode IDN",
            "description": "Punycode representation of an IDN lookalike domain.",
        },
    ],
    "⚡ Malicious Protocols": [
        {
            "name": "Inline JavaScript Execution Scheme",
            "url": "javascript:alert('Credentials Harvested');",
            "category": "Script Injection Scheme",
            "description": "Executes malicious client-side JavaScript without making standard network requests.",
        },
    ],
    "✅ Legitimate Benign Baselines": [
        {
            "name": "Wikipedia - Formal DFA Article",
            "url": "https://en.wikipedia.org/wiki/Deterministic_finite_automaton",
            "category": "Educational Reference",
            "description": "Standard legitimate HTTPS Wikipedia article with deep paths and clean domain structure.",
        },
        {
            "name": "Official PayPal Sign In Portal",
            "url": "https://www.paypal.com/signin",
            "category": "Financial Portal",
            "description": "Legitimate PayPal authentication portal with proper EV SSL and clean host structure.",
        },
        {
            "name": "GitHub Open Source Repository",
            "url": "https://github.com/torvalds/linux",
            "category": "Developer Service",
            "description": "Authoritative developer repository URL with clean ASCII characters.",
        },
    ],
}
