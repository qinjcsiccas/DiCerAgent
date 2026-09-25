"""
CADO (Context-Aware Dialogue Orchestrator) Module
Two-stage RAG pipeline
"""

import streamlit as st
from streamlit.errors import StreamlitAPIException
import time
from openai import OpenAI
import re
import config
from literature import LiteratureAgent
from plugins.base import get_property_info


def create_pdf_export(content_lines, title="DiCerAgent Chat"):
    """Generate a PDF file"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=10
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"], fontSize=9, leading=12
    )

    story = []
    for line in content_lines:
        line = line.strip()
        line = line.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        import re

        line = re.sub(r"<[^>]+>", "", line)
        if not line:
            story.append(Spacer(1, 4))
        elif "=" in line and len(line) < 50:
            story.append(Paragraph(line.replace("=", "").strip(), title_style))
        elif line.startswith("---"):
            continue
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def init_chat_state():
    """Initialize the chat state"""
    if (
        not hasattr(st.session_state, "chat_history")
        or st.session_state.chat_history is None
    ):
        st.session_state.chat_history = []


def detect_language(text):
    """Detect the text language"""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    english_chars = sum(1 for c in text if c.isalpha() and c.isascii())
    return "zh" if chinese_chars > english_chars else "en"


@st.cache_resource(show_spinner=False)
def _load_inverse_metadata():
    """Inverse-design degraded metadata: only reads the CSV and precomputes the formula-normalized columns; does not load faiss/MACE/feature_dict"""
    import pandas as pd
    from pymatgen.core import Composition

    df = pd.read_csv(config.PATH_MATERIAL_PROPERTIES, encoding="utf-8-sig")
    if "material_id" not in df.columns:
        df["material_id"] = df.index.astype(str)

    def _reduce(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:
            return str(f)

    def _elems(f):
        try:
            return "|".join(sorted(Composition(str(f)).get_el_amt_dict().keys()))
        except Exception:
            return ""

    df["_reduced_formula"] = df["formula"].apply(_reduce)
    df["_elem_str"] = df["formula"].apply(_elems)
    for col in ["pred_tm", "pred_er", "pred_qf", "pred_tcf", "spacegroup", "cif_path"]:
        if col not in df.columns:
            df[col] = None
    return df


def _inherited_formula(chat_history):
    """Multi-turn tool memory: inherit the formula in the tool_results of the previous assistant message"""
    if not chat_history:
        return ""
    for msg in reversed(chat_history):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        if msg.get("role") != "assistant":
            continue
        tr = msg.get("tool_results")
        if isinstance(tr, dict) and tr.get("formula"):
            return str(tr["formula"])
    return ""


    # Mechanism/principle-type question detection: such questions ask for the physics and defect mechanisms behind "why/how",
    # so they should go through free-text Q&A (RAG) and must NOT be treated as attribute extraction of the current material (which would output an irrelevant property table).
_MECHANISM_PATTERN = re.compile(
    r"机理|机制|原理|为什么|为何|原因|成因|起源|缺陷|空位|氧空位|补偿|解耦|权衡|主导|"
    r"如何(调控|解耦|影响|决定|主导|作用)|作用机理|"
    r"mechanis|mechanistic|princip|decoupl|defect|vacanc|charge compensat|trade-?off|"
    r"govern|underlying|origin of|\bwhy\b",
    re.IGNORECASE,
)


def _is_mechanism_question(text):
    """Judge whether the question is mechanism/principle-type (explaining causes, defect chemistry, performance decoupling and trade-offs, etc.)."""
    return bool(_MECHANISM_PATTERN.search(text or ""))


def intent_parse(prompt, current_data=None, chat_history=None):
    """LLM structured intent classification: predict / knowledge / inverse / quiz"""
    inherited_formula = _inherited_formula(chat_history)
    current_formula = (current_data or {}).get("Formula", "") if current_data else ""

    context_parts = []
    if current_formula:
        context_parts.append(f"Current material in context: {current_formula}")
    if inherited_formula:
        context_parts.append(f"Inherited material from previous turn: {inherited_formula}")
    if (current_data or {}).get("Structure_Type"):
        context_parts.append(f"Structure: {(current_data or {}).get('Structure_Type')}")

    system_prompt = (
        "You are an intent classifier for the CADO (Context-Aware Dialogue Orchestrator) assistant.\n"
        "Classify the user's question into exactly one of these intents:\n"
        "- predict: user asks about / asks to predict the microwave dielectric properties of a material formula (er/qxf/tcf/tm). This includes phrasing like \"how are the dielectric properties of X\", \"what is the er/qf/tcf of X\", \"predict X's performance\". A formula + a dielectric-performance question is predict EVEN IF it does not contain the word 'predict'.\n"
        "- knowledge: user asks about synthesis routes, processing guidance, or literature knowledge about a material (e.g. how to synthesize/prepare/process it, phase formation, doping effects). Such process/synthesis questions are knowledge, NOT predict.\n"
        "Mechanism/principle questions that ask WHY or HOW a physical mechanism works are NEITHER predict NOR knowledge. This includes questions about how substitutions/doping decouple or trade off one property against another, defect chemistry, oxygen vacancies, charge compensation, structure-property origins, and similar why/how mechanism questions. Classify them as quiz, so they are answered as free-text scientific Q&A about the mechanism, without extracting or tabulating the properties of the material currently loaded in context.\n"
        "Decision rule: when a chemical formula is present and the user asks about its dielectric/microwave performance (Chinese 介电/微波性能/性能如何 or English dielectric/microwave/performance/how), choose predict. Choose knowledge only for process/synthesis questions. When in doubt for a dielectric-property question, choose predict. When in doubt for a why/how mechanism question, choose quiz.\n"
        "- inverse: user asks to discover/design/recommend candidate materials achieving target properties or similar/better than a given material.\n"
        "- quiz: general question, follow-up, greeting, asking to explain already-shown results, or any why/how mechanism, principle or defect-chemistry question.\n"
        "Return ONLY valid JSON (no markdown) with keys:\n"
        '{"intent": "predict|knowledge|inverse|quiz", "formula": "chemical formula if any and relevant, else empty string; for why/how mechanism or principle questions leave it empty unless the formula is explicitly written in this question", '
        '"target_props": ["er","qxf","tcf","tm"] if predict/inverse otherwise empty, '
        '"keywords": ["search keywords"] if knowledge otherwise empty}'
    )
    user_prompt = (
        f"Context:\n{chr(10).join(context_parts) if context_parts else 'No material context'}\n\n"
        f"Question: {prompt}"
    )
    try:
        import json

        client = OpenAI(api_key=config.DS_API_KEY, base_url=config.DEEPSEEK_API_URL)
        response = client.chat.completions.create(
            model=config.DS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        elif "```" in text:
            text = text.replace("```", "")
        parsed = json.loads(text.strip())
        intent = str(parsed.get("intent", "quiz")).strip().lower()
        if intent not in ("predict", "knowledge", "inverse", "quiz"):
            intent = "quiz"
        formula = str(parsed.get("formula") or "").strip()
        if not formula and not _is_mechanism_question(prompt):
            formula = inherited_formula or current_formula
        target_props = parsed.get("target_props") or []
        if not isinstance(target_props, list):
            target_props = []
        keywords = parsed.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        return {
            "intent": intent,
            "formula": formula,
            "target_props": target_props,
            "keywords": keywords,
        }
    except Exception:
        return {
            "intent": "quiz",
            "formula": "" if _is_mechanism_question(prompt) else (inherited_formula or current_formula),
            "target_props": [],
            "keywords": [],
        }


def _extract_target_values(prompt):
    """Extract inverse-design target values from the prompt.

    Supports the symbols = : ≈ ~ > < ≥ ≤ and Chinese comparative phrases (below/less than/not greater than/not exceeding,
    greater than/higher than/exceeding, about/close to/approximately, near-zero/close to zero/tending to zero, etc.).
    Returns {key: (value, direction)}, direction ∈ {'<','<=','>','>=','≈','='},
    for natural expressions such as "εr below 10", "Qf > 20000", "TCF near zero", "εr≈10", "Q×f>20000", etc.
    """
    import re

    alias = {
        "tm":  ["tm", "melting", "熔点"],
        "er":  ["er", "epsr", "εr", "介电常数"],
        "qf":  ["qf", "qxf", "q×f", "Q·f", "品质因数"],
        "tcf": ["tcf", "τf", "tauf", "温度系数"],
    }
    # Direction words -> semantics
    cmp_words = [
        ("<=", ["不高于", "至多", "小于等于", "不多于", "≤"]),
        (">=", ["不低于", "至少", "大于等于", "不少于", "≥"]),
        (">",  ["大于", "高于", "超过", "以上", "不小于", ">"]),
        ("<",  ["低于", "小于", "不大于", "不超过", "以下", "<"]),
        ("≈",  ["约", "大约", "接近", "近似", "近于", "≈", "~"]),
    ]

    targets = {}
    for k in alias:
        body = "|".join(alias[k])
        keep = "值|系数|常数"
        # (a) Separator/direction-symbol form: KEY <sep> NUM (incl. > < >= <=
        m = re.search(
            rf"(?<![A-Za-z0-9])(?:{body})\s*((?:>=|<=|>|<|=|≈|~|≤|≥))\s*(-?[\d.]+)",
            prompt, re.IGNORECASE,
        )
        if m:
            sep, num = m.group(1), m.group(2)
            try:
                val = float(num)
            except ValueError:
                val = float("nan")
            if val == val:
                if sep in ("<=", "≤"):
                    targets[k] = (val, "<=")
                elif sep in (">=", "≥"):
                    targets[k] = (val, ">=")
                elif sep in ("≈", "~"):
                    targets[k] = (val, "≈")
                elif sep == "<":
                    targets[k] = (val, "<")
                elif sep == ">":
                    targets[k] = (val, ">")
                else:
                    targets[k] = (val, "=")
            continue
        # (b) KEY [value/coefficient/constant] direction-word NUM (Chinese comparatives)
        for cmp_, ws in cmp_words:
            w = "|".join(ws)
            m = re.search(
                rf"(?<![A-Za-z0-9])(?:{body})\s*(?:{keep})?\s*(?:{w})\s*(-?[\d.]+)",
                prompt, re.IGNORECASE,
            )
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    val = float("nan")
                if val == val:
                    targets[k] = (val, cmp_)
                break
        # (c) Fallback: KEY [value] [is/equals] NUM (no direction word; treated as a reference value)
        if k not in targets:
            m = re.search(
                rf"(?<![A-Za-z0-9])(?:{body})\s*(?:{keep})?\s*(?:为|是)?\s*(-?[\d.]+)",
                prompt, re.IGNORECASE,
            )
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    val = float("nan")
                if val == val:
                    targets[k] = (val, "=")
        # (d) Near-zero expressions: KEY near-zero / close to zero / tending to zero / ~=0 / is 0
        if k not in targets:
            if re.search(
                rf"(?<![A-Za-z0-9])(?:{body})\s*(?:近零|接近于零|趋于零|接近零|为\s*0|≈\s*0)",
                prompt, re.IGNORECASE,
            ):
                targets[k] = (0.0, "≈")
    return targets


class ChatToolRouter:
    """Tool-routing layer: call predict / knowledge / inverse-design modules by intent, and perform three-layer disclosure recombination"""

    def __init__(self, agent, nlp_processor=None, csv_engine=None, rag_engine=None,
                 extractor=None, scholar=None):
        self.agent = agent
        self.nlp_processor = nlp_processor
        self._csv_engine = csv_engine
        self._rag_engine = rag_engine
        self._extractor = extractor
        self._scholar = scholar

    # ---------- Sub-component access (prefer passing references; otherwise fetch from agent) ----------
    def _get_scholar(self):
        if self._scholar is not None:
            return self._scholar
        return getattr(self.agent, "scholar", None) if self.agent else None

    def _get_extractor(self):
        if self._extractor is not None:
            return self._extractor
        return getattr(self.agent, "extractor", None) if self.agent else None

    def _get_rag_engine(self):
        if self._rag_engine is not None:
            return self._rag_engine
        return getattr(self.agent, "rag_engine", None) if self.agent else None

    def _get_csv_engine(self):
        if self._csv_engine is not None:
            return self._csv_engine
        return getattr(self.agent, "csv_engine", None) if self.agent else None

    # ---------- Retrieval and formatting ----------
    def _retrieve_literature(self, formula, keywords=None, limit=4):
        """Enhanced hybrid literature retrieval: aligned with the full Property Prediction page pipeline
    aliases formula normalization + CSV/RAG high recall (50) + OpenAlex online + title-fingerprint merge/dedup + score sorting"""
        keywords = keywords or []
        import os

        # 1) Formula aliases normalization (fall back to the original query when unavailable)
        aliases = []
        ext = self._get_extractor()
        if ext is not None:
            try:
                aliases = ext.brainstorm_keywords(formula or " ".join(keywords) or "") or []
            except Exception:
                aliases = []
        if not aliases:
            base = (f"{formula} {' '.join(keywords[:2])}".strip()
                    if formula and keywords else (formula or " ".join(keywords) or ""))
            aliases = [base] if base else []

        # 2) Merge and deduplicate the multi-channel results by title fingerprint
        merged = {}
        csv_eng = self._get_csv_engine()

        def _fingerprint(title):
            if csv_eng is not None:
                try:
                    return csv_eng._get_fingerprint(title or "")
                except Exception:
                    pass
            t = (title or "").lower()
            return " ".join(t.split()) or None

        def _push(doc):
            if not isinstance(doc, dict):
                return
            title = doc.get("title", "")
            key = _fingerprint(title) or doc.get("doi") or doc.get("url") or title or ""
            if not key:
                return
            if key in merged:
                prev = merged[key]
                prev["content"] = doc.get("content") or prev.get("content")
                prev["has_fulltext"] = True
                s1 = prev.get("score") or 0
                s2 = doc.get("score") or 0
                prev["score"] = max(s1, s2) * 1.3 if (s1 and s2) else (s1 or s2)
                for f in ("doi", "url", "journal", "year"):
                    if f not in prev and doc.get(f):
                        prev[f] = doc.get(f)
            else:
                doc.setdefault("score", 0)
                doc.setdefault("has_fulltext", False)
                merged[key] = doc

        q = formula or " ".join(keywords or [])

        # Channel 1: local CSV (aliases + high recall 50)
        if csv_eng is not None and q:
            try:
                for doc in csv_eng.search(q, aliases=aliases, limit=50) or []:
                    _push(doc)
            except Exception:
                pass

        # Channel 2: local RAG (aliases + high recall 50; meta completed with CSV metadata)
        rag = self._get_rag_engine()
        if rag is not None and q:
            try:
                for doc in rag.search(q, aliases=aliases, k=50) or []:
                    if csv_eng is not None:
                        try:
                            fraw = doc.get("filename_raw", "")
                            fname = os.path.basename(str(fraw)) if fraw else ""
                            meta = csv_eng.match_metadata(fname, rag_title=doc.get("title"))
                            if meta is not None:
                                doc["title"] = str(meta.get("Title", doc.get("title", ""))).strip()
                                authors = meta.get("Authors", doc.get("authors", ""))
                                doc["authors"] = ([a.strip() for a in str(authors).split(";") if a.strip()]
                                                  if ";" in str(authors) else authors)
                                journal = meta.get("Source Title",
                                                    meta.get("Journal Abbreviation", "Local Repo"))
                                if journal:
                                    doc["journal"] = str(journal)
                                try:
                                    y = int(float(str(meta.get("Publication Year", "0")).replace("nan", "0")))
                                    if y:
                                        doc["year"] = y
                                except Exception:
                                    pass
                                doi_val = str(meta.get("DOI", meta.get("doi", ""))).strip()
                                if doi_val.lower() != "nan" and doi_val:
                                    doc["doi"] = doi_val
                            if fname:
                                doc["filename_raw"] = fname
                        except Exception:
                            pass
                    _push(doc)
            except Exception:
                pass

        # Channel 3: OpenAlex online retrieval (aligned with the Property Prediction page: search_literature + fulltext enhancement)
        scholar = self._get_scholar()
        if scholar is not None:
            try:
                if hasattr(scholar, "search_literature") and hasattr(
                    scholar, "process_papers_to_dicts"
                ):
                    raw_papers = scholar.search_literature(q, limit=10) or []
                    if raw_papers:
                        try:
                            scholar.enrich_papers_with_fulltext(raw_papers, max_fulltext=10)
                        except Exception:
                            pass
                        for doc in scholar.process_papers_to_dicts(raw_papers) or []:
                            if csv_eng is not None and doc.get("title"):
                                try:
                                    doc["score"] = csv_eng.scorer.evaluate(
                                        doc.get("title", ""), q, aliases=aliases
                                    )
                                except Exception:
                                    pass
                            _push(doc)
                else:
                    for kw in keywords[:2] or [""]:
                        sq = (f"{formula} {kw}".strip() if formula else str(kw)).strip()
                        if sq:
                            for doc in scholar.online_search(sq, limit=limit) or []:
                                if csv_eng is not None and doc.get("title"):
                                    try:
                                        doc["score"] = csv_eng.scorer.evaluate(
                                            doc.get("title", ""), q, aliases=aliases
                                        )
                                    except Exception:
                                        pass
                                _push(doc)
            except Exception:
                pass

        # 3) Sort by score (entries without score go last, for compatibility), keeping the original structure (refs list)
        ranked = sorted(merged.values(), key=lambda d: d.get("score") or 0, reverse=True)
        return ranked[:20]

    def _build_literature_context(self, refs, limit=5):
        lines = []
        for i, r in enumerate(refs[:limit], start=1):
            title = r.get("title", "Untitled")
            content = (r.get("content") or "")[:1500]
            lines.append(f"[{i}] {title}\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_props_panel(data):
        lines = ["### Performance Panel"]
        if not data:
            lines.append("_No prediction data._")
            return "\n".join(lines)
        if data.get("Formula"):
            lines.append(f"- **Formula**: {data.get('Formula')}")
        if data.get("Material_ID"):
            lines.append(f"- **Material ID**: {data.get('Material_ID')}")
        if data.get("Structure_Type"):
            lines.append(f"- **Structure**: {data.get('Structure_Type')}")

        for k, v in data.items():
            if k.endswith("_mean") and v is not None and not k.startswith("_"):
                prop = k[:-5]
                info = get_property_info(prop)
                symbol = info.symbol if info.symbol else prop
                unit = info.unit if info.unit else ""
                dev = data.get(k.replace("_mean", "_dev"), 0) or 0
                if unit:
                    lines.append(f"- {symbol}: {v:.2f} ± {dev:.2f} {unit}")
                else:
                    lines.append(f"- {symbol}: {v:.2f} ± {dev:.2f}")

        enhanced = data.get("enhanced_results")
        if enhanced:
            seen = {}
            for item in enhanced:
                if isinstance(item, dict) and item.get("method"):
                    seen[(item.get("method"), item.get("target"))] = item
            if seen:
                lines.append("- **(Enhanced routes)**")
                for (method, target), item in seen.items():
                    val = item.get("value")
                    if val is None:
                        continue
                    unit = item.get("unit", "")
                    if isinstance(val, (int, float)):
                        lines.append(f"  - {method} / {target}: {val:.2f} {unit}")
                    else:
                        lines.append(f"  - {method} / {target}: {val} {unit}")

        if data.get("Error"):
            lines.append(f"- ⚠ Error: {data.get('Error')}")
        return "\n".join(lines)

    @staticmethod
    def _format_refs(refs, limit=6):
        lines = ["### References"]
        if not refs:
            lines.append("_No references retrieved._")
            return "\n".join(lines)
        for i, r in enumerate(refs[:limit], start=1):
            title = r.get("title", "Untitled")
            authors = r.get("authors", [])
            if isinstance(authors, list):
                valid = [str(a) for a in authors if a and str(a).strip()]
                a_str = ", ".join(valid[:3]) if valid else "Unknown Author"
            else:
                a_str = str(authors) if authors else "Unknown Author"
            year = r.get("year", "n.d.")
            journal = r.get("journal", "Unknown Source")
            url = r.get("url") or r.get("doi") or ""
            lines.append(f'[{i}] {a_str}. "{title}" *{journal}*, {year}. {url}')
        return "\n".join(lines)

    @staticmethod
    def _fmt_prop_values(pv):
        """Render the extracted real literature data points as a Markdown table grouped by property"""
        # Extract keys -> display property names (keeping engineering symbols εr / Q×f / τf)
        _PROP_DISPLAY = {
            "er": "εr",
            "qxf": "Q×f (GHz)",
            "tcf": "τf (ppm/°C)",
        }
        pv = pv or {}
        labels = _PROP_DISPLAY
        group_lines = []
        for prop, entries in pv.items():
            if not entries:
                continue
            title = labels.get(prop, prop)
            rows = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                val = e.get("value", "")
                if not str(val).strip():
                    continue
                cond = e.get("condition") or "—"
                src = e.get("source") or "—"
                esc = lambda s: str(s).replace("|", "\\|")
                rows.append(f"| {esc(val)} | {esc(cond)} | {esc(src)} |")
            if rows:
                group_lines.append(f"**{title}**")
                group_lines.append("| Value | Condition | Source |")
                group_lines.append("|---|---|---|")
                group_lines.extend(rows)
        return "\n".join(group_lines) if group_lines else "None extracted"

    @staticmethod
    def _fmt_known_datapoints(pv):
        """Format the extracted real literature data points as a prompt block for the report context (only entries with an [n] literature source)"""
        lines = []
        for prop, entries in (pv or {}).items():
            for e in entries or []:
                if not isinstance(e, dict):
                    continue
                val = e.get("value", "")
                src = str(e.get("source", "") or "").strip()
                if not val or not src:
                    continue
                cond = e.get("condition", "")
                lines.append(f"- {prop}: {val} (source {src}{f', {cond}' if cond else ''})")
        return "\n".join(lines) if lines else ""

    # ---------- Three tool branches ----------
    def run_predict(self, formula, targets=None, manual_feats=None, user_context=None):
        """Forward prediction: analyze_single (runs only the ML + LLM-ML-Full convergence set) + three-layer disclosure recombination"""
        if self.agent is None:
            raise RuntimeError("Prediction engine (agent) is not loaded.")
        if not formula:
            raise ValueError("No chemical formula to predict.")
        enabled = ["ML", "LLM-ML-Full"]
        data = self.agent.analyze_single(
            formula,
            manual_feats=manual_feats or {},
            user_context=user_context,
            enabled_methods=enabled,
        )
        panel = self._format_props_panel(data)

        formula_show = data.get("Formula") or formula
        refs = self._retrieve_literature(formula_show, targets or [])
        guide = ""
        lit_values = {}
        ext = self._get_extractor()
        if ext is not None:
            lit_ctx = self._build_literature_context(refs, 20)
            # Extract real literature data points only from retrieved references; never mix model-completed values into "Extracted literature values"
            try:
                lit_values = ext.extract_property_values(formula_show, lit_ctx) or {}
            except Exception:
                lit_values = {}
            # Inject the extracted real literature data points into the report context so the LLM cites [n] first instead of regenerating from model knowledge
            guide_ctx = lit_ctx
            known_dp = self._fmt_known_datapoints(lit_values)
            if known_dp:
                guide_ctx = (
                    "KNOWN literature datapoints already extracted from the retrieved "
                    "references (authoritative; cite them as [n], do NOT relabel them "
                    "as model knowledge):\n" + known_dp + "\n\n" + lit_ctx
                )
            try:
                guide = ext.generate_report(formula_show, guide_ctx, style="focused")
            except Exception:
                guide = ""

        if lit_values:
            panel = (
                panel
                + "\n\n### Extracted literature values\n"
                + self._fmt_prop_values(lit_values)
            )

        guide_layer = f"### Synthesis & Interpretation\n\n{guide or '_Knowledge guide unavailable._'}"

        safe_data = {k: v for k, v in data.items() if k != "_structure_obj"}
        return {
            "formula": formula_show,
            "panel": panel,
            "guide": guide_layer,
            "refs_layer": self._format_refs(refs),
            "refs": refs,
            "lit_values": lit_values,
            "tool_results": {
                "formula": formula_show,
                "analyzed_data": safe_data,
                "references": refs,
                "extracted_literature_values": lit_values,
            },
        }

    def run_knowledge(self, formula, keywords=None, user_context=None):
        """Knowledge retrieval: online Literature + local RAG/CSV + extractor interpretation"""
        keywords = keywords or []
        refs = self._retrieve_literature(formula, keywords)
        lit_ctx = self._build_literature_context(refs, 20)

        prop_values = {}
        guide = ""
        ext = self._get_extractor()
        if ext is not None and formula:
            try:
                prop_values = ext.extract_property_values(formula, lit_ctx)
            except Exception:
                prop_values = {}
            guide_ctx = lit_ctx
            known_dp = self._fmt_known_datapoints(prop_values)
            if known_dp:
                guide_ctx = (
                    "KNOWN literature datapoints already extracted from the retrieved "
                    "references (authoritative; cite them as [n], do NOT relabel them "
                    "as model knowledge):\n" + known_dp + "\n\n" + lit_ctx
                )
            try:
                guide = ext.generate_report(formula, guide_ctx, style="focused")
            except Exception:
                guide = ""

        panel_lines = ["### Knowledge Retrieval"]
        panel_lines.append(f"- Material: {formula or '—'}")
        if prop_values:
            panel_lines.append("\n**Extracted property values:**")
            panel_lines.append(self._fmt_prop_values(prop_values))
        panel = "\n".join(panel_lines)

        guide_layer = (
            f"### Synthesis & Interpretation\n\n{guide or '_No interpretable guide generated._'}"
        )
        return {
            "formula": formula,
            "panel": panel,
            "guide": guide_layer,
            "refs_layer": self._format_refs(refs),
            "refs": refs,
            "tool_results": {
                "formula": formula,
                "prop_values": prop_values,
                "references": refs,
            },
        }


    def run_inverse(self, formula, targets=None, user_context=None):
        """Inverse-design degraded version: formula/property retrieval (L1/L2) + multi-objective scoring only; does not trigger MACE/MatterGen/faiss fingerprints"""
        try:
            import inverse_design as _inv  # lazy import (the top level loads torch/mace/faiss; triggered only by this branch)
        except Exception as e:
            raise RuntimeError(f"Inverse design engine unavailable: {e}")

        metadata = _load_inverse_metadata()

        lines = ["### Inverse Design (Formula / Property Retrieval)"]
        lines.append(f"- Query formula: {formula or '—'}")
        lines.append("- Mode: formula/property retrieval degraded version (MACE/MatterGen disabled this release)")

        # --- Step A: formula retrieval (no feature_dict/index passed -> only L1/L2; faiss fingerprints not triggered) ---
        matches = []
        if formula:
            try:
                matches = _inv._search_by_formula(formula, metadata)
            except Exception as e:
                lines.append(f"- ⚠ Formula search failed: {e}")

        def _fmt_cell(v):
            if v is None or (isinstance(v, float) and v != v):
                return "N/A"
            if isinstance(v, (int, float)):
                return f"{v:.2f}" if abs(v) < 1e7 else f"{v:.3g}"
            return str(v)

        if matches:
            lines.append(f"**Found {len(matches)} match(es) by formula search:**")
            tbl = ["| material_id | formula | spacegroup | pred_tm | pred_er | pred_qf | pred_tcf |",
                   "|---|---|---|---|---|---|---|"]
            for m in matches[:10]:
                tbl.append(
                    f"| {m.get('material_id','')} | {m.get('formula','')} | {m.get('spacegroup','')} "
                    f"| {_fmt_cell(m.get('pred_tm'))} | {_fmt_cell(m.get('pred_er'))} "
                    f"| {_fmt_cell(m.get('pred_qf'))} | {_fmt_cell(m.get('pred_tcf'))} |"
                )
            lines.append("\n".join(tbl))
        else:
            lines.append("_No formula match found._")

        # --- Step B: multi-objective scoring (only when the user gives numeric targets) ---
        target_dict = targets or {}
        if target_dict:
            try:
                prop_cols = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
                key_map = {"tm": "pred_tm", "er": "pred_er",
                           "qf": "pred_qf", "qxf": "pred_qf", "tcf": "pred_tcf"}
                defaults = {"tm": 1500.0, "er": 20.0, "qf": 50000.0, "tcf": 0.0}
                t_vec, w_vec, s_vec = [], [], []
                for col in prop_cols:
                    std_val = metadata[col].std() if col in metadata.columns else None
                    s_vec.append(float(std_val) if std_val and std_val == std_val and std_val != 0 else 1.0)
                    matched_key = next((k for k, v in key_map.items() if v == col), None)
                    if matched_key in target_dict and target_dict[matched_key] is not None:
                        t_vec.append(float(target_dict[matched_key]))
                        w_vec.append(0.25)
                    else:
                        t_vec.append(float(defaults.get(matched_key, 0.0)))
                        w_vec.append(0.0)
                df_top, _ = _inv._step1_score(metadata, t_vec, w_vec, s_vec, top_k=10)
                if df_top is not None and not df_top.empty:
                    tgt_str = ", ".join(f"{k}={v}" for k, v in target_dict.items())
                    lines.append(f"\n**Top candidates by multi-objective scoring (targets: {tgt_str}):**")
                    tbl2 = ["| material_id | formula | pred_er | pred_qf | pred_tcf | score |",
                            "|---|---|---|---|---|---|"]
                    for _, row in df_top.head(10).iterrows():
                        tbl2.append(
                            f"| {row.get('material_id','')} | {row.get('formula','')} "
                            f"| {_fmt_cell(row.get('pred_er'))} | {_fmt_cell(row.get('pred_qf'))} "
                            f"| {_fmt_cell(row.get('pred_tcf'))} | {row.get('score', 0):.4f} |"
                        )
                    lines.append("\n".join(tbl2))
            except Exception as e:
                lines.append(f"- ⚠ Scoring step failed (skipped): {e}")
        panel = "\n".join(lines)

        # --- Knowledge/interpretation layer + references layer ---
        refs = self._retrieve_literature(formula, [])
        guide = ""
        ext = self._get_extractor()
        if ext is not None and formula:
            try:
                guide = ext.generate_report(
                    formula, self._build_literature_context(refs, 20), style="focused"
                )
            except Exception:
                guide = ""
        guide_layer = f"### Synthesis & Interpretation\n\n{guide or '_Knowledge guide unavailable._'}"

        return {
            "formula": formula,
            "panel": panel,
            "guide": guide_layer,
            "refs_layer": self._format_refs(refs),
            "refs": refs,
            "tool_results": {
                "formula": formula,
                "inverse_search": matches,
                "references": refs,
            },
        }

    def run_inverse_full(self, collected, lang="en"):
        """Full-pipeline inverse design inside the chat (Steps 1-4, incl. optional MatterGen) + three-layer disclosure"""
        try:
            import inverse_design as _inv  # lazy import
        except Exception as e:
            raise RuntimeError(f"Inverse design engine unavailable: {e}")

        data = _inv.run_full_inverse(dict(collected))
        lines = ["### Inverse Design (Full Pipeline)"]
        tg = collected.get("targets") or {}
        tcm = collected.get("target_cmp") or {}
        tgt_str = ", ".join(
            f"{k}{('' if tcm.get(k) == '=' else (tcm.get(k) or '='))}{v:g}"
            for k, v in tg.items()
        ) if tg else "—"
        lines.append(f"- Targets: {tgt_str}")
        lines.append(f"- Formula hint: {collected.get('formula') or '—'}")
        lines.append(f"- MatterGen: **{'ON' if collected.get('mattergen') else 'OFF'}**")
        mace_ok = data.get("mace_available", False)
        lines.append(f"- MACE verification: **{'available' if mace_ok else 'unavailable'}**")
        if data.get("error"):
            lines.append(f"- ⚠ pipeline warning: {str(data['error'])[:300]}")

        def _fmt_cell(v):
            if v is None or (isinstance(v, float) and v != v):
                return "N/A"
            if isinstance(v, (int, float)):
                return f"{v:.2f}" if abs(v) < 1e7 else f"{v:.3g}"
            return str(v)

        dfk = data.get("df_known")
        if dfk is not None and not dfk.empty:
            lines.append(f"\n**Step 1 — Target-driven candidate lookup ({len(dfk)}):**")
            tbl = ["| material_id | formula | pred_er | pred_qf | pred_tcf | score |",
                   "|---|---|---|---|---|---|"]
            for _, row in dfk.head(10).iterrows():
                tbl.append(
                    f"| {row.get('material_id','')} | {row.get('formula','')} "
                    f"| {_fmt_cell(row.get('pred_er'))} | {_fmt_cell(row.get('pred_qf'))} "
                    f"| {_fmt_cell(row.get('pred_tcf'))} | {row.get('score', 0):.4f} |"
                )
            lines.append("\n".join(tbl))

        dfu = data.get("df_unknown")
        if dfu is not None and not dfu.empty:
            lines.append(f"\n**Step 2 — Unknown/analog candidates ({len(dfu)}):**")
            tbl2 = ["| material_id | formula | pred_er | pred_qf | pred_tcf |",
                    "|---|---|---|---|---|"]
            for _, r in dfu.head(10).iterrows():
                tbl2.append(
                    f"| {r.get('material_id','')} | {r.get('formula','')} "
                    f"| {_fmt_cell(r.get('pred_er'))} | {_fmt_cell(r.get('pred_qf'))} "
                    f"| {_fmt_cell(r.get('pred_tcf'))} |"
                )
            lines.append("\n".join(tbl2))

        dfs = data.get("df_unknown_scored")
        if dfs is not None and not dfs.empty:
            lines.append(f"\n**Step 4a — Unknown analogs validated ({len(dfs)}):**")
            tbl3 = ["| material_id | formula | pred_er | pred_qf | pred_tcf | score |",
                    "|---|---|---|---|---|---|"]
            for _, r in dfs.head(10).iterrows():
                tbl3.append(
                    f"| {r.get('material_id','')} | {r.get('formula','')} "
                    f"| {_fmt_cell(r.get('pred_er'))} | {_fmt_cell(r.get('pred_qf'))} "
                    f"| {_fmt_cell(r.get('pred_tcf'))} | {r.get('score', 0):.4f} |"
                )
            lines.append("\n".join(tbl3))

        dfg = data.get("df_gen")
        n_gen = data.get("generated_total", 0)
        n_pass = data.get("generated_pass", 0)
        if dfg is not None and not dfg.empty:
            lines.append(f"\n**Step 4b — MatterGen generated (passed {n_pass}/{n_gen}):**")
            tbl4 = ["| material_id | formula | pred_er | pred_qf | pred_tcf | score |",
                    "|---|---|---|---|---|---|"]
            for _, r in dfg.head(10).iterrows():
                tbl4.append(
                    f"| {r.get('material_id','')} | {r.get('formula','')} "
                    f"| {_fmt_cell(r.get('pred_er'))} | {_fmt_cell(r.get('pred_qf'))} "
                    f"| {_fmt_cell(r.get('pred_tcf'))} | {r.get('score', 0):.4f} |"
                )
            lines.append("\n".join(tbl4))
        elif n_gen:
            lines.append(f"\n**Step 4b — MatterGen generated (passed {n_pass}/{n_gen})**")

        panel = "\n".join(lines)

        # --- Knowledge/interpretation layer + references layer ---
        formula = collected.get("formula") or data.get("formula")
        refs = self._retrieve_literature(formula, []) if formula else []
        guide = ""
        ext = self._get_extractor()
        if ext is not None and formula:
            try:
                guide = ext.generate_report(
                    formula, self._build_literature_context(refs, 20), style="focused"
                )
            except Exception:
                guide = ""
        guide_layer = (
            f"### Synthesis & Interpretation\n\n"
            f"{guide or '_No formula provided; synthesis/literature interpretation not generated. Reply with [supplement system BaTiO3] or specify a candidate formula to rerun and unlock._'}"
        )

        return {
            "formula": formula,
            "panel": panel,
            "guide": guide_layer,
            "refs_layer": self._format_refs(refs),
            "refs": refs,
            "tool_results": {
                "formula": formula,
                "differentiable": True,
                "n_known": len(dfk) if dfk is not None else 0,
                "n_unknown": len(dfu) if dfu is not None else 0,
                "n_generated": n_gen,
                "n_scored": n_pass,
                "gen_out_dir": data.get("gen_out_dir"),
                "references": refs,
            },
        }


def _assemble_three_layer(res, lang="en"):
    """Three-layer information disclosure aggregation: performance panel + knowledge/synthesis interpretation + references"""
    parts = []
    if res.get("panel"):
        parts.append(res["panel"])
    if res.get("guide"):
        parts.append(res["guide"])
    if res.get("refs_layer"):
        parts.append(res["refs_layer"])
    return "\n\n---\n\n".join(parts)


# ===================================================================
#  Inverse-design multi-turn collection state machine (CADR chat path)
#  Added 2026-09-03
# ===================================================================

_INVERSE_FLOW_KEY = "inverse_flow"


def _ensure_inverse_flow():
    flow = st.session_state.get(_INVERSE_FLOW_KEY)
    if not isinstance(flow, dict):
        flow = {
            "stage": "idle",
            "collected": {
                "targets": {},
                "target_cmp": {},
                "formula": None,
                "mattergen": False,
                "weights": {},
                "sg_mode": "OFF",
                "mace_sim_mode": "OFF",
                "w_sim": 0.5,
            },
            "invalid_strikes": {"targets": 0},
            "no_progress": 0,
            "total_rounds": 0,
        }
        st.session_state[_INVERSE_FLOW_KEY] = flow
    return flow


def _clear_inverse_flow():
    st.session_state.pop(_INVERSE_FLOW_KEY, None)


def _inverse_manual_hint(lang="en"):
    if lang == "zh":
        return (
            "Having trouble collecting inverse-design parameters in the chat? To avoid back-and-forth, go directly to the **IDO** "
            "tab, fill in the performance targets and parameters manually, and run it in one click - the visualization there is more complete. If you prefer, you can also re-describe your requirements and I will try again."
        )
    return (
        "Parameter collection via chat hit a dead end. Please go to the **IDO** tab "
        "and run the pipeline manually for full visualization. You can also re-describe "
        "the requirement any time and I'll retry."
    )


def _inverse_capability_sheet(col, lang="en"):
    """Inverse-design chat-tunable parameter quick reference: current config + chat syntax for each option."""
    tg = col.get("targets") or {}
    tcm = col.get("target_cmp") or {}

    def _fmt_tgt():
        if not tg:
            return "(none collected yet)" if lang == "zh" else "(none collected yet)"
        parts = []
        for k, v in tg.items():
            c = tcm.get(k)
            sym = c if c in ("<", "<=", ">", ">=", "≈") else "="
            parts.append(f"{k}{sym}{v:g}")
        return ", ".join(parts)

    mg = (
            "on (collected; will confirm before running)" if col.get("mattergen")
            else "off (default; reply \"enable MatterGen\" to turn on generative candidates)"
    ) if lang == "zh" else (
        "ON (collected; will confirm before run)" if col.get("mattergen")
        else "OFF (default; reply 'enable MatterGen' to add generative candidates)"
    )
    wts = (
        "、".join(f"{k}={v}" for k, v in (col.get("weights") or {}).items())
            if col.get("weights") else "equal weights (customize by replying, e.g. \"weights er=0.5,qf=0.3,tcf=0.2\")"
    ) if lang == "zh" else (
        ", ".join(f"{k}={v}" for k, v in (col.get("weights") or {}).items())
        if col.get("weights") else "uniform (reply e.g. 'weights er=0.5,qf=0.3,tcf=0.2')"
    )
    sg = (
        str(col.get("sg_mode")) if col.get("sg_mode") != "OFF"
            else "off (reply \"constrain space group P4mm\" or \"enable sg\" to restrict the structure)"
    ) if lang == "zh" else (
        str(col.get("sg_mode")) if col.get("sg_mode") != "OFF"
        else "OFF (reply 'restrict space group P4mm' or 'enable sg')"
    )
    sim = (
        f"{col.get('mace_sim_mode')}（w_sim={col.get('w_sim')}）"
        if col.get("mace_sim_mode") != "OFF"
            else "off (reply \"enable mace simulation\" to add unknown-phase similarity validation)"
    ) if lang == "zh" else (
        f"{col.get('mace_sim_mode')} (w_sim={col.get('w_sim')})"
        if col.get("mace_sim_mode") != "OFF"
        else "OFF (reply 'enable mace sim' for unknown-phase similarity check)"
    )

    if lang == "zh":
        return (
            "**Chat-tunable parameters quick reference**\n"
            f"- Targets: `{_fmt_tgt()}` (support `≈` `>` `<` `≥` `≤` or words like below/greater than/near zero; at least 1)\n"
            f"- System/formula hint: `{col.get('formula') or 'not provided (optional) - providing it unlocks Step 2/4 structure analogy, synthesis interpretation, and the references layer'}`\n"
            f"- **MatterGen generation**: {mg}\n"
            f"- Target weights: {wts}\n"
            f"- Structure guidance (space group): {sg}\n"
            f"- MACE similarity simulation: {sim}"
        )
    return (
        "**Adjustable parameters in chat**\n"
        f"- Targets: `{_fmt_tgt()}` (supports `≈` `>` `<` `≥` `≤` or above/below/near-zero; >=1 required)\n"
        f"- Formula/system hint: `{col.get('formula') or 'not provided (optional) — unlocks Steps 2/4 analogs, synthesis guide & literature'}`\n"
        f"- **MatterGen generation**: {mg}\n"
        f"- Target weights: {wts}\n"
        f"- Space-group guidance: {sg}\n"
        f"- MACE similarity simulation: {sim}"
    )


def _inverse_review_menu(col, lang="en"):
    """Config-menu confirmation state: when targets are complete but the user has not yet confirmed, show the current config sheet and wait for setup/confirmation before running."""
    sheet = _inverse_capability_sheet(col, lang)
    if lang == "zh":
        msg = (
            "Inverse-design request received. Here is the **current config menu**; reply with \"confirm\" once everything is set to start:\n\n"
            + sheet
            + "\n\n"
            "You can keep adjusting in the chat:\n"
            "- Targets: e.g. \"dielectric constant below 10\" / \"Qf>30000\" / \"TCF≈0\" (support `≈` `>` `<` `≥` `≤` or words like below/greater than/near zero)\n"
            "- System/formula: \"supplement system BaTiO3\"\n"
            "- Generative candidates: \"enable MatterGen\" (\"no MatterGen\" disables it)\n"
            "- Target weights: \"weights er=0.5,qf=0.3,tcf=0.2\"\n"
            "- Structure guidance: \"constrain space group P4mm\" / \"enable sg\"\n"
            "- MACE similarity simulation: \"enable mace simulation\"\n\n"
            "If nothing else needs setting, just reply \"confirm\" / \"run\" / \"start\" to begin the design."
        )
        if col.get("mattergen"):
            msg += (
                "\n\n> ⚠️ MatterGen is enabled: it will invoke the diffusion model and MACE validation,"
                "blocking the chat for an estimated 30-45 minutes. This step runs first after confirmation."
            )
    else:
        msg = (
            "Inverse-design request received. Here is the **current config menu**; "
            "set everything, then reply **confirm** to run:\n\n"
            + sheet
            + "\n\n"
            "You may keep adjusting in chat, e.g.:\n"
            "- Targets: 'εr below 10' / 'Qf>30000' / 'τf≈0'\n"
            "- Formula: 'add formula BaTiO3'\n"
            "- MatterGen: 'enable MatterGen' / 'disable MatterGen'\n"
            "- Weights: 'weights er=0.5,qf=0.3,tcf=0.2'\n"
            "- Space group: 'restrict space group P4mm' / 'enable sg'\n"
            "- MACE sim: 'enable mace sim'\n\n"
            "If nothing else to tweak, reply **confirm/run/start** to begin."
        )
        if col.get("mattergen"):
            msg += (
                "\n\n> ⚠️ MatterGen is ON: diffusion model + MACE validation will be run, "
                "blocking the chat for ~30–45 min."
            )
    return msg


def _parse_inverse_prompt(prompt, parsed):
    """Extract the inverse-design parameter increment from the current prompt. Returns (updates, invalid, confirm_intent)."""
    updates = {
        "targets": {}, "target_cmps": {}, "formula": None, "mattergen": None, "weights": {},
        "sg_mode": None, "mace_sim_mode": None, "w_sim": None,
    }
    invalid = []

    # targets (illegal numeric values go to invalid; directions recorded as well)
    # In the weight segment, "key=value" is a weight assignment rather than a target; strip it first to avoid wrongly overriding/adding targets
    _tgt_prompt = prompt
    if re.search(r"权重|weight", prompt, re.IGNORECASE):
        _tgt_prompt = re.sub(
            r"(?:权重|weight)\s*[=:：]?[^，。；;\n]*",
            "", _tgt_prompt, flags=re.IGNORECASE,
        )
    for k, (fv, cmp_) in (_extract_target_values(_tgt_prompt) or {}).items():
        try:
            fv = float(fv)
            if fv != fv:  # NaN
                raise ValueError
            updates["targets"][k] = fv
            updates["target_cmps"][k] = cmp_
        except (TypeError, ValueError):
            invalid.append(f"target[{k}]")

    if parsed.get("formula"):
        updates["formula"] = str(parsed["formula"])

    # mattergen switch
    if re.search(r"mattergen|matter\s*gen|生成|m\s*gen", prompt, re.IGNORECASE):
    # Note: Chinese "不要/不用" must be matched as substrings (\b does not work on CJK boundaries)
        if re.search(
            r"\b(?:off|no|false|none|disable)\b|不要|不用|暂停|禁止",
            prompt,
            re.IGNORECASE,
        ):
            updates["mattergen"] = False
        else:
            updates["mattergen"] = True

    # weights (e.g. "εr weight 0.3")
    _wmap = {
        "tm": ["tm", "melting"],
        "er": ["er", "εr", "epsr"],
        "qf": ["qf", "q×f", "qxf"],
        "tcf": ["tcf", "τf", "tauf"],
    }
    def _norm_wkey(s):
        s = s.lower().replace("ε", "e").replace("τ", "t").replace("×", "x").strip()
        if s in ("er", "epsr"):
            return "er"
        if s in ("qf", "qxf"):
            return "qf"
        if s in ("tcf", "tauf"):
            return "tcf"
        if s in ("tm", "melting"):
            return "tm"
        return None

    def _set_weight(updates, key, val, invalid):
        try:
            wv = float(val)
            if wv >= 0:
                updates["weights"][key] = wv
        except ValueError:
            invalid.append(f"weight[{key}]")

    # Batch: all key=value in the weight segment (e.g. "weights er=0.5,qf=0.3,tcf=0.2")
    mseg = re.search(
        r"(?:权重|weight)[=:：]?\s*([^，。；;\n]*)", prompt, re.IGNORECASE
    )
    if mseg:
        for m in re.finditer(
            r"([A-Za-zετ×_]+)\s*[=:：]\s*([0-9]*\.?[0-9]+)",
            mseg.group(1),
            re.IGNORECASE,
        ):
            key = _norm_wkey(m.group(1))
            if key:
                _set_weight(updates, key, m.group(2), invalid)

    for key, tokens in _wmap.items():
        for tok in tokens:
            m = re.search(
                r"(?:weight|权重|w_)\s*%s\D{0,8}([0-9]*\.?[0-9]+)" % re.escape(tok),
                prompt,
                re.IGNORECASE,
            )
            if m:
                _set_weight(updates, key, m.group(1), invalid)
                break

    # sg_mode (structure guidance)
    if re.search(r"同空间群|same\s*(?:space|sg)|结构约束", prompt, re.IGNORECASE):
        updates["sg_mode"] = "same"
    elif re.search(r"按晶系|system", prompt, re.IGNORECASE):
        updates["sg_mode"] = "system"

    # mace_sim / w_sim (seed similarity)
    if re.search(r"相似度|similarity|seed.?sim", prompt, re.IGNORECASE):
        updates["mace_sim_mode"] = "Into composite score"
    m = re.search(
        r"(?:w_sim|similarity\s*weight|相似度权重)\D{0,8}([0-9]*\.?[0-9]+)",
        prompt,
        re.IGNORECASE,
    )
    if m:
        try:
            wv = float(m.group(1))
            if 0.0 <= wv <= 2.0:
                updates["w_sim"] = wv
        except ValueError:
            invalid.append("w_sim")

    confirm_intent = bool(
        re.search(
            r"\b(?:run|start|go|ok|yes|confirm)\b|确认|确定|开始|执行|启动|可以|运行|跑|试试",
            prompt,
            re.IGNORECASE,
        )
    )
    return updates, invalid, confirm_intent


def _handle_inverse_flow(router, prompt, parsed, lang="en"):
    """Inverse-design multi-turn parameter-collection state machine; returns a dict compatible with process_chat_message."""
    flow = _ensure_inverse_flow()
    col = flow["collected"]
    flow["total_rounds"] += 1

    updates, invalid, confirm = _parse_inverse_prompt(prompt, parsed)

    for k, v in updates["targets"].items():
        col["targets"][k] = v
    for k, c in updates["target_cmps"].items():
        col["target_cmp"][k] = c
    if updates["formula"]:
        col["formula"] = updates["formula"]
    if updates["mattergen"] is not None:
        col["mattergen"] = updates["mattergen"]
    for k, v in updates["weights"].items():
        col["weights"][k] = v
    if updates["sg_mode"]:
        col["sg_mode"] = updates["sg_mode"]
    if updates["mace_sim_mode"]:
        col["mace_sim_mode"] = updates["mace_sim_mode"]
    if updates["w_sim"] is not None:
        col["w_sim"] = updates["w_sim"]

    added_any = bool(
        updates["targets"]
        or updates["formula"]
        or updates["mattergen"] is not None
        or updates["weights"]
        or updates["sg_mode"]
        or updates["mace_sim_mode"]
    )
    if invalid:
        flow["invalid_strikes"]["targets"] = (
            flow["invalid_strikes"].get("targets", 0) + len(invalid)
        )
    if added_any or confirm:
        flow["no_progress"] = 0
    else:
        flow["no_progress"] += 1

    # ---- Guardrail: session-level invalid >=3 / total turns >=6 / consecutive no-progress >=2 -> redirect to the manual page ----
    if (
        sum(flow["invalid_strikes"].values()) >= 3
        or flow["total_rounds"] >= 6
        or flow["no_progress"] >= 2
    ):
        _clear_inverse_flow()
        return {
            "content": _inverse_manual_hint(lang),
            "intent": "inverse",
            "tool_results": None,
        }

    n_targets = len(col["targets"])
    wants_gen = bool(col["mattergen"])

    # ---- Runnable: >=1 valid target, and the user has explicitly confirmed in the config-menu confirmation state ----
    if n_targets >= 1 and confirm:
        collected_cfg = {
            "formula": col["formula"],
            "targets": dict(col["targets"]),
            "weights": dict(col["weights"]) or None,
            "mattergen": wants_gen,
            "mg_num": 8,
            "mg_bs": 4,
            "sg_mode": col["sg_mode"],
            "mace_sim_mode": col["mace_sim_mode"],
            "w_sim": col["w_sim"],
        }
        try:
            res = router.run_inverse_full(collected_cfg, lang=lang)
            _clear_inverse_flow()
            content = _assemble_three_layer(res, lang=lang)
    # When no formula is given, add an alternative hint to avoid a bare "unavailable" notice
            if not col["formula"]:
                extra = (
                    "> 💡 No formula/system hint provided: structure analogy (Steps 2/4), synthesis interpretation, and the references layer were not generated."
                    "Reply with \"supplement system BaTiO3\" or \"rerun with XXX\" to unlock the full results."
                    if lang == "zh"
                    else "> 💡 No formula/system hint given: structural analogs (Steps 2/4), "
                         "synthesis guide and literature layer were skipped. Reply "
                         "「add formula BaTiO3」or「rerun with XXX」to unlock the full result."
                )
                content = content + "\n\n---\n\n" + extra
    # After running, append a "tunable-parameters quick reference" screen so the user knows what else can be enabled
            content = content + "\n\n---\n\n" + _inverse_capability_sheet(col, lang)
            return {
                "content": content,
                "intent": "inverse",
                "tool_results": res.get("tool_results"),
            }
        except Exception as e:
    # Full pipeline failed -> fall back to the degraded retrieval version
            _clear_inverse_flow()
            try:
                resd = router.run_inverse(
                    col["formula"], targets=dict(col["targets"]), user_context=prompt
                )
                warn = (
                    "> ⚠️ The full in-chat pipeline failed; degraded to the retrieval-scoring version.\n\n"
                    if lang == "zh"
                    else "> ⚠️ Full chat pipeline failed, fell back to retrieval scoring.\n\n"
                )
                return {
                    "content": warn + _assemble_three_layer(resd, lang=lang),
                    "intent": "inverse",
                    "tool_results": resd.get("tool_results"),
                }
            except Exception as e2:
                return {
                    "content": _inverse_manual_hint(lang) + f"\n\n> {e}"[:600],
                    "intent": "inverse",
                    "tool_results": None,
                }

    # ---- Targets complete but not confirmed -> config-menu confirmation state (show the menu first; run only after the user confirms) ----
    if n_targets >= 1 and not confirm:
        msg = _inverse_review_menu(col, lang)
        return {"content": msg, "intent": "inverse", "tool_results": None}

    # ---- Not all targets given -> ask again (with the full capability quick reference) ----
    if lang == "zh":
        msg = (
            "OK, I will help you with inverse design. I have not received any valid performance-target values yet.\n\n"
            "**Please provide at least one target**, e.g.:\n"
            "- `εr≈50` / \"dielectric constant below 30\"\n"
            "- `Q×f>50000` / \"Qf greater than 20000\"\n"
            "- `τf≈0` / \"TCF near zero\"\n\n"
            "Besides targets, you can also set the following options directly in the chat (defaults are used if not set); "
            "once you have written them, I will run the full pipeline in one go:"
        )
    else:
        msg = (
            "Sure, let's set up the inverse design. **Please give at least one target**, e.g.:\n"
            "- `εr≈50` / `εr below 30`\n"
            "- `Q×f>50000` / `Qf above 20000`\n"
            "- `τf≈0` / `TCF near zero`\n\n"
            "You can also set the optional parameters below in chat (defaults apply otherwise). "
            "I'll run the full pipeline once targets are ready:"
        )
    msg = msg + "\n\n" + _inverse_capability_sheet(col, lang)
    return {"content": msg, "intent": "inverse", "tool_results": None}


def process_chat_message(prompt, current_data, current_refs, agent=None,
                         csv_engine=None, rag_engine=None, extractor=None, scholar=None):
    """Process a chat message: intent parsing -> tool routing -> three-layer disclosure aggregation; quiz goes through the original two-stage RAG path"""
    if (
        not hasattr(st.session_state, "chat_history")
        or st.session_state.chat_history is None
    ):
        st.session_state.chat_history = []

    chat_history = st.session_state.get("chat_history", [])
    parsed = intent_parse(prompt, current_data, chat_history)
    intent = parsed.get("intent", "quiz")
    lang = detect_language(prompt)

    # Heuristic fallback: when "how are the dielectric/microwave properties of material X" is misclassified as knowledge, force an upgrade to predict
    if intent == "knowledge" and parsed.get("formula"):
        import re as _re
        _p = prompt.lower()
        _perf_hit = _re.search(
            r"介电|微波|性能|品质|温度系数|ε|q[×x]?f|qxf|如何|怎样|\ber\b|\bqf\b|\btcf\b|dielectric|microwave|performance|how",
            _p,
        )
        _proc_hit = _re.search(
            r"合成|制备|工艺|机理|相变|文献|synthesis|prepar|fabricat|mechanism|process",
            _p,
        )
        if _perf_hit and not _proc_hit:
            intent = "predict"

    def _fallback(include_properties=True):
        content = _run_quiz_rag(
            prompt, current_data, current_refs, include_properties=include_properties
        )
        return {"content": content, "intent": "quiz", "tool_results": None}

    # Mechanism/principle-type questions: essentially scientific Q&A asking to explain causes/defect chemistry/decoupling trade-offs;
    # they go through the free-text Q&A branch and must NOT be bound to the current material in current_data for attribute extraction (which would output another property table).
    # Inverse design is handled by its own parameter-collection state machine and is outside this guardrail.
    if intent != "inverse" and _is_mechanism_question(prompt):
        return _fallback(include_properties=False)

    if intent == "quiz":
        return _fallback()

    # Non-dielectric-property questions such as process/cost/preparation: go through the classic RAG Q&A (original functionality);
    # do not generate a technical report or backfill the report page, avoiding the off-topic output of "asking about cost but getting a performance report"
    if intent == "knowledge":
        _ip = prompt.lower()
        _proc_cost_hit = re.search(
            r"合成|制备|工艺|流程|成本|价格|贵不贵|贵吗|便宜|原料|设备|产率|污染|环保|"
            r"怎么(做|制|造|弄)|如何(做|制备|合成|开发|实现|降低成本)|"
            r"cost|price|cheap|expensive|prepar|synthes|fabricat|process|route|"
            r"raw material|feedstock|yield|scalab",
            _ip,
        )
        if _proc_cost_hit:
            return _fallback()

    router = (
        ChatToolRouter(
            agent,
            csv_engine=csv_engine,
            rag_engine=rag_engine,
            extractor=extractor,
            scholar=scholar,
        )
        if agent is not None
        else None
    )
    if router is None:
        return _fallback()

    formula = parsed.get("formula") or ""
    try:
        if intent == "predict":
            if not formula:
                hint = (
                    "Please provide a material formula to start a prediction, e.g. predict er / Qxf / tf of BaTiO3."
                    if lang == "zh"
                    else "Please provide a chemical formula to predict, "
                         "e.g., \"predict BaTiO3 er/qf/tcf\"."
                )
                return {"content": hint, "intent": "predict", "tool_results": None}
            res = router.run_predict(
                formula, targets=parsed.get("target_props") or [], user_context=prompt
            )
        elif intent == "knowledge":
            res = router.run_knowledge(
                formula, keywords=parsed.get("keywords") or [], user_context=prompt
            )
        elif intent == "inverse":
            # Multi-turn parameter-collection state machine: ask for missing params -> guardrail -> run the full pipeline directly or after confirmation
            return _handle_inverse_flow(router, prompt, parsed, lang)
        else:
            return _fallback()

        content = _assemble_three_layer(res, lang=lang)
        tr = res.get("tool_results")
        # Attach the fields needed for report backfill, so the Render layer can link the Property Prediction / Technical Report pages
        if isinstance(tr, dict) and intent in ("predict", "knowledge"):
            tr["report_guide"] = res.get("guide")
            tr["report_formula"] = res.get("formula")
            rfs = res.get("refs") or []
            try:
                tr["report_context"] = router._build_literature_context(rfs, 20)
            except Exception:
                tr["report_context"] = None
        return {"content": content, "intent": intent, "tool_results": tr}
    except Exception as e:
        # On failure, degrade: fall back to quiz and notify
        fallback = _run_quiz_rag(prompt, current_data, current_refs)
        if lang == "zh":
            banner = f"> ⚠️ `{intent}` module call failed ({e}); fell back to regular Q&A.\n\n"
        else:
            banner = f"> ⚠️ `{intent}` module failed ({e}), fell back to regular Q&A.\n\n"
        return {"content": banner + fallback, "intent": "quiz", "tool_results": None}


def _run_quiz_rag(prompt, current_data, current_refs, include_properties=True):
    """Process a chat message and run two-stage RAG.

    When include_properties=False, the current material's predicted-property list is not written into the answer context
    (mechanism/principle-type questions go through this branch, avoiding an irrelevant property-table output)."""
    # Ensure chat_history exists
    if (
        not hasattr(st.session_state, "chat_history")
        or st.session_state.chat_history is None
    ):
        st.session_state.chat_history = []

    lang = detect_language(prompt)
    formula = current_data.get("Formula", "") if current_data else ""

    # Build context info
    context_info = []
    if current_data:
        context_info.append(f"Material: {current_data.get('Formula', 'Unknown')}")
        context_info.append(
            f"Structure: {current_data.get('Structure_Type', 'Unknown')}"
        )
        for k, v in current_data.items():
            if k.endswith("_mean") and v is not None:
                prop = k.replace("_mean", "")
                info = get_property_info(prop)
                symbol = info.symbol if info.symbol else prop
                dev = current_data.get(k.replace("_mean", "_dev"), 0)
                context_info.append(f"{symbol}: {v:.2f} ± {dev:.2f}")
    context_str = "\n".join(context_info) if context_info else "No data"

    # Stage 1: question understanding
    system_prompt = """Analyze the question and generate search keywords.
Output JSON: {"search_keywords": [...], "refined_question": "..."}"""

    client = OpenAI(api_key=config.DS_API_KEY, base_url=config.DEEPSEEK_API_URL)
    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
    )

    response = client.chat.completions.create(
        model=config.DS_MODEL, messages=messages, temperature=0.3, max_tokens=300
    )

    import json

    try:
        result_text = response.choices[0].message.content
        if "```json" in result_text:
            result_text = result_text.replace("```json", "").replace("```", "")
        elif "```" in result_text:
            result_text = result_text.replace("```", "")
        query_data = json.loads(result_text.strip())
    except:
        query_data = {"search_keywords": [], "refined_question": prompt}

    search_keywords = query_data.get("search_keywords", [])

    # Stage 2: search literature
    scholar = LiteratureAgent()
    all_results = []

    if current_refs and len(current_refs) >= 3:
        search_results = current_refs[:5]
    else:
        for kw in search_keywords[:3]:
            results = (
                scholar.online_search(f"{formula} {kw}", limit=3) if formula else []
            )
            all_results.extend(results)

        seen = set()
        search_results = []
        for r in all_results:
            title = r.get("title", "")
            if title and title not in seen:
                seen.add(title)
                search_results.append(r)
        search_results = search_results[:5]

    # Stage 3: generate the answer
    answer_parts = []
    if current_data and include_properties:
        answer_parts.append("## Predicted Properties")
        for k, v in current_data.items():
            if k.endswith("_mean") and v is not None:
                prop = k.replace("_mean", "")
                info = get_property_info(prop)
                symbol = info.symbol if info.symbol else prop
                unit = info.unit if info.unit else ""
                dev = current_data.get(k.replace("_mean", "_dev"), 0)
                if unit:
                    answer_parts.append(f"- {symbol}: {v:.2f} ± {dev:.2f} {unit}")
                else:
                    answer_parts.append(f"- {symbol}: {v:.2f} ± {dev:.2f}")

    answer_parts.append(f"\n## Literature ({len(search_results)} papers)")
    for ref in search_results[:3]:
        answer_parts.append(f"\n**{ref.get('title', 'Unknown')}**")
        answer_parts.append(ref.get("content", "")[:800])

    answer_context = "\n".join(answer_parts)

    if lang == "zh":
        sys_prompt = """You are a materials science expert. Answer in detail and comprehensively, following the language of the user's question. Must cite specific data and scientific mechanisms."""
    else:
        sys_prompt = """You are a materials science expert. Answer in detail, citing specific data and scientific mechanisms."""

    messages = [{"role": "system", "content": sys_prompt}]
    chat_history = st.session_state.get("chat_history", [])
    for msg in chat_history[-6:]:
        messages.append(msg)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{answer_context}\n\nQuestion: {prompt}\n\nProvide a detailed answer.",
        }
    )

    response = client.chat.completions.create(
        model=config.DS_MODEL, messages=messages, temperature=0.3, max_tokens=3000
    )
    return response.choices[0].message.content


def render_chat_tab(current_data, current_refs, agent=None,
                    csv_engine=None, rag_engine=None, extractor=None, scholar=None):
    """Render the chat tab; agent is optional, used for predict/knowledge/inverse tool routing"""
    init_chat_state()

    # Ensure data is not None
    if current_data is None:
        current_data = {}
    if current_refs is None:
        current_refs = []

    # Title: structure fully consistent with other pages (single .page-header block, no shift hack)
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Context-Aware Dialogue Orchestrator</p>
        <p class="subtitle">Interactive scientific reasoning and task orchestration across HRKE, ARMPP and IDO</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Current material (no threshold: chat works even without analysis context; predict/knowledge/inverse modules triggered on demand)
    if current_data and current_data.get("Formula"):
        st.success(f"📌 Current: {current_data.get('Formula', 'Unknown')}")
    else:
    # Session inheritance: the last successful prediction (current or historical session) can be reused as the chat context in one click
        last_pred = st.session_state.get("_last_prediction")
        if isinstance(last_pred, dict) and last_pred.get("Formula"):
            inherit_col, _ = st.columns([1, 5])
            with inherit_col:
                if st.button(
                    f"📌 Reusing last prediction: {last_pred.get('Formula')}",
                    help="Use the last prediction as dialogue context (links with Report / Predict tabs)",
                    use_container_width=True,
                ):
                    st.session_state.analyzed_data = last_pred
                    st.rerun()

    # Ensure chat_history exists
    if (
        not hasattr(st.session_state, "chat_history")
        or st.session_state.chat_history is None
    ):
        st.session_state.chat_history = []

    # Chat display area (chat_input cannot be used inside a container)
    chat_history = st.session_state.get("chat_history", [])
    for msg in chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Thinking-state placeholder
    thinking_placeholder = st.empty()

    # Input box (must be at the top level, not inside a container)
    try:
        prompt = st.chat_input("Ask about this material...")
    except StreamlitAPIException:
        prompt = st.text_input("Ask about this material...", key="chat_fallback_input")

    if prompt:
        # Immediately record and display the user message
        if not hasattr(st.session_state, "chat_history"):
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # If using text_input, clear the input box
        if "chat_fallback_input" in st.session_state:
            st.session_state.chat_fallback_input = ""
        st.rerun()

    # Processing logic - check whether the last entry is a user message
    chat_history = st.session_state.get("chat_history", [])
    if chat_history and chat_history[-1]["role"] == "user":
        # Get the last user message
        last_user_msg = chat_history[-1]["content"]

        # Record the start time
        start_time = time.time()

        # Show a thinking state in the chat box
        with thinking_placeholder:
            with st.chat_message("assistant"):
            # Process in a thread to avoid blocking the UI
                import threading

                result = [None]
                error_msg = [None]

                def run_process():
                    try:
                        result[0] = process_chat_message(
                            last_user_msg, current_data, current_refs, agent,
                            csv_engine=csv_engine, rag_engine=rag_engine,
                            extractor=extractor, scholar=scholar,
                        )
                    except Exception as e:
                        error_msg[0] = str(e)

                # Start the thread
                thread = threading.Thread(target=run_process)
                # Bind the current script-run context to the background thread; otherwise st.session_state inside the thread
                # degrades to an empty shell (reads return empty, writes are dropped), @st.cache_resource cannot hit,
                # and a "missing ScriptRunContext" notice is printed.
                try:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx

                    add_script_run_ctx(thread)
                except Exception:
                # Compatible with environments that do not support this interface: silently degrade on binding failure without affecting the main flow
                    pass
                thread.start()

                # Show the initial state
                status_placeholder = st.empty()
                status_placeholder.markdown("⏳ *Thinking...*")

                # Wait for the thread to finish, updating the display periodically
                while thread.is_alive():
                    elapsed = int(time.time() - start_time)
                    status_placeholder.markdown(f"⏳ *Thinking... {elapsed}s*")
                    time.sleep(1)

                thread.join()

                # Processing finished
                if error_msg[0]:
                    if (
                        hasattr(st.session_state, "chat_history")
                        and st.session_state.chat_history
                    ):
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": f"❌ Error: {error_msg[0]}",
                            }
                        )
                    else:
                        st.error(f"❌ Error: {error_msg[0]}")
                else:
                    if (
                        hasattr(st.session_state, "chat_history")
                        and st.session_state.chat_history
                    ):
                        resp = result[0]
                        if isinstance(resp, dict):
                            msg_entry = {"role": "assistant", "content": resp.get("content", "")}
                            if resp.get("intent"):
                                msg_entry["intent"] = resp["intent"]
                            if resp.get("tool_results"):
                                msg_entry["tool_results"] = resp["tool_results"]
                            st.session_state.chat_history.append(msg_entry)
                            # Prediction succeeded -> write back analyzed_data (linking Report / prediction tab / multi-turn inheritance)
                            tr = resp.get("tool_results")
                            # Only after a successful predict are the two tabs below fully backfilled;
                            # knowledge (process/cost/review etc.) is not backfilled, to avoid a technical report overwriting existing prediction data
                            if resp.get("intent") == "predict" and isinstance(tr, dict):
                            # Prediction succeeded -> write back analyzed_data (linking Report / prediction tab / multi-turn inheritance)
                                ad = tr.get("analyzed_data")
                                if isinstance(ad, dict) and ad.get("Formula"):
                                    st.session_state.analyzed_data = ad
                                    st.session_state._last_prediction = ad
                            # Fully link Property Prediction: extracted literature values
                                lit = tr.get("extracted_literature_values")
                                if lit is None:
                                    lit = tr.get("prop_values")
                                if lit is not None:
                                    st.session_state.kr_props = lit
                            # Fully link Technical Report: fill in the report directly, no need to click again on the report page
                                guide_full = tr.get("report_guide")
                                if isinstance(guide_full, str) and guide_full.strip():
                                    body = guide_full
                                    if body.startswith("### Synthesis & Interpretation"):
                                        _parts = body.split("\n", 2)
                                        if len(_parts) == 3 and _parts[2].strip():
                                            body = _parts[2].lstrip()
                                    st.session_state.generated_report = body
                                    st.session_state.report_formula = (
                                        tr.get("report_formula")
                                        or (ad or {}).get("Formula")
                                        or tr.get("formula")
                                    )
                                    if tr.get("report_context") is not None:
                                        st.session_state.report_context = tr["report_context"]
                                    st.session_state.report_style = "concise"
                        else:
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": resp}
                            )
                    else:
                        resp = result[0]
                        content = resp.get("content", "") if isinstance(resp, dict) else resp
                        st.success(content)

                st.rerun()

        # Bottom buttons
    if hasattr(st.session_state, "chat_history") and st.session_state.chat_history:
        col_btn, _ = st.columns([1, 4])

        with col_btn:
            st.markdown(
                """
                <style>
        /* Only effective inside the chat module: export-button column */
                div[data-testid="stPopoverBody"] {
                    width: 180px !important;
                    min-width: 150px !important;
                    padding: 10px !important;
                }
                div[data-testid="stPopoverBody"] button {
                    width: 100% !important;
                    margin-bottom: 5px !important;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)

            with c1:
                with st.popover("📥 Export"):
                    export_text = "=" * 50 + "\nCADO Dialogue\n=" * 50 + "\n"
                    if current_data:
                        export_text += (
                            f"\nMaterial: {current_data.get('Formula', '')}\n\n"
                        )
                    chat_history = st.session_state.get("chat_history", [])
                    for msg in chat_history:
                        r = msg.get("role", "")
                        c = msg.get("content", "")
                        export_text += f"[{r}]\n{c}\n\n---\n\n"

                    html = f"<h1>Chat</h1><p>Material: {current_data.get('Formula', '')}</p>"
                    for m in chat_history:
                        r = m.get("role", "")
                        c = m.get("content", "").replace("\n", "<br>")
                        html += f"<p><b>{r}:</b></p><p>{c}</p><hr>"

                    st.download_button(
                        "📄 TXT",
                        export_text,
                        "chat.txt",
                        "text/plain",
                        use_container_width=True,
                    )
                    st.download_button(
                        "📝 HTML",
                        html,
                        "chat.html",
                        "text/html",
                        use_container_width=True,
                    )
                    pdf_data = create_pdf_export(export_text.split("\n"))
                    st.download_button(
                        "📑 PDF",
                        pdf_data,
                        "chat.pdf",
                        "application/pdf",
                        use_container_width=True,
                    )

            with c2:
                with st.popover("🗑️ Clear"):
                    st.write("Clear chat history?")
                    if st.button("Confirm Clear", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
