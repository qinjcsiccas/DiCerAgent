import sys
import os
import re
import math
import pandas as pd
from datetime import datetime

from forward_prediction import ForwardPredictionAgent

KNOWLEDGE_ENGINE_AVAILABLE = False
IMPORT_ERROR_MSG = ""
try:
    from literature import LiteratureAgent
    from extractor import KnowledgeExtractor
    from knowledge_engines import LocalRAGEngine, LocalCSVEngine

    KNOWLEDGE_ENGINE_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_MSG = str(e)
    print(f"⚠️ Knowledge Engine Import Failed: {e}")
except Exception as e:
    IMPORT_ERROR_MSG = str(e)
    print(f"⚠️ Knowledge Engine Init Failed: {e}")


class CentralOrchestrationAgent:
    def __init__(self):
        self.prediction_agent = ForwardPredictionAgent()

        if KNOWLEDGE_ENGINE_AVAILABLE:
            try:
                self.scholar = LiteratureAgent()
            except:
                self.scholar = None

            try:
                self.extractor = KnowledgeExtractor()
                self.rag_engine = LocalRAGEngine()
                self.csv_engine = LocalCSVEngine()
                print("[💡] Knowledge engines initialized (Lazy Loading enabled)")
            except Exception as e:
                print(f"[!] Engine Initialization Error: {e}")
                self.init_error_msg = str(e)
                self.scholar = None
        else:
            self.scholar = None

    def reload_external_plugins(self):
        self.prediction_agent.reload_external_plugins()

    def analyze_single(self, query, manual_feats=None, db_pref=None, user_context=None, enabled_methods=None):
        return self.prediction_agent.analyze_single(query, manual_feats, db_pref, user_context=user_context, enabled_methods=enabled_methods)

    def parse_user_input(self, text):
        db_pref = None
        if re.search(r"db=cod|--cod", text, re.IGNORECASE):
            db_pref = "cod"
            text = re.sub(r"db=cod|--cod", " ", text, flags=re.IGNORECASE)
        elif re.search(r"db=mp|--mp", text, re.IGNORECASE):
            db_pref = "mp"
            text = re.sub(r"db=mp|--mp", " ", text, flags=re.IGNORECASE)

        pattern_params = r"(vm|p|en|pm|sp|cvd|bvs|m|vi|tt)\s*=\s*([\d\.\-]+)"
        matches = re.findall(pattern_params, text, re.IGNORECASE)
        manual_feats = {}
        for key, val in matches:
            try:
                manual_feats[key.lower()] = float(val)
            except:
                pass

        text = re.sub(pattern_params, " ", text, flags=re.IGNORECASE)
        clean_query = text.strip().strip('"').strip("'")
        clean_query = re.sub(r"\s+", "", clean_query)

        return [
            {"query": clean_query, "manual_features": manual_feats, "db_pref": db_pref}
        ]

    def _format_mla(self, doc):
        authors = doc.get("authors", [])
        if isinstance(authors, str):
            if ";" in authors:
                authors = [a.strip() for a in authors.split(";")]
            else:
                authors = [authors]

        author_str = "Unknown Author"
        if authors:
            valid_authors = [
                str(a)
                for a in authors
                if a and str(a).strip() and str(a).strip().lower() != "nan"
            ]
            if not valid_authors:
                author_str = "Unknown Author"
            elif len(valid_authors) > 3:
                author_str = f"{valid_authors[0]}, et al"
            elif len(valid_authors) == 3:
                author_str = (
                    f"{valid_authors[0]}, {valid_authors[1]}, and {valid_authors[2]}"
                )
            elif len(valid_authors) == 2:
                author_str = f"{valid_authors[0]} and {valid_authors[1]}"
            else:
                author_str = valid_authors[0]

        year = doc.get("year", "n.d.")
        title = doc.get("title", "Untitled").strip().strip(".")
        journal = doc.get("journal", "Unknown Source")
        return f'{author_str}. "{title}." *{journal}*, {year}.'

    def print_full_report(self, data, guide, refs, kr_props=None):
        if kr_props is None:
            kr_props = {}

        def _fmt(val, dev=None):
            if val is None:
                return "N/A"
            if dev:
                return f"{val:.2f} ± {dev:.2f}"
            return f"{val:.2f}"

        def _fmt_kr_list(key):
            items = kr_props.get(key)
            if not items:
                return "N/A"
            if isinstance(items, dict):
                items = [items]
            pure_entries = []
            other_entries = []
            for item in items:
                if isinstance(item, str):
                    continue
                val = item.get("value")
                src = item.get("source", "")
                cond = item.get("condition", "Unspecified")
                if val is None or str(val).lower() in ["null", "none", "n/a"]:
                    continue
                src_clean = src.strip()
                if src_clean and not src_clean.startswith("["):
                    src_clean = f"[{src_clean}]"
                if cond.lower() == "pure":
                    s = f"{val} {src_clean}".strip()
                    pure_entries.append(s)
                else:
                    s = f"{val} {src_clean} ({cond})".strip()
                    other_entries.append(s)
            all_entries = pure_entries + other_entries
            if not all_entries:
                return "N/A"
            return "; ".join(all_entries)

        props = {
            "εr": {
                "ml": _fmt(data.get("εr_mean"), data.get("εr_dev")),
                "kr": _fmt_kr_list("er"),
            },
            "Q×f": {
                "ml": _fmt(data.get("Qxf_mean"), data.get("Qxf_dev")),
                "kr": _fmt_kr_list("qxf"),
            },
            "τf": {
                "ml": _fmt(data.get("TCF_mean"), data.get("TCF_dev")),
                "kr": _fmt_kr_list("tcf"),
            },
        }

        print("\n" + "=" * 80)
        print(f"🔬 RESEARCH REPORT: {data.get('Input_Query')}")
        print(
            f"   (ID: {data.get('Material_ID')} | Type: {data.get('Structure_Type')})"
        )
        print("=" * 80)

        print(f"\n📊 1. PROPERTIES DASHBOARD")
        print(f"   {'Property':<10} | {'ML Prediction':<20} | {'Literature Data':<35}")
        print(f"   {'-' * 10}-|-{'-' * 20}-|-{'-' * 35}")
        for k, v in props.items():
            lit_val = v["lit"]
            if len(lit_val) > 40:
                print(f"   {k:<10} | {v['ml']:<20} | {lit_val}")
            else:
                print(f"   {k:<10} | {v['ml']:<20} | {lit_val:<35}")

        if data.get("Error"):
            print(f"\n   ⚠️ Prediction Warning: {data['Error']}")

        print(f"\n🔥 2. SYNTHESIS & PROCESSING GUIDE")
        print("-" * 80)
        clean_guide = guide.replace("# Technical Assessment Report", "").strip()
        print(clean_guide)

        print(f"\n📚 3. REFERENCES (MLA Style, Sorted by Relevance)")
        print("-" * 80)
        if refs:
            for idx, doc in enumerate(refs):
                icon = "📄" if doc.get("has_fulltext") else "📊"
                print(
                    f"   [{idx + 1}] {icon} {self._format_mla(doc)} (Score: {doc.get('score', 0):.2f})"
                )
        else:
            print("   (No specific literature retrieved.)")
        print("=" * 80 + "\n")

    def run(self, user_input):
        targets = self.parse_user_input(user_input)
        display_results = []

        for t in targets:
            query = t.get("query")

            print(f"📈 Running Forward Prediction Agent...")
            data = self.prediction_agent.analyze_single(
                query, t.get("manual_features"), db_pref=t.get("db_pref")
            )

            processing_guide = "Knowledge Engine Unavailable."
            unique_references_list = []
            kr_props = {}

            if not KNOWLEDGE_ENGINE_AVAILABLE:
                processing_guide = (
                    f"⚠️ Knowledge Engine Failed to Load.\nReason: {IMPORT_ERROR_MSG}"
                )
            elif query:
                try:
                    print(f"🧠 Running Literature Agent...")
                    if (
                        not hasattr(self.rag_engine, "vectorstore")
                        or self.rag_engine.vectorstore is None
                    ):
                        self.rag_engine.load()
                    if not hasattr(self.csv_engine, "df") or self.csv_engine.df is None:
                        self.csv_engine.load()

                    aliases = self.extractor.brainstorm_keywords(query)
                    print(f"      -> Brainstormed Keywords: {aliases}")

                    csv_results = self.csv_engine.search(
                        query, aliases=aliases, limit=50
                    )
                    rag_results = self.rag_engine.search(query, aliases=aliases, k=50)

                    merged_docs = {}

                    # Ch2: ChemicalScorer unified scoring on title + abstract
                    for doc in csv_results:
                        cs_input = (doc.get("title", "") + " " + doc.get("abstract_meta", doc.get("abstract", ""))).strip()
                        doc["score"] = self.csv_engine.scorer.evaluate(cs_input, query, aliases=aliases)
                        doc["channel_weight"] = 0.5  # Discount factor for metadata-only channel (tunable)
                        key = self.csv_engine._get_fingerprint(doc["title"])
                        if not key:
                            continue
                        merged_docs[key] = doc

                    for doc in rag_results:
                        full_path = doc.get("filename_raw", "")
                        fraw = os.path.basename(full_path)
                        meta = self.csv_engine.match_metadata(
                            fraw, rag_title=doc.get("title")
                        )

                        if meta is not None:
                            doc["title"] = str(meta.get("Title", doc["title"])).strip()
                            doc["authors"] = str(meta.get("Authors", ""))
                            doc["journal"] = str(meta.get("Source Title", "Local Repo"))
                            try:
                                doc["year"] = int(
                                    float(
                                        str(meta.get("Publication Year", "0")).replace(
                                            "nan", "0"
                                        )
                                    )
                                )
                            except:
                                doc["year"] = 0
                            doc["abstract_meta"] = str(meta.get("Abstract", ""))

                            doi_val = str(meta.get("DOI", meta.get("doi", ""))).strip()
                            if doi_val.lower() != "nan" and doi_val:
                                doc["doi"] = doi_val

                        doc["has_fulltext"] = True

                        title_key = self.csv_engine._get_fingerprint(
                            doc.get("title", "")
                        )

                        if title_key in merged_docs:
                            existing = merged_docs[title_key]
                            existing["content"] = doc["content"]
                            existing["has_fulltext"] = True
                            existing["score"] = (
                                max(existing["score"], doc["score"]) * 1.3
                            )
                            if "abstract_meta" in doc:
                                existing["abstract_meta"] = doc["abstract_meta"]
                        else:
                            merged_docs[title_key] = doc

                    # === Channel 3: OpenAlex online retrieval ===
                    if self.scholar:
                        try:
                            raw_papers = self.scholar.search_literature(query, limit=10)
                            if raw_papers:
                                self.scholar.enrich_papers_with_fulltext(raw_papers, max_fulltext=3)
                                scholar_docs = self.scholar.process_papers_to_dicts(raw_papers)
                                for doc in scholar_docs:
                                    title_key = self.csv_engine._get_fingerprint(doc.get("title", ""))
                                    if not title_key:
                                        continue
                                    # Re-score via ChemicalScorer for fair comparison with local engines
                                    doc["score"] = self.csv_engine.scorer.evaluate(
                                        doc.get("title", ""), query, aliases=aliases
                                    )
                                    doc["channel_weight"] = 0.5  # Discount factor for online metadata channel (tunable)
                                    if title_key in merged_docs:
                                        existing = merged_docs[title_key]
                                        if "doi" not in existing and doc.get("doi"):
                                            existing["doi"] = doc["doi"]
                                    else:
                                        merged_docs[title_key] = doc
                                print(f"      -> OpenAlex: {len(scholar_docs)} results")
                        except Exception as e:
                            print(f"      -> OpenAlex error: {e}")

                    final_docs = list(merged_docs.values())

                    # Apply channel-level discount factors in Fusion stage
                    for d in final_docs:
                        weight = d.get("channel_weight", 1.0)
                        d["score"] = d["score"] * weight

                    filtered_docs = [d for d in final_docs if d["score"] > 0.0]
                    filtered_docs.sort(key=lambda x: x["score"], reverse=True)

                    unique_references_list = filtered_docs
                    llm_context_docs = (
                        filtered_docs[:20] if len(filtered_docs) > 20 else filtered_docs
                    )

                    print(
                        f"      -> Unique relevant documents: {len(filtered_docs)} (Top {len(llm_context_docs)} used for AI)"
                    )

                    abstracts_context = "=== SECTION 1: ABSTRACTS ===\n"
                    details_context = "=== SECTION 2: TEXT CHUNKS ===\n"

                    for idx, doc in enumerate(final_docs):
                        global_id = idx + 1
                        title = doc.get("title", "Untitled")
                        abstract_text = doc.get(
                            "abstract_meta", doc.get("content", "")
                        )[:800]
                        full_chunk = doc.get("content", "")
                        abstracts_context += f"Ref [{global_id}] Title: {title}\nAbstract: {abstract_text}\n\n"
                        if doc.get("has_fulltext"):
                            details_context += (
                                f"Ref [{global_id}] Chunk: {full_chunk}\n\n"
                            )

                    full_context = abstracts_context + "\n" + details_context
                    print(f"   🧪 Synthesizing Intelligence (LLM)...")

                    processing_guide = self.extractor.generate_processing_guide(
                        query, full_context
                    )
                    kr_props = self.extractor.extract_property_values(
                        query, full_context, summary=processing_guide
                    )

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    processing_guide = f"⚠️ Literature Agent Error: {str(e)}"

            self.print_full_report(
                data, processing_guide, unique_references_list, kr_props
            )

            display_results.append(
                {
                    "Material": data.get("Input_Query"),
                    "εr (Pred)": data.get("εr_mean"),
                    "Q×f (Pred)": data.get("Qxf_mean"),
                    "τf (Pred)": data.get("TCF_mean"),
                }
            )

        return pd.DataFrame(display_results)

    def run_batch(self, csv_path):
        pass

    def run_directory(self, dir_path):
        pass
