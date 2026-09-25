import os
import warnings
import pandas as pd
import re
import torch
import math

# Set the local model save path
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLM")
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.embeddings import HuggingFaceEmbeddings

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "faiss_index")
DEFAULT_INDEX_PATH = DATA_DIR
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "wos_metadata_high_scored_literature_final.csv")

class ChemicalScorer:
    def __init__(self):
        self.REAL_ELEMENTS = {
            'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K', 'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni', 'No', 'Np', 'O', 'Og', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Ts', 'U', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr'
        }
        
        # Same-group / substitutable element groups (used for impurity-penalty exemption)
        self.HOMOLOGUE_GROUPS = [
            {'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'},           # IIA
            {'Ti', 'Zr', 'Hf'},                             # IVB
            {'V', 'Nb', 'Ta'},                              # VB
            {'Cr', 'Mo', 'W'},                              # VIB
            {'Mn', 'Tc', 'Re'},                             # VIIB
            {'Fe', 'Ru', 'Os'},                             # VIII
            {'Co', 'Rh', 'Ir'},                             # VIII
            {'Ni', 'Pd', 'Pt'},                             # VIII
            {'Cu', 'Ag', 'Au'},                             # IB
            {'Zn', 'Cd', 'Hg'},                             # IIB
            {'Al', 'Ga', 'In', 'Tl'},                       # IIIA
            {'Si', 'Ge', 'Sn', 'Pb'},                       # IVA
            # Rare earths + Y + Sc
            {'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
        ]

        # Suspected impurity list (if not in the same group as the target element, it is treated as a poison)
        self.TOXIC_CANDIDATES = {'Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Al', 'Mg', 'Pb', 'Cu', 'Zn'}
        
        # Method/review-type keywords (used to exempt penalty)
        self.METHOD_KEYWORDS = {
            "review", "overview", "progress", "advance", "perspective", "status",
            "prediction", "machine learning", "data-driven", "approach", "strategy", "computational", "dft",
            "synthesis", "fabrication", "sintering", "processing", "preparation", 
            "characterization", "microstructure", "phase evolution", "mechanism"
        }

    def is_chemical_token(self, token):
        if not token: return False
        if any(char.isdigit() for char in token): return True
        if token in self.REAL_ELEMENTS: return True
        if '-' in token:
            parts = token.split('-')
            if parts[0] in self.REAL_ELEMENTS: return True
        return False

    def extract_elements_safe(self, text):
        elements = set()
        tokens = re.findall(r'\b[A-Za-z0-9\-]+\b', text)
        for token in tokens:
            if self.is_chemical_token(token):
                raw_els = re.findall(r'[A-Z][a-z]?', token)
                valid_els = {e for e in raw_els if e in self.REAL_ELEMENTS}
                elements.update(valid_els)
        return elements

    def evaluate(self, title, query, aliases=None, base_score=0.0):
        """
        Core scoring logic v2.0
        Fix: one-vote-veto mechanism for missing key elements
        """
        title_str = str(title).strip()
        title_lower = title_str.lower()
        query_lower = query.lower().strip()
        
        score = base_score
        alias_hit = False

        # --- 1. Alias matching (highest priority) ---
        if aliases:
            for alias in aliases:
                if alias.lower() == query_lower: continue
                if alias.lower() in title_lower:
                    score += 60
                    alias_hit = True # mark: alias hit, some checks can be exempted later
                    break

        # --- 2. Basic matching ---
        if query_lower in title_lower: 
            score += 100
        elif len(query) > 4:
            base_formula = re.sub(r'^[A-Z][a-z]?[\d\.]*', '', query) 
            if len(base_formula) > 3 and base_formula.lower() in title_lower:
                score += 80 
        
        is_method_paper = any(kw in title_lower for kw in self.METHOD_KEYWORDS)
        if is_method_paper: score += 30 # lower the method-paper reward a bit, preventing generic reviews from scoring too high

        # --- 3. Element parsing and checking ---
        target_all = self.extract_elements_safe(query)
        target_cations = {e for e in target_all if e not in {'O', 'C', 'H', 'N', 'F', 'Cl'}}
        
        # Build the exemption list
        safe_relatives = set(target_cations)
        for cation in target_cations:
            for group in self.HOMOLOGUE_GROUPS:
                if cation in group:
                    safe_relatives.update(group)

        # Key feature elements (Target + strong-feature list)
        essential_elements = target_cations

        # === Key fix: one-vote-veto for missing key elements (VETO) ===
        if essential_elements:
            title_check = title_str 
            # Extend the synonym map to prevent false kills (e.g. Si -> Silicon/Silicate)
            symbol_map = {
                'Cu': ['copper', 'cuprate'], 
                'Ag': ['silver'], 
                'Au': ['gold'], 
                'Fe': ['iron', 'ferrite'], 
                'Pb': ['lead'], 
                'Sn': ['tin', 'stannate'], 
                'W':  ['tungsten', 'tungstate'],
                'Si': ['silicon', 'silicate', 'silica'],
                'Ti': ['titanium', 'titanate'],
                'Zr': ['zirconium', 'zirconate'],
                'Al': ['aluminium', 'aluminum', 'aluminate'],
                'Mg': ['magnesium', 'magnesiate'],
                'Nb': ['niobium', 'niobate'],
                'Ta': ['tantalum', 'tantalate']
            }
            
            missing_count = 0
            for el in essential_elements:
                # Check logic:
                # 1. The element symbol appears in the title (e.g. "Si")
                # 2. The English full name appears in the title (e.g. "Silicon")
                # 3. The derivative name appears in the title (e.g. "Silicate")
                
                check_terms = [el] + symbol_map.get(el, [])
                
                # A hit on any single term counts as found
                found = any(term in title_check or term.lower() in title_lower for term in check_terms)
                
                if not found:
                    missing_count += 1
            
            if missing_count > 0:
                # Exemption conditions:
                # 1. If this is a method paper (e.g. "ML for dielectric ceramics") -> penalize but do not kill
                # 2. If a specific alias is hit (e.g. "Gillespite structure") -> no penalty (the alias already implies the element)
                
                if alias_hit:
                    pass # alias hit, safe
                elif is_method_paper:
                    score -= (100 * missing_count) # method-paper penalty
                else:
                    # Neither a method nor an alias hit, and a key element is missing -> death sentence
                    # e.g. CaCuSi4O10 missing Cu -> becomes CaSiO3 (wollastonite/Ternesite), a completely different system
                    return -9999.0 

        # --- 4. Poison-impurity check ---
        active_toxic = self.TOXIC_CANDIDATES - safe_relatives
        if len(target_cations) > 0:
            title_elements = self.extract_elements_safe(title_str)
            toxic_hits = 0
            for el in title_elements:
                if el in active_toxic:
                    toxic_hits += 1
            
            if toxic_hits > 0:
                if not is_method_paper:
                    score -= (300 + 100 * toxic_hits)

        return score

class LocalRAGEngine:
    def __init__(self, index_path=None):
        self.index_path = index_path if index_path else DEFAULT_INDEX_PATH
        self.vectorstore = None
        self.is_ready = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.scorer = ChemicalScorer()
        # Dual-path union + rerank config params (aligned with the ablation run_ablation_union.py)
        self.coarse_multiplier = 20   # FAISS coarse recall = k * 20 (originally k*10)
        self.truncate_multiplier = 3  # per-path truncation = k * 3

    # ================================================================
    # Static helper methods (fully consistent with the ablation run_ablation_union.py)
    # ================================================================
    @staticmethod
    def _get_fingerprint(text):
        if not text:
            return ""
        s = str(text).lower()
        s = re.sub(r'\.[a-z]{3,4}$', '', s)
        return re.sub(r'[^a-z0-9]', '', s)

    @staticmethod
    def _guess_title(filename):
        base_name = os.path.basename(filename).replace(".pdf", "").replace("_", " ")
        return base_name if len(base_name) > 3 else "Unknown Title"

    @staticmethod
    def _get_base_score(title_guess):
        if title_guess == "Unknown Title" or "paper" in title_guess.lower():
            return 10.0
        return 40.0

    @staticmethod
    def _compute_chem_factor(raw_chem):
        """Five-segment score factor, fully consistent with the ablation"""
        if raw_chem < -200:
            return 0.01
        elif raw_chem < 0:
            return 0.3
        elif raw_chem < 50:
            return 1.0
        elif raw_chem < 100:
            return 1.5
        else:
            return 2.0

    # ================================================================
    # Dual-path union + rerank Pipeline method
    # ================================================================
    def _faiss_l2_merge(self, docs_with_scores, truncate_k):
        """Path B: FAISS L2 distance -> similarity -> title-fingerprint merge -> paper list"""
        chunks = []
        for d, dist in docs_with_scores:
            filename = d.metadata.get('source', '') or d.metadata.get('filename', 'Unknown PDF')
            title_guess = self._guess_title(filename)
            faiss_score = 1.0 / (1.0 + dist)  # smaller L2 distance = more similar, mapped to (0,1]
            chunks.append({
                "title": title_guess,
                "source": filename,
                "score": faiss_score,
            })

        chunks.sort(key=lambda x: x["score"], reverse=True)

        # title-fingerprint merge: for multiple chunks of the same paper, take the highest score x1.3
        merged = {}
        for chunk in chunks[:truncate_k]:
            title_key = self._get_fingerprint(chunk.get("title", ""))
            if not title_key:
                continue
            if title_key in merged:
                merged[title_key]["score"] = max(
                    merged[title_key]["score"], chunk["score"]
                ) * 1.3
            else:
                merged[title_key] = chunk.copy()

        result = [d for d in merged.values() if d["score"] > 0.0]
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def _chem_scorer_merge(self, docs_with_scores, query, aliases, truncate_k):
        """Path C: ChemicalScorer independent scoring -> aggregate by title fingerprint taking max -> top truncate_k"""
        chunks = []
        for d, dist in docs_with_scores:
            filename = d.metadata.get('source', '') or d.metadata.get('filename', 'Unknown PDF')
            title_guess = self._guess_title(filename)
            raw_chem = self.scorer.evaluate(
                d.page_content, query, aliases=aliases, base_score=0.0
            )
            chunks.append({
                "title": title_guess,
                "source": filename,
                "score": raw_chem,
            })

        # Aggregate by title fingerprint, take the highest chem score
        merged = {}
        for chunk in chunks:
            title_key = self._get_fingerprint(chunk.get("title", ""))
            if not title_key:
                continue
            if title_key in merged:
                merged[title_key]["score"] = max(
                    merged[title_key]["score"], chunk["score"]
                )
            else:
                merged[title_key] = chunk.copy()

        result = list(merged.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:truncate_k]

    def _union_pool(self, pool_b, pool_c):
        """Dual-path union: B union C, dedup by title fingerprint, score takes the max of the two paths"""
        union = {}
        for paper in pool_b:
            title_key = self._get_fingerprint(paper.get("title", ""))
            if not title_key:
                continue
            union[title_key] = paper.copy()

        for paper in pool_c:
            title_key = self._get_fingerprint(paper.get("title", ""))
            if not title_key:
                continue
            if title_key in union:
                union[title_key]["score"] = max(
                    union[title_key]["score"], paper["score"]
                )
            else:
                union[title_key] = paper.copy()

        return list(union.values())

    def _hybrid_rerank(self, candidates, docs_with_scores, query, aliases):
        """Rerank: for each paper in the candidate pool, concatenate full text -> ChemicalScorer -> hybrid_score"""
        # Build the chunk map: title_fp -> concatenated full text
        paper_chunks_map = {}
        for paper in candidates:
            title_fp = self._get_fingerprint(paper.get("title", ""))
            chunks_text = []
            for d, dist in docs_with_scores:
                filename = d.metadata.get('source', '') or d.metadata.get('filename', 'Unknown PDF')
                if self._get_fingerprint(self._guess_title(filename)) == title_fp:
                    chunks_text.append(d.page_content)
            paper_chunks_map[title_fp] = " ".join(chunks_text)[:10000]

        reranked = []
        for paper in candidates:
            title_fp = self._get_fingerprint(paper.get("title", ""))
            merged_text = paper_chunks_map.get(title_fp, paper.get("title", ""))
            raw_chem = self.scorer.evaluate(
                merged_text, query, aliases=aliases, base_score=0.0
            )
            chem_factor = self._compute_chem_factor(raw_chem)
            paper["hybrid_score"] = paper["score"] * chem_factor
            paper["chem_score_raw"] = raw_chem
            # title_guess chemical scoring: restore same-scale competition with CSV/OpenAlex (refer to the base-40 mechanism in the backup version)
            base_score = 40.0
            title_guess = paper.get("title", "")
            if title_guess == "Unknown Title" or "paper" in title_guess.lower():
                base_score = 10.0
            paper["title_score"] = self.scorer.evaluate(
                title_guess, query, aliases=aliases, base_score=base_score
            )
            reranked.append(paper)

        reranked.sort(key=lambda x: x["title_score"], reverse=True)
        return reranked

    def load(self):
        if self.is_ready: return
        print(f"[*] Loading Local RAG Index from: {self.index_path}...")
        if not os.path.exists(self.index_path): return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Set the local model path
                local_model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "LLM", "hub", "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots"
                )
                
                # Find the actual model folder
                model_path = self.embedding_model_name
                if os.path.exists(local_model_path):
                    snapshots_dir = local_model_path
                    if os.path.exists(snapshots_dir):
                        subdirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                        if subdirs:
                            model_path = os.path.join(snapshots_dir, subdirs[0])
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            self.vectorstore = FAISS.load_local(
                self.index_path, embeddings, allow_dangerous_deserialization=True
            )
            self.is_ready = True
            print("[+] RAG Index Loaded.")
        except Exception as e:
            print(f"[!] RAG Load Failed: {type(e).__name__}: {str(e)[:100]}")

    def search(self, query, aliases=None, k=5):
        """
        Dual-path union + rerank retrieval (aligned with the ablation run_ablation_union.py)

        Pipeline:
          FAISS coarse recall (k*20 chunks)
            |- Path B: FAISS L2 -> similarity -> top truncate_k*3 -> title-fingerprint merge -> paper list B
            |- Path C: ChemicalScorer independent scoring -> aggregate by title taking max -> top truncate_k*3 -> paper list C
            `- Union: B U C (dedup by title, score takes max)
                 `- Rerank: concatenate full text -> ChemicalScorer -> hybrid_score = score x chem_factor -> sort

        Return format fully compatible with the old version: list[dict], fields title/authors/year/journal/content/filename_raw/type/score
        """
        if not self.is_ready:
            self.load()
        if not self.vectorstore:
            return []

        try:
            clean_q = query.strip()
            COARSE_K = k * self.coarse_multiplier      # k * 20
            TRUNCATE_K = k * self.truncate_multiplier   # k * 3

            # --- 1. FAISS coarse recall (only once) ---
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                clean_q, k=COARSE_K
            )
            # docs_with_scores: [(Document, L2_distance), ...]

            # --- 2. Path B: FAISS L2 → merge ---
            pool_b = self._faiss_l2_merge(docs_with_scores, TRUNCATE_K)

            # --- 3. Path C: ChemicalScorer independent scoring ---
            pool_c = self._chem_scorer_merge(
                docs_with_scores, clean_q, aliases, TRUNCATE_K
            )

            # --- 4. Union ---
            candidates = self._union_pool(pool_b, pool_c)

            # --- 5. Rerank ---
            reranked = self._hybrid_rerank(
                candidates, docs_with_scores, clean_q, aliases
            )

            # --- 6. Assemble the output (format fully compatible with the old version) ---
            # Build a fast lookup map: title_fp -> first chunk content and path
            chunk_lookup = {}
            for d, dist in docs_with_scores:
                filename = d.metadata.get('source', '') or d.metadata.get('filename', 'Unknown PDF')
                fp = self._get_fingerprint(self._guess_title(filename))
                if fp not in chunk_lookup:
                    chunk_lookup[fp] = {
                        "content": d.page_content.replace("\n", " "),
                        "filename": filename,
                    }

            results = []
            for paper in reranked[:k * 3]:
                fp = self._get_fingerprint(paper.get("title", ""))
                lookup = chunk_lookup.get(fp, {})
                content = lookup.get("content", "")
                filename = lookup.get("filename", paper.get("source", ""))

                # The output score uses the title_guess chemical score, at the same scale as CSV/OpenAlex scoring (full-text docs can compete for top-20)
                output_score = paper.get("title_score", paper.get("chem_score_raw", paper["score"]))

                results.append({
                    "title": paper["title"],
                    "authors": ["Local PDF"],
                    "year": "n.d.",
                    "journal": "Local Repository",
                    "content": content[:5000],
                    "filename_raw": filename,
                    "type": "Local PDF",
                    "score": output_score,
                })

            return results

        except Exception as e:
            print(f"[!] RAG Search Error: {e}")
            return []

class LocalCSVEngine:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path if csv_path else DEFAULT_CSV_PATH
        self.df = None
        self.fingerprint_index = {} 
        self.is_ready = False
        self.scorer = ChemicalScorer()

    def _get_fingerprint(self, text):
        if not text: return ""
        s = str(text).lower()
        s = re.sub(r'\.[a-z]{3,4}$', '', s)
        return re.sub(r'[^a-z0-9]', '', s)

    def load(self):
        if self.is_ready: return
        print(f"[*] Loading Metadata CSV: {self.csv_path}...")
        if not os.path.exists(self.csv_path): return
        try:
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            
            # fillna handling
            obj_cols = self.df.select_dtypes(include=['object']).columns
            self.df[obj_cols] = self.df[obj_cols].fillna('')
            num_cols = self.df.select_dtypes(include=['number']).columns
            self.df[num_cols] = self.df[num_cols].fillna(0)
            
            if 'Similarity_Score' in self.df.columns:
                self.df['Similarity_Score'] = pd.to_numeric(self.df['Similarity_Score'], errors='coerce').fillna(0)
            
            fname_col = None
            for col in ['Matched_Filename', 'filename', 'File Name', 'PDF Name']:
                if col in self.df.columns:
                    fname_col = col
                    break
            
            if fname_col:
                for idx, row in self.df.iterrows():
                    raw_name = str(row[fname_col])
                    if len(raw_name) > 3:
                        fp_name = self._get_fingerprint(raw_name)
                        self.fingerprint_index.setdefault(fp_name, []).append(row)
                    
                    raw_title = str(row.get('Title', ''))
                    if len(raw_title) > 5:
                        fp_title = self._get_fingerprint(raw_title)
                        self.fingerprint_index.setdefault(fp_title, []).append(row)

            self.is_ready = True
            print("[+] CSV Loaded & Indexed.")
        except Exception as e:
            print(f"[!] CSV Load Failed: {e}")

    def match_metadata(self, rag_filename, rag_title=None):
        if not self.is_ready: self.load()
        candidates = []
        rag_fp_name = self._get_fingerprint(rag_filename)
        if rag_fp_name in self.fingerprint_index:
            candidates.extend(self.fingerprint_index[rag_fp_name])
        if rag_title:
            rag_fp_title = self._get_fingerprint(rag_title)
            if rag_fp_title in self.fingerprint_index:
                candidates.extend(self.fingerprint_index[rag_fp_title])
        if not candidates: return None
        return candidates[0]

    def search(self, query, aliases=None, limit=50):
        if not self.is_ready: self.load()
        if self.df is None or self.df.empty: return []
        try:
            candidates = self.df.copy() 
            def row_scorer(row):
                title = str(row.get('Title', ''))
                
                # === Key fix: CSV score normalization ===
                # The raw Similarity_Score can be as high as 450000+
                # We no longer trust its absolute value; instead apply extreme compression
                # Logic: if raw > 0, give it a small base score (0-20), only as a slight advantage under identical chemical matches
                raw_sim = float(row.get('Similarity_Score', 0))
                
                # Log Scaling + Cap
                # Even if raw_sim is 1,000,000 -> log10 is 6 -> times 2 = 12 points
                # Thus CSV relevance can only fine-tune, never dominate
                if raw_sim > 1:
                    sim_base = math.log10(raw_sim) * 3.0 
                else:
                    sim_base = 0.0
                
                # Hard cap at 30 points
                sim_base = min(sim_base, 30.0)

                return self.scorer.evaluate(title, query, aliases=aliases, base_score=sim_base)

            candidates['Search_Score'] = candidates.apply(row_scorer, axis=1)
            candidates = candidates[candidates['Search_Score'] > -200]
            candidates.sort_values(by='Search_Score', ascending=False, inplace=True)
            
            top_k = candidates.head(limit) 
            results = []
            for _, row in top_k.iterrows():
                raw_authors = str(row.get('Authors', ''))
                authors_list = [a.strip() for a in raw_authors.split(";")] if ";" in raw_authors else [raw_authors]
                
                
                doi_val = str(row.get('DOI', row.get('doi', ''))).strip()
                if doi_val.lower() == 'nan': doi_val = ""

                results.append({
                    "title": str(row.get('Title', 'No Title')),
                    "authors": authors_list,
                    "year": str(row.get('Publication Year', 'n.d.')).replace(".0", ""),
                    "journal": str(row.get('Source Title', row.get('Journal Abbreviation', 'Unknown'))),
                    "content": str(row.get('Abstract', 'No Abstract'))[:2000],
                    "source": "Local Metadata CSV",
                    "filename_raw": str(row.get('Matched_Filename', '')),
                    "doi": doi_val,
                    "type": "Metadata",
                    "score": row.get('Search_Score', 0)
                })
            return results
        except Exception as e:
            print(f"[!] CSV Search Error: {e}")
            return []