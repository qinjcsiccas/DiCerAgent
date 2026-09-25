import requests
import re
import io
import sys
import logging
import warnings
import threading
from pymatgen.io.cif import CifParser
from pymatgen.core import Composition, Structure

# Ignore Pymatgen warnings about CIF occupancy
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
logger = logging.getLogger("COD_Rester")

class JournalAbbreviator:
    def __init__(self):
        self.abbr_dict = {}
        self.load_material_science_dict()

    def load_material_science_dict(self):
        # 1. Core dictionary: exact match
        core_journals = {
            "acta crystallographica": "Acta Cryst.",
            "acta crystallographica section a": "Acta Cryst. A",
            "acta crystallographica section b": "Acta Cryst. B",
            "acta crystallographica section c": "Acta Cryst. C",
            "acta crystallographica section e": "Acta Cryst. E",
            "zeitschrift fuer kristallographie": "Z. Kristallogr.",
            "zeitschrift fur kristallographie": "Z. Kristallogr.",
            "physical review b": "Phys. Rev. B",
            "physical review letters": "Phys. Rev. Lett.",
            "physical review materials": "Phys. Rev. Mater.",
            "journal of the american ceramic society": "J. Am. Ceram. Soc.",
            "journal of the european ceramic society": "J. Eur. Ceram. Soc.",
            "ceramics international": "Ceram. Int.",
            "nature": "Nature",
            "science": "Science",
            "nature materials": "Nat. Mater.",
            "nature communications": "Nat. Commun.",
            "scientific reports": "Sci. Rep.",
            "angewandte chemie international edition": "Angew. Chem. Int. Ed.",
            "advanced materials": "Adv. Mater.",
            "journal of modern physics": "J. Mod. Phys.",
            "american mineralogist": "Am. Mineral."
        }
        for k, v in core_journals.items():
            self.abbr_dict[k] = v

    def heuristic_abbreviate(self, name):
        """Rule-based abbreviation engine"""
        clean = re.sub(r'\b(of|the|and|in|for|section)\b', '', name, flags=re.IGNORECASE)
        replacements = [
            ("Journal", "J."), ("Physical", "Phys."), ("Chemical", "Chem."),
            ("Materials", "Mater."), ("Science", "Sci."), ("Society", "Soc."),
            ("Applied", "Appl."), ("Research", "Res."), ("Letters", "Lett."),
            ("International", "Int."), ("Ceramics", "Ceram."), ("Physics", "Phys."),
            ("Crystallography", "Cryst."), ("Engineering", "Eng."), ("Review", "Rev."),
            ("Solid", "Sol."), ("State", "St."), ("Chemistry", "Chem."),
            ("Japanese", "Jpn."), ("American", "Am."), ("European", "Eur."),
            ("Bulletin", "Bull."), ("Communications", "Commun."), ("Structure", "Struct."),
            ("Acta", "Acta"), ("Physica", "Phys.")
        ]
        for old, new in replacements:
            clean = re.sub(f"\\b{old}\\b", new, clean, flags=re.IGNORECASE)
        return " ".join(clean.split())

    def get_abbr(self, full_name):
        if not full_name or full_name == "---": return "---"
        key = full_name.lower().strip().replace(".", "").replace("  ", " ")
        if key in self.abbr_dict: return self.abbr_dict[key]
        abbr = self.heuristic_abbreviate(full_name)
        return abbr[:35] + ".." if len(abbr) > 35 else abbr

def extract_metadata(cif_text, abbreviator):
    meta = {"year": "----", "journal": "---", "sg": "---", "title": "---", "year_int": 0}
    try:
        # 1. Year
        m_year = re.search(r"_journal_year\s+(['\"]?)([\d]{4})\1", cif_text)
        if m_year: 
            meta["year"] = m_year.group(2)
            meta["year_int"] = int(m_year.group(2))
        
        # 2. Journal
        m_jour = re.search(r"_journal_name_full\s+(['\"])(.*?)\1", cif_text, re.S)
        if m_jour: 
            raw_jour = m_jour.group(2).replace("\n", " ").strip()
            meta["journal"] = abbreviator.get_abbr(raw_jour)
            
        # 3. Space Group
        m_sg = re.search(r"_symmetry_space_group_name_H-M\s+(['\"])(.*?)\1", cif_text)
        if m_sg: meta["sg"] = m_sg.group(2).strip()
        
        # 4. Title (Enhanced)
        m_title_q = re.search(r"_publ_section_title\s+(['\"])(.*?)\1", cif_text, re.S)
        m_title_s = re.search(r"_publ_section_title\s*\n;\s*(.*?)\s*;", cif_text, re.S)
        
        raw_title = ""
        if m_title_q: raw_title = m_title_q.group(2)
        elif m_title_s: raw_title = m_title_s.group(1)
            
        if raw_title:
            t = raw_title.replace("\n", " ").strip()
            # === Modification point: increased to 120 characters ===
            meta["title"] = t[:120] + "..." if len(t)>120 else t
            
    except Exception: pass
    return meta


def _input_with_timeout(prompt, timeout=10):
    """Input with timeout; returns None on timeout, otherwise the .strip()'ed string. Thread-safe."""
    result = [None]

    def _reader():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = ""

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return result[0].strip() if result[0] is not None else None


class CODRester:
    def __init__(self):
        self.abbreviator = JournalAbbreviator()

    def get_structure_interactive(self, formula):
        try:
            comp = Composition(formula)
            hill_formula = comp.hill_formula
            
            print(f"   🔎 Searching COD for formula: {hill_formula} ...")
            
            url = "https://www.crystallography.net/cod/result.php"
            params = {"formula": hill_formula, "submit": "Search"}
            
            response = requests.get(url, params=params, timeout=20)
            pattern = r'href="(\d{7})\.cif"'
            cod_ids = list(set(re.findall(pattern, response.text)))
            
            if not cod_ids:
                print(f"   ⚠️ COD Search returned 0 results for '{hill_formula}'.")
                return None, None

            # === Modification point: let the user choose how many results to show ===
            total_found = len(cod_ids)
            print(f"   🎉 Found {total_found} candidates.")
            
            limit = 15
            if total_found > 15:
                # Pause prompting
                user_choice = input(f"   👀 Show top 15 only? (Press Enter for Yes, type 'all' to fetch metadata for ALL {total_found}): ").strip().lower()
                if user_choice == 'all':
                    limit = total_found
                    print(f"   ☕ Fetching all metadata... This might take a while.")
                else:
                    print(f"   📥 Fetching top 15...")

            candidates = []
            ids_to_check = cod_ids[:limit]
            
            for cid in ids_to_check:
                cif_url = f"https://www.crystallography.net/cod/{cid}.cif"
                try:
                    cif_resp = requests.get(cif_url, timeout=5)
                    if cif_resp.status_code != 200: continue
                    
                    cif_text = cif_resp.text
                    meta = extract_metadata(cif_text, self.abbreviator)
                    
                    candidates.append({
                        "id": cid,
                        "text": cif_text,
                        "meta": meta
                    })
                    sys.stdout.write(".") 
                    sys.stdout.flush()
                except: continue
            
            print("\n")
            # Sort by year descending
            candidates.sort(key=lambda x: x['meta']['year_int'], reverse=True)

            print("   " + "="*140) # widen the separator line to fit long titles
            print(f"   {'IDX':<4} | {'COD ID':<9} | {'Year':<6} | {'SG':<12} | {'Journal':<25} | {'Title'}")
            print("   " + "-"*140)
            
            for idx, item in enumerate(candidates):
                m = item['meta']
                print(f"   {idx:<4} | {item['id']:<9} | {m['year']:<6} | {m['sg']:<12} | {m['journal']:<25} | {m['title']}")
            print("   " + "="*140)

            while True:
                user_sel = _input_with_timeout(
                    f"   📥 Select IDX (0-{len(candidates)-1}, or 'n' to skip, auto-0 in 10s): ",
                    timeout=10
                )
                if user_sel is None:
                    print(f"   ⏰ Timeout, auto-selecting IDX 0")
                    user_sel = "0"
                if user_sel.lower() == 'n':
                    return None, None

                if user_sel.isdigit():
                    sel_idx = int(user_sel)
                    if 0 <= sel_idx < len(candidates):
                        selected_item = candidates[sel_idx]
                        print(f"   🚀 Parsing Structure for COD-{selected_item['id']}...")

                        try:
                            parser = CifParser(io.StringIO(selected_item['text']), occupancy_tolerance=10.0)
                            # pymatgen version compatibility: new parse_structures / legacy get_structures
                            if hasattr(parser, 'parse_structures'):
                                structures = parser.parse_structures(primitive=False)
                            elif hasattr(parser, 'get_structures'):
                                structures = parser.get_structures(primitive=False)
                            else:
                                raise AttributeError("No parse method found on CifParser")

                            if not structures:
                                print("   ❌ Parse failed: No structure found.")
                                continue

                            s = structures[0]
                            s.properties['year'] = selected_item['meta']['year_int']

                            return s, f"COD-{selected_item['id']}"

                        except Exception as e:
                            print(f"   ❌ Parse Error: {e}. Try another one.")
                            continue

                print("   ❌ Invalid selection.")

        except Exception as e:
            print(f"   ❌ COD Error: {e}")
            return None, None