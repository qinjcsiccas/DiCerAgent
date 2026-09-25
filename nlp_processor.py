import re
import config
from openai import OpenAI

class NLPProcessor:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.client = None
        
        if self.use_llm:
            if not config.DS_API_KEY:
                print("⚠️ Warning: DS_API_KEY not found. LLM disabled.")
                self.use_llm = False
            else:
                try:
                    self.client = OpenAI(
                        api_key=config.DS_API_KEY, 
                        base_url=config.DEEPSEEK_API_URL
                    )
                except Exception as e:
                    print(f"⚠️ NLP Init Error: {e}")
                    self.use_llm = False

    def process_query(self, text):
        # 1. Preprocessing: DeepSeek may hang, so add a fallback tag first
        pre_db_tag = ""
        lower_text = text.lower()
        if "cod" in lower_text and "db=" not in lower_text: pre_db_tag = " db=cod"
        elif ("mp" in lower_text or "materials project" in lower_text) and "db=" not in lower_text: pre_db_tag = " db=mp"

        # 2. Try to use LLM
        if self.use_llm and self._is_natural_language(text):
            try:
                llm_result = self._extract_via_deepseek(text)
                if llm_result and "Error" not in llm_result:
                    return llm_result
            except Exception as e:
                print(f"⚠️ DeepSeek Error: {e}, falling back to Regex.")
        
        # 3. If LLM is off or failed, use the enhanced regex extraction (fallback)
        return self._extract_via_regex_smart(text) + pre_db_tag

    def _is_natural_language(self, text):
        # Simple heuristic: contains non-ASCII characters or is a long sentence
        if len(text.encode('utf-8')) != len(text): return True 
        if len(text.split()) > 3: return True
        return False

    def _extract_via_regex_smart(self, text):
        """
        Powerful regex extraction: extract only chemical formulas and parameters, discard non-ASCII text and filler
        """
        # 1. Extract parameters (vm=100, etc.)
        params = re.findall(r"(?:vm|p|en|pm|sp|cvd|bvs|m|vi|tt)\s*=\s*[\d\.\-]+", text, re.IGNORECASE)
        param_str = " ".join(params)
        
        # 2. Extract chemical formula (e.g. Ca0.5Sr0.5TiO3)
        # Logic: find the longest continuous string that looks like a formula (non-ASCII excluded)
        # Exclude keywords such as "COD", "MP", "db=cod"
        clean_text = re.sub(r"(?:vm|p|en|pm|sp|cvd|bvs|m|vi|tt)\s*=\s*[\d\.\-]+", "", text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\b(cod|mp|database|find|get|from|db=cod|db=mp)\b", "", clean_text, flags=re.IGNORECASE)
        
        # Match MP-ID
        mp_match = re.search(r'\b(mp-\d+)\b', clean_text, re.IGNORECASE)
        if mp_match: 
            return f"{mp_match.group(1)} {param_str}".strip()

        # Match chemical formula (starts with uppercase letter, contains letters, digits and decimal points)
        candidates = re.findall(r"([A-Z][a-z]?\d*[\d\.]*[A-Za-z0-9\(\)\.]*)", clean_text)
        
        best_formula = ""
        for cand in candidates:
            # Filter out plain words like "The", "Please" (usually short or all letters)
            if len(cand) > len(best_formula):
                best_formula = cand
        
        if best_formula:
            return f"{best_formula} {param_str}".strip()
        
        # If nothing can be extracted, return the original content (after removing parameters)
        return text.strip()

    def _extract_via_deepseek(self, text):
        system_prompt = (
            "Extract Material Formula and Intent. Output format: 'FORMULA db=SOURCE parameters'. "
            "SOURCE must be 'cod' or 'mp'. "
            "Example: 'CaTiO3 from COD' -> 'CaTiO3 db=cod'. "
            "Example: 'Predict SrTiO3' -> 'SrTiO3'. "
            "Return ONLY the formatted string."
        )
        try:
            response = self.client.chat.completions.create(
                model=config.DS_MODEL, 
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                temperature=0.1, stream=False
            )
            return response.choices[0].message.content.strip()
        except: return None
