import streamlit as st
import requests
import re
import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Note: set_page_config has already been set in app_unified.py; do not set it again here

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config


def normalize_frac_formula(s):
    """Normalize fractional coefficients in composition text (e.g. La1/2, Sn1/3) into
    decimal forms parseable by pymatgen. pymatgen does not accept '1/2'/'1/3' fraction forms,
    and Composition('(La1/2Pr1/2)2...') raises '/2/2 is an invalid formula!'.
    Use full-precision floats (.17g) so the subsequent Fraction(float).limit_denominator
    can still exactly recover the minimal integer ratio (e.g. 1/3 -> 0.3333333333333333 -> Fraction(1,3)).

    In the data source, a fractional coefficient may be immediately followed by a redundant decimal
    (e.g. Eu1/30.33 is actually Eu1/3 + a redundant 0.33); first normalize via the "unit digit of the denominator followed by a digit/decimal point" glued pattern and drop the redundant
    decimal segment, to prevent the ordinary fraction regex from greedily reading 1/3+0.33 as 1/30.
    """
    def _rep(m):
        num, den = int(m.group(2)), int(m.group(3))
        return m.group(1) + ("%.17g" % (num / den))

    # Parenthesis repair: the data source may drop a left parenthesis (e.g. La0.2Gd0.2Y0.2Yb0.2Er0.2)2(...)),
    # restore the opening '(' of the group corresponding to the orphaned right ')', avoiding pymatgen's ')2 is an invalid formula!' error.
    _stack = []
    _orphan = []
    for _i, _ch in enumerate(s):
        if _ch == "(":
            _stack.append(_i)
        elif _ch == ")":
            if _stack:
                _stack.pop()
            else:
                _orphan.append(_i)
    for _pos in sorted(_orphan, reverse=True):
        _start = s.rfind(")", 0, _pos)
        if _start != -1:
            _start += 1
            # Skip the group coefficient of the previous group (e.g. the '2' in (A0.5B0.5)2),
            # the parenthesis insertion point should be at the element start of the new group
            while _start < _pos and s[_start] in "0123456789.":
                _start += 1
        else:
            _start = 0
        s = s[:_start] + "(" + s[_start:]

    # Consecutive writing where a fractional coefficient is immediately followed by a redundant decimal (Eu1/30.33 -> Eu0.3333333333333333)
    s = re.sub(r"([A-Za-z])(\d+)/(\d)(?=[.\d])", _rep, s)
    # Normalize ordinary fractional coefficients (La1/2, Sn1/3, etc.)
    s = re.sub(r"([A-Za-z])(\d+)/(\d+)", _rep, s)
    # Glued-digit fallback repair: two complete decimals directly concatenated (e.g. 0.33333333333333330.33) take the first;
    # malformed digit-pairs with a dot in between (e.g. 0.33.44) take the first.
    s = re.sub(r"(\d+\.\d+)(\d+\.\d+)", r"\1", s)
    s = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1", s)
    return s


def comp_str_to_structure(comp_str, attach_orig_formula=True):
    """Convert the composition text into a minimal pymatgen Structure (element ratios consistent with the formula).

    Supports fractional coefficient forms (La1/2, Sn1/3, etc.), normalized to decimals before parsing;
    place atoms by the minimal integer ratio of the formula's elements (Fraction least-common-multiple),
    to avoid round dropping decimals like 0.4 to 1 and distorting the ratios.
    Attach the _orig_formula attribute (normalized original parenthesized-form string),
    so plugins can parse A/B sites via getattr(structure, '_orig_formula', None).
    """
    from pymatgen.core import Composition, Structure
    from fractions import Fraction
    import math

    _norm_str = normalize_frac_formula(str(comp_str).strip())
    comp = Composition(_norm_str)
    reduced = comp.reduced_composition
    fracs = {
        el.symbol: Fraction(float(amt)).limit_denominator(1000000)
        for el, amt in reduced.items()
    }
    lcm = 1
    for f in fracs.values():
        lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
    species = []
    coords = []
    i = 0
    for el, amt in reduced.items():
        n = int(fracs[el.symbol] * lcm)
        for _k in range(n):
            species.append(el.symbol)
            coords.append([(i % 5) * 0.2, ((i // 5) % 5) * 0.2, (i // 25) * 0.2])
            i += 1
    structure = Structure(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        species, coords, coords_are_cartesian=True,
    )
    if attach_orig_formula:
        try:
            structure._orig_formula = _norm_str
        except Exception:
            pass
    return structure


def patch_tree_sklearn_compat(model_data):
    """Recursively add the monotonic_cst attribute to tree-based models, compatible with sklearn>=1.4 deserialization.

    DecisionTree/RandomForest trained by older sklearn (e.g. 1.3.2) lack monotonic_cst after serialization,
    and newer sklearn (1.8.0) raises an error when deserializing them.
    Works for a list / dict / a single estimator.
    """
    if isinstance(model_data, dict):
        for v in model_data.values():
            patch_tree_sklearn_compat(v)
    elif isinstance(model_data, (list, tuple)):
        for v in model_data:
            patch_tree_sklearn_compat(v)
    elif model_data is not None and type(model_data).__name__ in (
        "DecisionTreeRegressor", "DecisionTreeClassifier",
        "ExtraTreeRegressor", "ExtraTreeClassifier",
        "RandomForestRegressor", "RandomForestClassifier",
        "GradientBoostingRegressor", "GradientBoostingClassifier",
    ):
        if not hasattr(model_data, "monotonic_cst"):
            try:
                model_data.monotonic_cst = None
            except Exception:
                pass
        if hasattr(model_data, "estimators_"):
            patch_tree_sklearn_compat(list(model_data.estimators_))
    return model_data


st.markdown(
    """
<style>
    .stApp { background-color: #f8f9fa; }
    h1 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; color: #2c3e50; }
    .convert-box { background-color: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e9ecef; margin-top: 20px; }
    .success-box { background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .error-box { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .info-box { background-color: #d1ecf1; border-color: #bee5eb; color: #0c5460; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .code-output { background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 13px; max-height: 500px; overflow: auto; }
    .pdf-tag { display: inline-block; background: #e74c3c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
    .folder-tag { display: inline-block; background: #3498db; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
    .data-tag { display: inline-block; background: #27ae60; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
    .code-tag { display: inline-block; background: #9b59b6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

DEEPSEEK_API_KEY = config.DS_API_KEY
DEEPSEEK_API_URL = config.DEEPSEEK_API_URL

EXTERNAL_MODELS_DIR = os.path.join(current_dir, "external_models")


def extract_python_code(response: str) -> str:
    """Extract Python code from the AI response"""
    if not response:
        return ""

    code_block_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    code_block_match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
        if "import " in code or "def " in code or "class " in code:
            return code

    lines = response.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            start_idx = i
            break

    if start_idx > 0:
        return "\n".join(lines[start_idx:]).strip()

    return response.strip()


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF"""
    try:
        import PyPDF2

        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except:
            return None


def call_deepseek(prompt, context="", retry_count=3):
    """Call the DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return (
            None,
            "[!] DeepSeek API Key not configured. Please set DS_API_KEY in settings.ini",
        )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    full_prompt = prompt
    if context:
        full_prompt = f"""## Reference Paper Content
Please analyze the following content to understand the model:

{context}

---

## Original Request
{prompt}"""

    payload = {
        "model": config.DS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional materials science ML engineer. Your task is to convert external prediction models into standardized plugin code with retraining capability. IMPORTANT: Output ONLY the Python code, without any explanatory text.",
            },
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.3,
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(
                f"{DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"], None
            elif response.status_code == 429:
                import time

                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None, f"[!] API Error: {response.status_code}"
        except Exception as e:
            if attempt < retry_count - 1:
                import time

                time.sleep(3)
                continue
            return None, f"[!] Request failed: {str(e)}"

    return None, "[!] Max retries exceeded"


def call_deepseek_for_analysis(prompt, retry_count=2):
    """Call the DeepSeek API for analysis (different from code generation)"""
    if not DEEPSEEK_API_KEY:
        return None, "[!] DeepSeek API Key not configured"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.DS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a materials science ML expert. Analyze the provided information and output ONLY the analysis results in the specified format. No extra explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(
                f"{DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"], None
            elif response.status_code == 429:
                import time

                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None, f"[!] API Error: {response.status_code}"
        except Exception as e:
            if attempt < retry_count - 1:
                import time

                time.sleep(3)
                continue
            return None, f"[!] Request failed: {str(e)}"

    return None, "[!] Max retries exceeded"


def analyze_model_config(paper_context, user_code, training_data_info):
    """Use AI to analyze model and suggest configuration"""

    ANALYZER_PROMPT = """You are a materials science ML expert. Analyze the provided model and training data to suggest optimal configuration.

## Your Task
Based on the paper/code and training data, determine:

1. **Property**: What is being predicted? (e.g., melting point, band gap, conductivity)
2. **Property Key**: Short identifier for the property (e.g., "Tm", "Eg", "sigma")
3. **Property Symbol**: Display symbol (e.g., "Tm", "E_g", "σ")
4. **Property Unit**: Physical unit (e.g., "K", "eV", "S/cm")
5. **Target Column**: Which column in the training data is the target variable?
6. **Stage 1 Features**: Which columns represent elemental composition (e.g., H, Li, Ca, O)?
7. **Stage 2 Features**: Which additional columns are used? (e.g., na, d, fepa)
8. **Is Two-Stage?**: Does the model use a two-stage (cascaded) architecture?
9. **Scope/Applicability**: What materials does this model apply to?
   - Compound types: oxides, fluorides, sulfides, nitrides, etc.
   - Structure types: perovskite (ABO3), spinel (AB2O4), ABO4, general, etc.
   - Example: "Oxides, general structure" or "Perovskite ABO3 oxides only"
10. **Model Notes**: Any special requirements or notes for implementation.

## Training Data Columns
{training_data_info}

## Paper/Code Context
{paper_context}
{user_code}

## Output Format
Provide your analysis in this JSON-like format (but plain text):
```
PROPERTY: [What is being predicted]
KEY: [Short key like Tm, Eg]
SYMBOL: [Display symbol]
UNIT: [Physical unit]
TARGET_COLUMN: [Column name in data]
STAGE1_FEATURES: [Comma-separated list of element columns]
STAGE2_FEATURES: [Comma-separated list of additional columns]
TWO_STAGE: [YES or NO]
SCOPE: [Applicability description - compound types, structure types]
NOTES: [Any special requirements]
```

If you cannot determine something, use "UNKNOWN".
"""

    prompt = ANALYZER_PROMPT.format(
        training_data_info=training_data_info or "No training data provided",
        paper_context=paper_context or "No paper provided",
        user_code=user_code or "No code provided",
    )

    response, error = call_deepseek_for_analysis(prompt)

    if error:
        print(f"[!] AI Analysis Error: {error}")
        return None

    if not response:
        print("[!] AI returned empty response")
        return None

    # Parse the response
    config = {}
    lines_found = 0
    for line in response.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            lines_found += 1

            if key == "PROPERTY":
                config["property"] = value
            elif key == "KEY":
                config["key"] = value
            elif key == "SYMBOL":
                config["symbol"] = value
            elif key == "UNIT":
                config["unit"] = value
            elif key == "TARGET_COLUMN":
                config["target_column"] = value
            elif key == "STAGE1_FEATURES":
                features = [f.strip() for f in value.split(",") if f.strip()]
                config["stage1_features"] = features
            elif key == "STAGE2_FEATURES":
                features = [f.strip() for f in value.split(",") if f.strip()]
                config["stage2_features"] = features
            elif key == "TWO_STAGE":
                config["two_stage"] = value.upper() == "YES"
            elif key == "SCOPE":
                config["scope"] = value
            elif key == "NOTES":
                config["notes"] = value

    if lines_found == 0:
        print(f"[!] AI Response parsing failed. Response preview: {response[:200]}")
        return None

    print(f"[OK] AI parsed {lines_found} config lines")
    return config


def build_constants_info(constant_files, plugin_name=None):
    """Build a prompt segment from the user-tagged constant-file contents.

    Constant files are lookup tables needed for feature calculation (e.g. the element-attribute table Table_S2.csv),
    which differ from the training dataset: they do not participate in training but are used by the LLM when generating a plugin,
    so the plugin can compute feature quantities from formula + constants even when only a formula is given.
    Constant files are automatically copied next to the plugin on save
    to plugins/external/<plugin_name>_constants/; the plugin should read them with a module-level read_csv,
    and must not hand-hard-code partial tables (hard-coding easily drops rows/mangles column names and breaks lookups).
    Returns (constants_info, error): constants_info is the prompt-injected text, error is the read-failure message.
    """
    if not constant_files:
        return "", None
    parts = []
    if plugin_name:
        parts.append(
            "## Constant Files Deployment (automatic)\n"
            "When the plugin is saved, the constant files listed below are "
            "automatically copied to:\n"
            f"`plugins/external/{plugin_name}_constants/`\n"
            "(same folder as the plugin). The generated plugin MUST load them at "
            "module level with pd.read_csv, e.g.:\n"
            "```\n"
            "import os, pandas as pd\n"
            f"_CONST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{plugin_name}_constants')\n"
            "CONSTANT_TABLE = pd.read_csv(os.path.join(_CONST_DIR, '<FILE_NAME>'))\n"
            "```\n"
            "Use these constants to compute features from the chemical formula alone. "
            "Do NOT hardcode a partial copy of the table as the single source of "
            "truth (it loses rows/column names and breaks lookups)."
        )
    parts.append(
        "## User-Provided Constant Files\n"
        "The following files are CONSTANT LOOKUP TABLES provided by the user. "
        "They are NOT training data - they contain physical/elemental constants needed to "
        "compute features from a chemical formula. They are auto-deployed next to the "
        "plugin (see above); the generated plugin MUST load them via pd.read_csv at "
        "module level (see above code snippet), NOT hardcode a partial copy."
    )
    errors = []
    for cf in constant_files:
        try:
            name = cf.get("name", "")
            path = cf.get("path", "")
            if not os.path.exists(path):
                errors.append(f"{name}: file not found")
                continue
            if name.endswith(".csv"):
                df = pd.read_csv(path)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                errors.append(f"{name}: unsupported format")
                continue
            parts.append(
                f"\n### Constant file: {name}\n"
                f"- Columns: {list(df.columns)}\n"
                f"- Rows: {len(df)}\n"
                f"- First rows:\n{df.head(20).to_string()}\n"
            )
        except Exception as e:
            errors.append(f"{name}: {e}")
    if errors:
        return "\n".join(parts), "; ".join(errors)
    return "\n".join(parts), None


def deploy_constant_files(constant_files, plugin_path):
    """Automatically copy the user constant files into the plugin-level <plugin_name>_constants/ directory.

    This makes the generated/saved plugin self-contained (constant tables included); at prediction time the user no longer needs
    to place files manually or rely on the original upload path at runtime. Returns (deployment description text, error).
    """
    if not constant_files:
        return "", None
    plugin_dir = os.path.dirname(os.path.abspath(plugin_path))
    plugin_stem = os.path.splitext(os.path.basename(plugin_path))[0]
    const_dir = os.path.join(plugin_dir, plugin_stem + "_constants")
    try:
        os.makedirs(const_dir, exist_ok=True)
    except Exception as e:
        return "", f"failed to create constants dir {const_dir}: {e}"
    errors = []
    deployed = []
    for cf in constant_files:
        try:
            name = cf.get("name", "")
            path = cf.get("path", "")
            if not path or not os.path.exists(path):
                errors.append(f"{name}: source file not found: {path}")
                continue
            dst = os.path.join(const_dir, name)
            shutil.copy2(path, dst)
            deployed.append(dst)
        except Exception as e:
            errors.append(f"{cf.get('name','')}: {e}")
    if not deployed:
        return "", ("; ".join(errors) if errors else "no constant files deployed")
    info = (
        f"Constant files auto-deployed to `{const_dir}`:\n"
        + "\n".join(f"- {os.path.basename(p)}" for p in deployed)
    )
    return info, ("; ".join(errors) if errors else None)


def detect_file_type(filename, content_preview=""):
    """Automatically detect the file type"""
    ext = os.path.splitext(filename)[1].lower()
    name_lower = filename.lower()

    if ext in [".py", ".r", ".m", ".matlab", ".cpp", ".c", ".f", ".f90"]:
        if ext == ".py":
            return "code", "Python"
        elif ext in [".r", ".m", ".matlab"]:
            return "code", ext[1:].upper()
        else:
            return "code", "Code"

    if ext in [".pdf"]:
        content_lower = content_preview.lower()
        if "supporting" in name_lower or "supplementary" in name_lower:
            return "supporting", "Supporting Info"
        return "paper", "Paper"

    if ext in [".csv", ".xlsx", ".xls", ".txt"]:
        if ext in [".csv", ".txt"]:
            try:
                if len(content_preview) > 0:
                    return "data", "Training Data"
            except:
                pass
        return "data", "Data"

    if ext in [
        ".pkl",
        ".pickle",
        ".joblib",
        ".model",
        ".h5",
        ".pb",
        ".pt",
        ".pth",
        ".onnx",
        ".rds",
        ".rda",
    ]:
        return "model", "Model File"

    return "other", "Other"


def scan_folder(folder_path):
    """Scan a folder and detect file types"""
    files_info = []

    if not os.path.exists(folder_path):
        return files_info

    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path):
            content_preview = ""
            try:
                with open(full_path, "rb") as file:
                    content_preview = file.read(2000).decode("utf-8", errors="ignore")
            except:
                pass

            file_type, type_label = detect_file_type(f, content_preview)
            files_info.append(
                {
                    "name": f,
                    "path": full_path,
                    "type": file_type,
                    "type_label": type_label,
                    "size": os.path.getsize(full_path),
                }
            )

    return files_info


def detect_model_file_info(path):
    """Probe model-file information: file size, serialization format (joblib/pickle), top-level structure (dict/list/other)

    Injected into the conversion prompt to help the LLM generate loading code that matches the real model file.
    """
    info = {"path": path, "size": os.path.getsize(path)}
    try:
        data = joblib.load(path)
        info["format"] = "joblib"
        if isinstance(data, dict):
            info["top_level"] = "dict"
            info["keys"] = list(data.keys())
        elif isinstance(data, list):
            info["top_level"] = "list"
            info["len"] = len(data)
        else:
            info["top_level"] = type(data).__name__
    except Exception:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            info["format"] = "pickle"
            if isinstance(data, dict):
                info["top_level"] = "dict"
                info["keys"] = list(data.keys())
            elif isinstance(data, list):
                info["top_level"] = "list"
                info["len"] = len(data)
            else:
                info["top_level"] = type(data).__name__
        except Exception as e:
            info["format"] = "unknown"
            info["error"] = str(e)
    return info


def build_model_files_info(folders):
    """Scan the given folder for .pkl/.joblib model files and generate a description text for prompt injection"""
    lines = []
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith((".pkl", ".joblib")):
                full = os.path.join(folder, f)
                info = detect_model_file_info(full)
                size_mb = info["size"] / 1048576.0
                line = (
                    f"- {full} (size={size_mb:.1f}MB, "
                    f"format={info.get('format', '?')}, "
                    f"top-level={info.get('top_level', '?')})"
                )
                lines.append(line)
                if "keys" in info:
                    lines.append(f"    dict keys: {info['keys']}")
                if "len" in info:
                    lines.append(f"    list length: {info['len']}")
                if "error" in info:
                    lines.append(f"    load error: {info['error']}")
    if not lines:
        return "No model files (.pkl/.joblib) detected in the target folders."
    return "\n".join(lines)


# ============================================================
# Two-branch conversion core (faithful to the original model; hard-coded default templates are forbidden)
# Branch A: ready-made model files exist under external_models/<model_name>/ -> reuse them, only adapting format/interface, no retraining
# Branch B: no ready-made model files -> LLM reads the original training code + literature to extract a training-config JSON, then retrains from that config
# ============================================================

MODEL_FILE_EXTS = (
    ".pkl", ".joblib", ".pt", ".pth", ".h5", ".hdf5",
    ".onnx", ".sav", ".pkl.gz", ".joblib.gz", ".npy", ".npz",
)

TRAINING_CONFIG_START = "<!--TRAINING_CONFIG_START-->"
TRAINING_CONFIG_END = "<!--TRAINING_CONFIG_END-->"


def detect_model_files(folder_path):
    """Probe whether ready-made model files exist under external_models/<model_name>/.

    Returns {"branch": "A"|"B", "files": [structured model-file info], "text": description text injected into the prompt}
    - Branch A: at least one model file exists (.pkl/.joblib/.pt/.h5 etc.)
    - Branch B: no model files at all; requires LLM config extraction + retraining
    """
    files = []
    if folder_path and os.path.isdir(folder_path):
        for f in sorted(os.listdir(folder_path)):
            full = os.path.join(folder_path, f)
            if not os.path.isfile(full):
                continue
            low = f.lower()
            if low.endswith(MODEL_FILE_EXTS):
                info = {"name": f, "path": full, "size": os.path.getsize(full)}
                if low.endswith((".pkl", ".joblib", ".sav")):
                    # Skip probing large files to avoid slowing down the flow
                    if info["size"] > 200 * 1048576:
                        info["format"] = "unknown"
                        info["top_level"] = "too_large_to_probe"
                    else:
                        try:
                            data = joblib.load(full)
                            info["format"] = "joblib"
                            if isinstance(data, dict):
                                info["top_level"] = "dict"
                                info["keys"] = list(data.keys())[:30]
                            elif isinstance(data, list):
                                info["top_level"] = "list"
                                info["len"] = len(data)
                            else:
                                info["top_level"] = type(data).__name__
                        except Exception as e:
                            info["format"] = "unknown"
                            info["error"] = str(e)[:200]
                else:
                    info["format"] = "native"
                    info["top_level"] = "unknown (non-pickle format)"
                files.append(info)
    branch = "A" if files else "B"
    text_lines = [f"## Conversion Branch: {branch}"]
    if branch == "A":
        text_lines.append(
            "Detected existing model file(s) in the source folder. REUSE them (Branch A). "
            "Do NOT retrain. Do NOT fall back to a default RandomForest template."
        )
        for f in files:
            line = (
                f"- {f['name']} (size={f['size'] / 1048576.0:.1f}MB, "
                f"format={f.get('format', '?')}, top-level={f.get('top_level', '?')})"
            )
            if "keys" in f:
                line += f", keys={f['keys']}"
            if "len" in f:
                line += f", len={f['len']}"
            if "error" in f:
                line += f", load-error={f['error']}"
            text_lines.append(line)
    else:
        text_lines.append(
            "No existing model file detected in the source folder. Branch B: read the "
            "Original Model Code and Paper below, extract the REAL algorithm / hyper-"
            "parameters / preprocessing / feature engineering / data split / target "
            "column, and emit a structured TRAINING_CONFIG JSON. Retrain with that "
            "config. Never fall back to a hardcoded RandomForest default."
        )
    return {
        "branch": branch,
        "files": files,
        "text": "\n".join(text_lines),
    }


def build_branch_info(model_detection, source_folder, plugin_dir, plugin_name):
    """Build the branch-info text injected into the conversion prompt.

    Branch A: provide the model-file list, format-probe results, and the final load path after copying;
    Branch B: emphasize that the training-config JSON must be extracted from code/literature; default fallback is forbidden.
    """
    branch = model_detection["branch"]
    lines = [f"## Conversion Branch: {branch}"]
    if branch == "A":
        lines.append(
            "Existing model file(s) detected (Branch A). MUST reuse the model files below; "
            "do NOT retrain and do NOT use a default RandomForest template."
        )
        lines.append(
            "Model files will be copied verbatim into the plugin subdirectory "
            "(plugins/external/<plugin_name>/). Build load paths as listed:"
        )
        for f in model_detection["files"]:
            target = os.path.join(plugin_dir, f["name"])
            line = (
                f"- {f['name']} (size={f['size'] / 1048576.0:.1f}MB, "
                f"format={f.get('format', '?')}, top-level={f.get('top_level', '?')})"
            )
            if "keys" in f:
                line += f", keys={f['keys']}"
            if "len" in f:
                line += f", len={f['len']}"
            if "error" in f:
                line += f", load-error={f['error']}"
            lines.append(line)
            lines.append(f"  -> final load path: {target}")
        lines.append(
            "The Predictor MUST joblib.load from that path and adapt to the ACTUAL top-level "
            "layout (dict / plain list / single estimator) revealed by the probe. If the file "
            "is missing or fails to load, predict() must return a dict with a 'note' field "
            "carrying the reason; never silently return 0.0."
        )
    else:
        lines.append(
            "No existing model file detected (Branch B). MUST read 'Original Model Code' and "
            "'Paper/Supporting Information' to extract the REAL algorithm name, hyperparameters, "
            "preprocessing steps, feature columns, target column and data split. Output the "
            "structured TRAINING_CONFIG JSON (see TRAINING_CONFIG section) and generate plugin "
            "code whose training logic is driven by that config. It is FORBIDDEN to fall back to "
            "a hardcoded RandomForest default; if the algorithm cannot be determined, the code "
            "must raise / report an explicit error instead of silently training a default model."
        )
    return "\n".join(lines)


def extract_training_config(response_text):
    """Extract the training-config JSON (TRAINING_CONFIG marker block) from the LLM response.

    Returns (config_dict, None) on success; (None, error_str) on failure.
    Branch B must raise an explicit error on parse failure; default-model fallback is forbidden.
    """
    import json as _json

    if not response_text:
        return None, "LLM response is empty, cannot extract training config"
    start = response_text.find(TRAINING_CONFIG_START)
    end = response_text.find(TRAINING_CONFIG_END)
    if start == -1 or end == -1 or end <= start:
        return (
            None,
            "TRAINING_CONFIG marker block not found in LLM response; training config extraction failed"
            "(Branch B must output the training config JSON after the code; default model fallback is forbidden)",
        )
    block = response_text[start + len(TRAINING_CONFIG_START):end].strip()
    if not block:
        return None, "TRAINING_CONFIG marker block is empty"
    if block.startswith("```"):
        block = block.strip("`").strip()
        if block.startswith("json"):
            block = block[4:].strip()
        elif block.startswith("JSON"):
            block = block[4:].strip()
    try:
        cfg = _json.loads(block)
    except Exception as e:
        return None, f"Failed to parse training config JSON: {e}"
    if not isinstance(cfg, dict):
        return None, "Top level of training config JSON must be an object"
    if not cfg.get("algorithm"):
        return (
            None,
            "Training config missing 'algorithm' field (Branch B must give the algorithm name; default model fallback is forbidden)",
        )
    return cfg, None


def resolve_sklearn_estimator(algorithm):
    """Resolve the estimator class from sklearn modules by algorithm name; return None if not found."""
    import importlib

    algo = (algorithm or "").strip()
    if not algo:
        return None
    modules = [
        "sklearn.svm", "sklearn.ensemble", "sklearn.tree",
        "sklearn.linear_model", "sklearn.neighbors",
        "sklearn.neural_network", "sklearn.gaussian_process",
        "sklearn.kernel_ridge", "sklearn.discriminant_analysis",
        "sklearn.naive_bayes", "sklearn.cross_decomposition",
        "sklearn.multioutput", "sklearn.semi_supervised",
        # Third-party algorithm libraries (external models commonly use xgboost / lightgbm / catboost)
        "xgboost", "lightgbm", "catboost",
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, algo, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    lower = algo.lower()
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            for name in dir(mod):
                if name.lower() == lower:
                    return getattr(mod, name)
        except Exception:
            continue
    return None


def build_preprocessing_pipeline(preprocessing, estimator):
    """Build a sklearn Pipeline from the config.

    preprocessing supports: ["standard_scaler"] / ["min_max_scaler"] / ["robust_scaler"] / ["none"]
    Returns a Pipeline or a bare estimator.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    steps = []
    for p in (preprocessing or []):
        key = (p or "").strip().lower()
        if key in ("standardscaler", "standard_scaler", "std", "standardize"):
            steps.append(("scaler", StandardScaler()))
        elif key in ("minmaxscaler", "min_max_scaler", "minmax", "min-max"):
            steps.append(("scaler", MinMaxScaler()))
        elif key in ("robustscaler", "robust_scaler", "robust"):
            steps.append(("scaler", RobustScaler()))
        elif key in ("none", "null", "", "no"):
            continue
        else:
            raise ValueError(f"Unsupported preprocessing step: {p}")
    if not steps:
        return estimator
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_with_config(X, y, feature_cols, target_col, config, n_iterations, model_path,
                      progress_cb=None):
    """Branch B: config-driven training core.

    X: numeric feature matrix (n_samples, n_features); column order consistent with feature_cols
    y: target array
    config: training-config JSON dict extracted by the LLM
    n_iterations: ensemble iteration count (bootstrap)
    model_path: final dump path
    Returns a save_data dict; raises an explicit ValueError on parse failure / unknown algorithm (default fallback is forbidden).
    """
    import inspect
    import json as _json
    import sklearn as _sklearn

    algorithm = (config.get("algorithm") or "").strip()
    hyperparameters = config.get("hyperparameters") or {}
    preprocessing = config.get("preprocessing") or []
    split = config.get("split") or {}
    if isinstance(split, dict):
        train_ratio = float(split.get("train_ratio", 0.7))
        random_state = split.get("random_state", 42)
    else:
        train_ratio, random_state = 0.7, 42

    estimator_cls = resolve_sklearn_estimator(algorithm)
    if estimator_cls is None:
        raise ValueError(
            f"Cannot resolve algorithm '{algorithm}' from sklearn (training config extraction unusable,"
            "default model fallback is forbidden). Check the algorithm name in the model code/literature and retry."
        )

    # Hyperparameter filtering: keep only the parameters accepted by this estimator
    try:
        sig = inspect.signature(estimator_cls.__init__)
        valid_params = set(sig.parameters.keys())
        hparams = {
            k: v for k, v in hyperparameters.items()
            if k in valid_params and k != "self"
        }
    except Exception:
        hparams = dict(hyperparameters or {})

    n_samples = len(y)
    if n_samples == 0:
        raise ValueError("Training data is empty, cannot train")
    train_ratio = max(0.1, min(0.99, train_ratio))

    all_models = []
    all_preds = []
    for i in range(n_iterations):
        if progress_cb:
            progress_cb(i / n_iterations)
        rng = np.random.RandomState(
            (random_state + i) if random_state is not None else i
        )
        train_idx = rng.choice(
            n_samples, size=int(train_ratio * n_samples), replace=True
        )
        model = build_preprocessing_pipeline(
            preprocessing, estimator_cls(**hparams)
        )
        model.fit(X[train_idx], y[train_idx])
        all_preds.append(model.predict(X))
        all_models.append(model)

    if progress_cb:
        progress_cb(1.0)

    save_data = {
        "models": all_models,
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "algorithm": algorithm,
        "hyperparameters": hparams,
        "preprocessing": preprocessing,
        "n_iterations": n_iterations,
        "train_ratio": train_ratio,
        "sklearn_version": _sklearn.__version__,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # sklearn version compatibility: add monotonic_cst to all trees before dump
    patch_tree_sklearn_compat(save_data)
    joblib.dump(save_data, model_path)
    return save_data


def copy_existing_models(model_files, plugin_dir):
    """Branch A: copy ready-made model files from the source folder into the plugin subdirectory (preserving original file names).

    Returns {"copied": [target paths...], "errors": [errors...]}
    """
    os.makedirs(plugin_dir, exist_ok=True)
    copied, errors = [], []
    for f in model_files:
        src = f["path"]
        dst = os.path.join(plugin_dir, f["name"])
        try:
            shutil.copy2(src, dst)
            copied.append(dst)
        except Exception as e:
            errors.append(f"{f['name']}: {e}")
    return {"copied": copied, "errors": errors}


CONVERSION_PROMPT_TEMPLATE = '''You are a materials science ML expert. Convert external prediction models into DiCerAgent plugin format.

CRITICAL: OUTPUT ONLY THE PYTHON CODE. NO EXPLANATIONS.

## HARD CONSTRAINTS (MANDATORY - GENERATED PLUGIN MUST SATISFY ALL)
The generated plugin is UNUSABLE unless ALL of the following hard constraints hold. Violating any of them is a conversion FAILURE:
a) Feature-driven, NEVER hardcoded placeholders: get_required_features() MUST return the REAL numeric feature columns - from the loaded model's feature_cols (Branch A) or from the TRAINING_CONFIG feature_cols (Branch B). Each feature MUST be calculated from the parsed composition/structure (e.g. composition-weighted average of elemental properties from the constant table Element_Property_Table.csv). NEVER hardcode placeholder element-symbol names (e.g. ['H','Li','Ca','O']) as if they were property column names.
b) Element-property missing-value semantics - distinguish THREE cases (MUST NOT raise for case i):
   (i) SOME elements lack a value in a property column (e.g. O/Sr have no 'Bulk' modulus in Element_Property_Table.csv - physically undefined, a NORMAL data gap, NOT an error). Follow the training-side convention (Features.py: props.fillna(props.median(numeric_only=True))): load the table and fill NaN with the column MEDIAN (numeric-only) BEFORE the weighted average, so a single element missing a property NEVER raises. In per-element lookup, treat NaN as "skip this element / use the pre-filled median" (pd.isna check / dict.get with default) - NEVER raise "Element 'X' has no value for property 'Y'".
   (ii) A feature column has NO value for ALL constituent elements (median is also NaN): ONLY THEN raise an explicit error at calculate/predict time.
   (iii) Column-name resolution failure: when a feature column name does not exactly match a column in Element_Property_Table.csv, implement an explicit alias map or fuzzy match (e.g. Density <-> 'Density (g/mL)', ionization <-> '1st ionization potential (kJ/mol)', Bulk <-> 'Bulk'). If a required feature column still cannot be resolved to a table column, the plugin MUST raise an explicit error at calculate/predict time - NEVER silently return 0.0.
c) Output key convention: predict() MUST return keys of the form <prop_key>_mean / <prop_key>_dev (e.g. tec_mean/tec_dev, er_mean/er_dev, melt_temp_mean/melt_temp_dev). The legacy 'property_mean'/'property_dev' keys are DEPRECATED and MUST NOT be used.
d) Mandatory self-check: before outputting the plugin, run an end-to-end self-check with at least TWO different sample compositions (e.g. BaTiO3 and SrTiO3 for ABO3): import the plugin, call calculate() then predict(), and verify (i) output keys follow (c); (ii) values are NOT all zero; (iii) values DIFFER between different compositions (identical outputs mean the features are broken/zero); (iv) magnitudes are physically plausible. Any failure means the plugin is broken - fix the code before outputting it.
e) User-provided constant files are AUTO-DEPLOYED and MUST be read via read_csv: if the "Constant Files Deployment" / "User-Provided Constant Files" section below lists constant lookup tables (e.g. element property tables), the framework automatically copies them next to the plugin (plugins/external/<plugin_name>_constants/). The generated plugin MUST load them AT MODULE LEVEL with pd.read_csv (see the code snippet in the deployment section), e.g.:
   import os, pandas as pd
   _CONST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '<plugin_name>_constants')
   CONSTANT_TABLE = pd.read_csv(os.path.join(_CONST_DIR, '<FILE_NAME>'))
   Do NOT hardcode a partial copy of the table as the single source of truth - partial copies lose rows and column names (e.g. missing Hf/Eu/Sn rows, or 'velocity' vs 'Velocity Of Sound (m/s)') and break lookups. If the deployment section is ABSENT (no constant files), the plugin may embed a table from the paper/code as a module-level constant instead. In all cases NEVER let the plugin fail with "No numeric features found!" just because the input is only a chemical formula: derive numeric features from the formula + constants.
f) No "No numeric features found!" failure: when the plugin receives only a chemical formula (structure), calculate() MUST still produce the full numeric feature vector. If the model's features are elemental-property-derived, compute them from composition.get_el_amt_dict() (mole fractions) times the embedded constant table (composition-weighted average). Missing values follow constraint (b). NEVER raise "No numeric features found!" inside calculate().
g) EXACT reproduction of the original feature formula: when 'Original Model Code' below defines a specific feature-engineering recipe (e.g. A/B-site separated weighting like Sr_a - Sr_b, cation-only standard deviation, specific elemental property columns and data sources), the generated FeatureCalculator.calculate() MUST reproduce THAT EXACT recipe - same element grouping (A-site vs B-site vs cation vs total), same weightings, same standard-deviation denominators, same property table column mapping. Do NOT silently substitute a generic whole-composition weighted average for a site-separated formula. If the original code parses the formula into A/B sites (e.g. (A)2(B)2O7), replicate that parsing logic inside calculate() from the composition so the numeric features match the original code's output for the same formula.
h) A/B-site parsing from the Structure object: pymatgen's structure.composition.formula / reduced_formula NEVER contains parentheses, so a regex like r'\((.*?)\)(\d*\.?\d*)' applied to those strings will ALWAYS fail and raise 'Cannot parse A-site'. In calculate(), to recover the original A/B grouping, FIRST try: orig = getattr(structure, '_orig_formula', None); if orig is not None and '(' in orig, parse THAT string (it keeps the user's original parentheses, e.g. (Y0.2Gd0.2Er0.2Yb0.2Lu0.2)2Zr2O7) with the original regex. If orig is missing OR contains NO '(' (e.g. BaTiO3, SrTiO3 - plain ABO3 formulas), you MUST derive the sites from structure.composition.get_el_amt_dict() using the VERBATIM fallback template below - it uses a physically-motivated site-assignment rule (B-site = typical B cations Zr/Ti/Hf/Sn/Nb/Ta/W/Mo/Mn, A-site = remaining cations; single-cation oxides split equally) and NEVER raises for any cation-containing oxide. NEVER apply a parenthesis regex to structure.composition.formula / reduced_formula strings. The generated calculate() MUST include this fallback and use it whenever the formula string has no parentheses:

    def _split_a_b_sites(amt_dict):
        """amt_dict = structure.composition.get_el_amt_dict().
        Returns (a_amt, b_amt) as dicts {element: amount}, each separately
        normalized to relative fractions. NEVER raises for cation-containing input.
        """
        cations = {el: amt for el, amt in amt_dict.items() if el != "O"}
        if not cations:
            raise ValueError("Cannot parse A-site: no cation in composition")
        B_CANDIDATES = ("Zr", "Ti", "Hf", "Sn", "Nb", "Ta", "W", "Mo", "Mn")
        b_els = [el for el in cations if el in B_CANDIDATES]
        if b_els:
            b_amt = {el: cations[el] for el in b_els}
            a_amt = {el: amt for el, amt in cations.items() if el not in b_els}
            if not a_amt:  # single-cation case: put the smallest B cation into A
                min_el = min(b_amt, key=b_amt.get)
                a_amt = {min_el: b_amt.pop(min_el)}
        else:
            order = sorted(cations.items(), key=lambda kv: kv[1])
            if len(order) == 1:  # simple binary oxide MO
                a_amt = dict(order)
                b_amt = dict(order)
            else:
                b_el, b_v = order[0]
                b_amt = {b_el: b_v}
                a_amt = {el: amt for el, amt in order[1:]}

        def _norm(d):
            s = sum(d.values()) or 1.0
            return {k: v / s for k, v in d.items()}

        return _norm(a_amt), _norm(b_amt)

For BaTiO3/SrTiO3 this fallback yields a_amt={Ba|Sr: 1.0}, b_amt={Ti: 1.0}; the smoke check REQUIRES these formulas to compute successfully.
i) NEVER load data inside __init__: file I/O (pd.read_csv / pd.read_excel / open() / np.load / joblib.load / torch.load, etc.) is FORBIDDEN inside __init__ or any other method. All constant tables MUST be loaded AT MODULE LEVEL (top of file, exactly as in constraint e) and __init__ MUST only reference them via lightweight assignment (e.g. self.feature_table = CONSTANT_TABLE). Putting read_csv inside __init__ is a conversion FAILURE: the training branch instantiates FeatureCalculator() with a stub __file__ and no deployed constants dir yet, and the smoke check only sees module-level DataFrames - both break with confusing runtime errors (e.g. "None of ['Element'] are in the columns" when reading a non-constant CSV, or identical outputs because no module-level table was found). NEVER read any file inside __init__.

j) Scalar input features MUST support dual-source resolution (base_features first, structure attribute second, default last): when the model has scalar input features that are NOT derived from composition (e.g. temperature, pressure, time), calculate(structure, base_features) MUST resolve them in this order: (1) from the base_features dict if the key is present (e.g. 'temperature'); (2) from a structure attribute if present (e.g. hasattr(structure, 'temperature') -> float(structure.temperature)); (3) a physically sensible default (e.g. 300.0 K) ONLY as the last resort. NEVER implement a single-source lookup: callers disagree on where scalars live - some pass them via base_features, some attach them to the structure object. A single-source implementation silently forces every sample to the default (e.g. all samples evaluated at 300K while the dataset spans 5K-1832K), which breaks temperature-dependent predictions WITHOUT raising any error and can only be caught by full-dataset feature-level comparison. Single-source scalar resolution is a conversion FAILURE.

k) Count-type composition features MUST be derived from the REDUCED formula, NEVER from the input structure's absolute cell size: features that count atoms per formula (e.g. na = number of atoms, and elemental composition columns expressed as absolute atom counts like Ca=1, O=1) MUST be computed from Composition(structure.composition.reduced_formula) - i.e. one formula unit - and NOT from structure.composition (the actual supercell/primitive cell), because the SAME material can be passed as Ca2O2 / Ca4O4 / mp-2605's 4-atom cell etc., and using the cell's absolute atom count silently changes na (e.g. 2 -> 40) and the element columns with cell size, breaking the trained model's input distribution without raising any error. This failure is only catchable by full-dataset feature-level comparison (e.g. all samples pinned to a wrong cell size). ALWAYS: red_comp = Composition(structure.composition.reduced_formula); use red_comp.get_el_amt_dict() for element columns and red_comp.num_atoms for na. Cell-dependent properties that ARE legitimately cell-based (density, volume) keep using the structure. Count-from-cell is a conversion FAILURE.

l) Materials-Project-sourced features (fepa, density, band_gap, etc.) MUST be fetched by REAL NETWORK CALLS at calculate/predict time - caching is NEVER a substitute:
   (i) The generated plugin MUST include a runtime MP-API fetch path (mp_api.client.MPRester) for EVERY external-property feature present in the training columns, with the API key read from settings.ini (configparser, utf-8) - NEVER hardcode 'YOUR_MP_KEY' or any literal key.
   (ii) Fetch by material_id FIRST when an mp_id is available; otherwise by reduced_formula. When multiple candidates match, pick the most STABLE one (min energy_above_hull) - request fields=['formation_energy_per_atom','energy_above_hull', ...] so the sort key is actually present. NEVER blindly take docs[0] without sorting.
   (iii) EMPTY result list / missing key / API exception MUST NOT be silently mapped to 0.0: 0.0 is a VALID-looking but WRONG value for fepa/density and silently corrupts predictions. Return None and let the caller treat the feature as MISSING (stage-degrade or explicit error), or raise an explicit error. Silent 0.0 fallback is a conversion FAILURE.
   (iv) A local cache is ALLOWED only as an optional speed-up BEHIND a global switch (e.g. USE_LOCAL_CACHE) that defaults to OFF in validation/testing. NEVER make the cache the primary or only source. Hardcoding cached values into the plugin source is a conversion FAILURE.
   (v) The self-check (constraint d) MUST exercise the REAL network path at least once (e.g. BaTiO3 -> fetch fepa/density from MP API), not just a mocked/stubbed value.

## STEP 0: BRANCH DECISION (MOST IMPORTANT - NEVER FALL BACK TO DEFAULTS)
{{branch_info}}

Branch A (model files exist): reuse the existing model file(s) only. Generate LOADING/ADAPTATION code:
- Do NOT include any training/retraining logic in the plugin (training is skipped).
- Load exactly the paths listed above via joblib.load; adapt to the ACTUAL top-level layout.
- Keep feature calculator / predict() aligned with the model's real input features.

Branch B (no model files): generate plugin code whose training logic is CONFIG-DRIVEN.
- After the plugin code, emit the structured training configuration JSON inside:
<!--TRAINING_CONFIG_START-->
{...}
<!--TRAINING_CONFIG_END-->
- The JSON MUST contain: algorithm (exact sklearn estimator class name), hyperparameters (dict),
  preprocessing (list, e.g. ["standard_scaler"]), feature_cols (list of numeric training columns),
  target_col, split ({"method": "random", "train_ratio": 0.9, "random_state": 42}).
- Extract these from the REAL Original Model Code and Paper below. If the algorithm cannot be
  determined, the plugin MUST raise an explicit error (or return a note) - it is FORBIDDEN to
  silently fall back to a default RandomForest.
- The training code in the generated plugin MUST instantiate the estimator from the config and
  save a dict {"models": [...], "feature_cols": [...], "target_col": ..., "algorithm": ...} to
  model.pkl (single-stage) or two_stage_model.pkl (two-stage, with models_stage1/models_stage2).


## STEP 1: ANALYZE TRAINING DATA (MOST IMPORTANT!)
Look at the training data columns FIRST. The column names tell you EXACTLY what features are used.
- If you see columns like: H, Li, Ca, O, ... -> elemental composition (atomic number)
- If you see: na, d, fepa, blsd -> specific features defined in paper
- DO NOT invent features that are NOT in the training data!

## STEP 2: READ PAPER FOR FEATURE CALCULATION
After identifying features from data, read paper to understand HOW to calculate each feature:
- Is it from composition? (structure.composition)
- Is it from structure? (density, lattice, etc.)
- Is it from DFT? (formation_energy from Materials Project)
- Is it a CONSTANT or DERIVED from other features?

## STEP 3: GENERATE ACCURATE CODE
Only implement features that are ACTUALLY USED in the model.

## Training Data Analysis (KEY!)
{{training_data_info}}

{{constants_info}}

## Common Feature Patterns:
- Element columns (H, Li, Ca, O, etc.): atomic NUMBER = Composition(structure.composition.reduced_formula).get_el_amt_dict()[elem] (see constraint k - MUST use the reduced formula, NEVER the cell's absolute counts)
- na: Number of atoms = Composition(structure.composition.reduced_formula).num_atoms (one formula unit - see constraint k)
- d: Density = structure.density
- fepa: Formation energy per atom -> FROM MATERIALS PROJECT API
- blsd, blnpv: Bond length stats -> from neighbor analysis

## For Materials Project Features (like fepa):
Use mp-api to fetch formation energy (REAL network call; see constraint l):
```python
from mp_api.client import MPRester
api_key = <read MP_API_KEY from settings.ini, utf-8, never hardcode>
with MPRester(api_key) as mpr:
    docs = mpr.materials.summary.search(
        formula=formula,
        fields=["formation_energy_per_atom", "energy_above_hull"]  # MUST include the sort key
    )
    if docs:
        # multiple candidates: pick the MOST STABLE (min energy_above_hull), NEVER blindly docs[0]
        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
        features['fepa'] = doc.formation_energy_per_atom
    else:
        features['fepa'] = None  # MISSING - NEVER 0.0 (silent 0.0 corrupts predictions)
```

## Framework Template
```python
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin, register_property_info
from plugins.registry import register_plugin

metadata = ModelMetadata(
    name="{{model_name}}",
    version="1.0.0",
    author="{{author}}",
    description="Converted from external model",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

ELEMENT_COLS = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 
                'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Cu', 
                'Zn', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Ag', 
                'Cd', 'Te', 'I', 'Cs', 'Ba', 'W', 'Re', 'Tl', 'Pb']

class YourFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    
    def calculate(self, structure, base_features):
        features = base_features.copy()
        composition = structure.composition
        
        # STEP 1: Elemental composition (from data columns - atomic numbers)
        el_amt = composition.get_el_amt_dict()
        for elem in ELEMENT_COLS:
            features[elem] = el_amt.get(elem, 0)  # atomic number
        
        # STEP 2: Calculate ONLY features that exist in training data
        # Based on paper analysis:
        
        # na (number of atoms) - if in training data
        if 'na' in self.get_feature_cols():
            features['na'] = composition.num_atoms
        
        # d (density) - if in training data
        if 'd' in self.get_feature_cols():
            features['d'] = structure.density
        
        # fepa (formation energy per atom) - FROM MATERIALS PROJECT, REAL network call (constraint l)
        if 'fepa' in self.get_feature_cols():
            # NEVER hardcode the API key: read from settings.ini via self._get_mp_api_key()
            try:
                from mp_api.client import MPRester
                comp_str = composition.reduced_formula
                with MPRester(self._get_mp_api_key()) as mpr:
                    docs = mpr.materials.summary.search(
                        formula=comp_str,
                        fields=["formation_energy_per_atom", "energy_above_hull"]
                    )
                    if docs:
                        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
                        features['fepa'] = doc.formation_energy_per_atom
                    else:
                        features['fepa'] = None  # MISSING - NEVER 0.0
            except:
                features['fepa'] = None  # MISSING - NEVER 0.0 (silent 0.0 corrupts predictions)
        
        # Only implement other features if they were in training data!
        
        return features
    
    def _get_mp_api_key(self):
        """Read MP_API_KEY from settings.ini - NEVER hardcode a literal key (constraint l)."""
        import configparser
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.ini')
        with open(config_path, 'r', encoding='utf-8') as f:
            config.read_file(f)
        return config.get('DEFAULT', 'MP_API_KEY', fallback='')
    
    def get_required_base_features(self):
        return ["Mass", "Va", "P_MLR_pv"]
    
    def get_feature_cols(self):
        # This should be loaded from model file - placeholder here
        return []

class YourPredictor(BasePredictor):
    metadata = metadata
    
    def __init__(self):
        plugin_file = os.path.abspath(__file__)
        plugin_name = os.path.splitext(os.path.basename(plugin_file))[0]
        parent_dir = os.path.dirname(plugin_file)
        self.model_dir = os.path.join(parent_dir, plugin_name)
        
        self.prop_key = "{{prop_key}}"
        self.prop_symbol = "{{prop_symbol}}"
        self.prop_unit = "{{prop_unit}}"
        self._load_models()
    
    def _load_models(self):
        self.models_loaded = False
        try:
            model_path = os.path.join(self.model_dir, "two_stage_model.pkl")
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                self.models_stage1 = data["models_stage1"]
                self.models_stage2 = data["models_stage2"]
                self.stage1_cols = data.get("stage1_cols", [])
                self.stage2_cols = data.get("stage2_cols", [])
                self.prop_key = data.get("prop_key", self.prop_key)
                self.prop_symbol = data.get("prop_symbol", self.prop_symbol)
                self.prop_unit = data.get("prop_unit", self.prop_unit)
                register_property_info(self.prop_key, self.prop_symbol, self.prop_unit)
                self.models_loaded = True
        except Exception as e:
            print(f"[!] Load error: {e}")
    
    def predict(self, features):
        if not self.models_loaded:
            return {f"{self.prop_key}_mean": 0.0, f"{self.prop_key}_dev": 0.0}
        try:
            stage1_feats = np.array([[features.get(f, 0) for f in self.stage1_cols]])
            preds = []
            for i in range(len(self.models_stage1)):
                sp = self.models_stage1[i].predict(stage1_feats)[0]
                stage2_feats = []
                for f in self.stage2_cols:
                    if f == 'stage1_pred':
                        stage2_feats.append(sp)
                    else:
                        stage2_feats.append(features.get(f, 0))
                stage2_feats = np.array([stage2_feats])
                pred = self.models_stage2[i].predict(stage2_feats)[0]
                preds.append(pred)
            return {f"{self.prop_key}_mean": float(np.mean(preds)), f"{self.prop_key}_dev": float(np.std(preds))}
        except Exception as e:
            print(f"[!] Prediction error: {e}")
            return {f"{self.prop_key}_mean": 0.0, f"{self.prop_key}_dev": 0.0}

register_plugin(Plugin(
    id="{{plugin_id}}",
    structure_type="general",
    feature_calculator=YourFeatureCalculator(),
    predictor=YourPredictor()
))
```

## Paper Context
{{paper_context}}

## Original Model Code
{{user_code}}

## Framework Interface - TWO-STAGE MODEL
```python
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin

metadata = ModelMetadata(
    name="Model Name",
    version="1.0.0",
    author="Author",
    description="Two-stage model with retraining support",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

class YourFeatureCalculator(BaseFeatureCalculator):
    """Calculate features for stage 1 (composition-based)
    IMPORTANT: Use existing base_features from DiCerAgent!
    """
    metadata = metadata
    
    def calculate(self, structure, base_features):
        """Stage 1: Use base_features from DiCerAgent feature calculator
        DO NOT calculate new features - just use base_features dict!
        """
        return base_features
    
    def get_required_base_features(self):
        # USE THESE EXACT NAMES - they are pre-calculated by DiCerAgent:
        return ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"]

class Stage2FeatureCalculator(BaseFeatureCalculator):
    """Calculate features for stage 2 (structure + stage1 output)"""
    metadata = metadata
    
    def calculate(self, structure, base_features):
        """Stage 2: Use base_features + stage1 prediction
        """
        features = {}
        # Include all base features + stage1 output
        return features
    
    def get_required_base_features(self):
        return []

class YourPredictor(BasePredictor):
    """Two-stage predictor with joint retraining"""
    metadata = metadata
    
    def __init__(self):
        self.model_stage1 = None
        self.model_stage2 = None
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage2 = StandardScaler()
        
        # Model save paths: subdirectory named after the plugin (e.g. plugins/external/<plugin_name>/)
        plugin_file = os.path.abspath(__file__)
        plugin_name = os.path.splitext(os.path.basename(plugin_file))[0]
        parent_dir = os.path.dirname(plugin_file)
        self.model_dir = os.path.join(parent_dir, plugin_name)
        self.model1_path = os.path.join(self.model_dir, "stage1_model.pkl")
        self.model2_path = os.path.join(self.model_dir, "stage2_model.pkl")
        self.scaler1_path = os.path.join(self.model_dir, "stage1_scaler.pkl")
        self.scaler2_path = os.path.join(self.model_dir, "stage2_scaler.pkl")
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if exist"""
        for path in [self.model1_path, self.model2_path]:
            if not os.path.exists(path):
                return
        try:
            self.model_stage1 = joblib.load(self.model1_path)
            self.model_stage2 = joblib.load(self.model2_path)
            self.scaler_stage1 = joblib.load(self.scaler1_path)
            self.scaler_stage2 = joblib.load(self.scaler2_path)
        except:
            pass
    
    def _extract_stage_features(self, df, feature_cols):
        """Extract stage 1 features (composition-based)"""
        return df[feature_cols].values
    
    def _extract_stage2_features(self, df, stage1_pred_col, feature_cols):
        """Extract stage 2 features (base + stage1 output)"""
        stage2_feats = df[feature_cols].copy()
        stage2_feats[stage1_pred_col] = df[stage1_pred_col]
        return stage2_feats.values
    
    def train(self, features_df, target_values, stage1_feature_cols, stage2_add_feature_cols=None):
        """Two-stage training:
        1. Train stage 1 model on composition features
        2. Use stage1 predictions + structure features to train stage 2
        """
        # Stage 1: Train on composition features
        X1 = self._extract_stage_features(features_df, stage1_feature_cols)
        y = np.array(target_values)
        
        X1_scaled = self.scaler_stage1.fit_transform(X1)
        self.model_stage1 = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_stage1.fit(X1_scaled, y)
        
        # Get stage 1 predictions
        stage1_pred = self.model_stage1.predict(X1_scaled)
        
        # Add stage 1 prediction to features for stage 2
        features_df_copy = features_df.copy()
        features_df_copy['stage1_pred'] = stage1_pred
        
        # Stage 2: Train on base features + stage1 prediction
        all_stage2_cols = stage1_feature_cols + (stage2_add_feature_cols or []) + ['stage1_pred']
        X2 = features_df_copy[all_stage2_cols].values
        
        X2_scaled = self.scaler_stage2.fit_transform(X2)
        self.model_stage2 = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_stage2.fit(X2_scaled, y)
        
        # Save all models
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model_stage1, self.model1_path)
        joblib.dump(self.model_stage2, self.model2_path)
        joblib.dump(self.scaler_stage1, self.scaler1_path)
        joblib.dump(self.scaler_stage2, self.scaler2_path)
    
    def predict(self, features):
        """Two-stage prediction"""
        if self.model_stage1 is None or self.model_stage2 is None:
            return {"property_mean": 0.0, "property_dev": 0.0}
        
        # Stage 1: Get intermediate prediction
        stage1_X = np.array([[features.get(f, 0) for f in self.get_stage1_features()]])
        stage1_scaled = self.scaler_stage1.transform(stage1_X)
        stage1_pred = self.model_stage1.predict(stage1_scaled)[0]
        
        # Stage 2: Include stage1 prediction in features
        all_features = self.get_all_features()
        stage2_feats = np.array([[features.get(f, 0) for f in all_features]])
        stage2_feats[0][len(all_features)-1] = stage1_pred  # Add stage1 output
        
        stage2_scaled = self.scaler_stage2.transform(stage2_feats)
        final_pred = self.model_stage2.predict(stage2_scaled)[0]
        
        return {"property_mean": float(final_pred), "property_dev": 0.0}
    
    # REQUIRED: Must implement get_required_features() for plugin system
    def get_required_features(self):
        """Required by BasePredictor - returns all features needed"""
        return self.get_all_features()
    
    def get_stage1_features(self):
        """Features for stage 1 (use DiCerAgent base_features)"""
        return ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"]
    
    def get_stage2_features(self):
        """Features for stage 2 (stage1 features + stage1 prediction)"""
        return ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev", "stage1_pred"]
    
    def get_all_features(self):
        """All features needed for final prediction"""
        return self.get_stage2_features()

register_plugin(Plugin(
    id="your_model_id",
    structure_type="general",
    feature_calculator=YourFeatureCalculator(),
    predictor=YourPredictor()
))
```

## Requirements
1. Output ONLY the Python code - no explanations
2. MUST include get_required_features() method - THIS IS REQUIRED BY THE PLUGIN SYSTEM
3. Branch A (model files exist): training is SKIPPED. The converter copies the model file(s) into plugins/external/<plugin_name>/; your code MUST load from the exact paths listed in {{branch_info}} and adapt to the actual layout.
   Branch B (no model files): the training system dumps ONE file inside the SUBDIRECTORY named after the plugin: plugins/external/<plugin_name>/two_stage_model.pkl (two-stage) or plugins/external/<plugin_name>/model.pkl (single-stage). Build the subdirectory path with: plugin_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]; self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_name)
4. Load models with joblib.load. The file top-level layout varies: DICT (Branch B single-stage keys: models/feature_cols/target_col/algorithm; two-stage keys: models_stage1/models_stage2/stage1_cols/stage2_cols) OR PLAIN LIST of estimators (raw single-stage dump). Inspect the top-level type with isinstance() and adapt; NEVER assume a fixed layout, NEVER call .get() on a list.
5. If the model file is missing or loading fails, predict() MUST NOT silently return 0.0; return a dict with a "note" field carrying the reason instead, e.g. {"{{prop_key}}_mean": None, "{{prop_key}}_dev": None, "note": "<reason>"}.
6. In predict(): use all loaded models for ensemble prediction (mean + std)
7. Return format: {"{{prop_key}}_mean": value, "{{prop_key}}_dev": std}
8. IMPORTANT: read MP_API_KEY from settings.ini using configparser with utf-8 encoding
9. Include get_required_features() method that returns the feature list
10. End with register_plugin() call
11. sklearn version compatibility: the model file may be trained with an older sklearn (e.g. 1.3.2) whose trees lack the monotonic_cst attribute required by newer sklearn (e.g. 1.8.0) during unpickling. After joblib.load, the Predictor MUST apply a compatibility patch before predicting: walk all tree estimators (dict/list nesting supported) and set monotonic_cst=None when missing. Standard pattern:
    def _patch_sklearn_compat(model_data):
        if isinstance(model_data, dict):
            for v in model_data.values(): _patch_sklearn_compat(v)
        elif isinstance(model_data, (list, tuple)):
            for v in model_data: _patch_sklearn_compat(v)
        elif hasattr(model_data, 'estimators_'):
            if not hasattr(model_data, 'monotonic_cst'):
                try: model_data.monotonic_cst = None
                except Exception: pass
            for est in model_data.estimators_: _patch_sklearn_compat(est)
        elif type(model_data).__name__ in ('DecisionTreeRegressor','DecisionTreeClassifier','ExtraTreeRegressor','ExtraTreeClassifier','RandomForestRegressor','RandomForestClassifier','GradientBoostingRegressor','GradientBoostingClassifier'):
            if not hasattr(model_data, 'monotonic_cst'):
                try: model_data.monotonic_cst = None
                except Exception: pass
    Call _patch_sklearn_compat(model_data) right after joblib.load, BEFORE any predict() call. Do NOT catch-and-swallow this patch; if patching raises, propagate the error into the "note" field.
12. Feature alignment: the training data must use ONLY numeric feature columns that exactly match get_required_features(). NEVER mix text/non-numeric columns (e.g. Composition/formula strings) into features even after fillna(0) — that changes n_features and breaks predict(). If a column cannot be converted to numeric (all NaN after pd.to_numeric(errors='coerce')), exclude it from training features. In calculate(), produce EXACTLY the features listed in get_required_features(), with the SAME ORDER and SAME COUNT as the trained model (model.n_features_in_).
13. FORBIDDEN: never fall back to a hardcoded RandomForest default template. Branch A must reuse the existing model file; Branch B must use the algorithm extracted from code/paper via TRAINING_CONFIG. If Branch B training config is missing/invalid, the plugin code should raise an explicit error when training.
14. Branch B: your plugin's training code MUST be config-driven — read the algorithm/hyperparameters/preprocessing/feature_cols/target_col from TRAINING_CONFIG (or from a JSON file next to the plugin) and instantiate the sklearn estimator accordingly. Never hardcode RandomForestRegressor(n_estimators=100, max_features=3, max_leaf_nodes=150) as a substitute.

## Branch Decision & Available Model Files
{{branch_info}}

## Raw Model File Probe (auto-detected from the target folders)
{{model_files_info}}

## Training Data Available
{{training_data_info}}

## Model Configuration
- Property Key: {{prop_key}}
- Property Symbol: {{prop_symbol}}
- Property Unit: {{prop_unit}}
- Ensemble Iterations: {{n_ensembles}}

## Special Requirements from User
{{model_notes}}

## Original Model Code (may contain multiple files for multi-step model)
{{user_code}}

## Paper/Supporting Information
{{paper_context}}

## IMPORTANT: Feature Column Detection
Analyze the training data columns to identify:
- stage1_feature_cols: features used in step 1 (composition-based, like H, Li, Ca, O)
- stage2_add_feature_cols: additional features in step 2 (like na, d, fepa)
- target column: the final property to predict

## TRAINING_CONFIG JSON (Branch B ONLY - REQUIRED OUTPUT)
If Branch B, AFTER the plugin code block, output the training configuration JSON wrapped in:
<!--TRAINING_CONFIG_START-->
{ "algorithm": "SVR", "hyperparameters": {"C": 3.1623, "epsilon": 0.01, "gamma": 0.001},
  "preprocessing": ["standard_scaler"], "feature_cols": ["H", "O", "na", "d"],
  "target_col": "TEC", "split": {"method": "random", "train_ratio": 0.9, "random_state": 42},
  "two_stage": false }
<!--TRAINING_CONFIG_END-->
Rules:
- algorithm MUST be an exact sklearn estimator class name (e.g. SVR, RandomForestRegressor, GradientBoostingRegressor, KernelRidge, MLPRegressor, PLSRegression). Extract it from the real code/paper.
- hyperparameters MUST match the values found in the original code (e.g. C/gamma/epsilon grids). Do NOT invent defaults.
- preprocessing is a list; use ["standard_scaler"] only if the original code standardizes.
- feature_cols are the numeric columns used to train (from training data / code).
- split: use the real train/test ratio and random_state from the original code.
- two_stage: true ONLY if the original code has a two-stage pipeline (stage1 features -> stage1 prediction -> stage2 features). In that case also output stage1_algorithm / stage2_algorithm / stage2_hyperparameters / stage2_preprocessing when they differ between stages; otherwise omit them.
- The converter parses this JSON and drives retraining with it. A missing/invalid JSON means conversion FAILS with an explicit error - never a default model.

## Training Data Format (Branch B config-driven)
The training system (Branch B) will:
1. Use feature_cols from TRAINING_CONFIG to select numeric columns (fillna(0)).
2. Apply preprocessing steps from TRAINING_CONFIG (e.g. StandardScaler inside a Pipeline).
3. Instantiate the estimator named by algorithm with hyperparameters.
4. Bootstrap-sample the data train_ratio times (default 0.7) and train n_iterations models.
5. Save models in: two_stage_model.pkl with keys models_stage1/models_stage2/stage1_cols/stage2_cols (two-stage) OR model.pkl with keys models/feature_cols/target_col/algorithm (single-stage).

## How to Calculate Features in calculate()
Based on training data columns, implement:
- Element columns: structure.composition.get_el_amt_dict()[elem] (atomic numbers)
- na: composition.num_atoms
- d: structure.density
- fepa: Fetch from Materials Project API using mp-api

## MP API Integration (if fepa / density / other MP-sourced features are in data)
REAL network calls only - caching is an optional speed-up behind USE_LOCAL_CACHE (default OFF). See constraint l.
Use this pattern to read API key and fetch MP properties:
```python
# Global cache switch: False = always real network; True = optional offline speed-up
USE_LOCAL_CACHE = False

def _get_mp_api_key(self):
    import configparser
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.ini')
    with open(config_path, 'r', encoding='utf-8') as f:
        config.read_file(f)
    return config.get('DEFAULT', 'MP_API_KEY', fallback='')

def _fetch_fepa(self, composition, mp_id=None):
    """fepa: mp_id first, else reduced_formula. Stable candidate first (min energy_above_hull).
    Returns float, or None when MISSING - NEVER 0.0 (0.0 silently corrupts predictions)."""
    from mp_api.client import MPRester
    api_key = self._get_mp_api_key()
    if not api_key:
        return None
    with MPRester(api_key) as mpr:
        if mp_id:
            docs = mpr.materials.summary.search(material_ids=[mp_id],
                                                fields=["formation_energy_per_atom", "energy_above_hull"])
        else:
            docs = mpr.materials.summary.search(formula=composition.reduced_formula,
                                                fields=["formation_energy_per_atom", "energy_above_hull"])
        if not docs:
            return None
        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
        val = getattr(doc, 'formation_energy_per_atom', None)
        return float(val) if val is not None else None

def _fetch_density(self, composition):
    """density from MP API (needed when the input structure is a default skeleton, not a real cell)."""
    from mp_api.client import MPRester
    api_key = self._get_mp_api_key()
    if not api_key:
        return None
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(formula=composition.reduced_formula,
                                            fields=["density", "energy_above_hull"])
        if not docs:
            return None
        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
        val = getattr(doc, 'density', None)
        return float(val) if val is not None else None
```
Then in calculate(): if the caller passes only a skeleton structure and the training features include density,
resolve d as: explicit density arg -> MP API density (real network) -> structure.density.

## Model Loading Pattern
```python
def _load_models(self):
    # self.model_dir MUST be set in __init__ as the plugin-named subdirectory:
    #   plugin_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
    #   self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_name)
    model_path = os.path.join(self.model_dir, "two_stage_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(self.model_dir, "model.pkl")
    data = joblib.load(model_path)
    if isinstance(data, dict):
        # Branch B single-stage dict: keys models/feature_cols/target_col/algorithm
        # Branch B two-stage dict: keys models_stage1/models_stage2/stage1_cols/stage2_cols
        self.models_stage1 = data.get("models_stage1", data.get("models", []))
        self.models_stage2 = data.get("models_stage2", [])
        self.stage1_cols = data.get("stage1_cols", data.get("feature_cols", ELEMENT_COLS))
        self.stage2_cols = data.get("stage2_cols", [])
    else:
        # raw plain list of estimators (single-stage dump)
        self.models_stage1 = list(data)
        self.models_stage2 = []
        self.stage1_cols = ELEMENT_COLS
        self.stage2_cols = []
```
'''

def main():
    st.title("🔌 Model Converter")
    st.markdown(
        "Convert third-party prediction models into smart agent plugin format with retraining"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Folder Mode", "Upload Mode", "Template", "Installed Plugins"]
    )

    with tab1:
        st.markdown("### Step 1: Select Model Folder")
        st.markdown(f"Place your model files in: `external_models/<your_model_name>/`")

        os.makedirs(EXTERNAL_MODELS_DIR, exist_ok=True)

        folders = [
            d
            for d in os.listdir(EXTERNAL_MODELS_DIR)
            if os.path.isdir(os.path.join(EXTERNAL_MODELS_DIR, d)) and not d.startswith(".")
        ]

        if not folders:
            st.info(
                "No model folders found. Create a folder in external_models/ with your model files."
            )
            st.markdown("""
        **Expected folder structure:**
        ```
        external_models/
        └── oxide_MeltTemp/
            ├── model.py          # Model code
            ├── train_data.csv     # Training data
            ├── paper.pdf          # Reference paper
            └── supporting.pdf     # Supporting information
        ```
        """)
            selected_folder = None
        else:
            col_select, col_create = st.columns([2, 1])
            with col_select:
                selected_folder = st.selectbox(
                    "Select model folder:",
                    options=folders,
                    help="Choose a folder from external_models/",
                )
            with col_create:
                new_folder_name = st.text_input(
                    "New folder name:",
                    key="new_folder_name",
                    placeholder="e.g., oxide_MeltTemp",
                )
                if st.button("Create Folder", key="create_folder_btn"):
                    if new_folder_name.strip():
                        new_path = os.path.join(
                            EXTERNAL_MODELS_DIR, new_folder_name.strip()
                        )
                        os.makedirs(new_path, exist_ok=True)
                        st.success(f"Created: {new_folder_name}")
                        st.rerun()

        if selected_folder:
            folder_path = os.path.join(EXTERNAL_MODELS_DIR, selected_folder)
            st.markdown(f"**Selected:** `{selected_folder}`")

            st.markdown("### Step 2: Review Detected Files")

            files_info = scan_folder(folder_path)

            if not files_info:
                st.warning("No files found in the selected folder.")
            else:
                col_type1, col_type2, col_type3 = st.columns(3)

                code_files = [f for f in files_info if f["type"] == "code"]
                data_files = [f for f in files_info if f["type"] == "data"]
                paper_files = [f for f in files_info if f["type"] == "paper"]
                supporting_files = [f for f in files_info if f["type"] == "supporting"]

                selected_code = None
                selected_data = None
                selected_paper = None
                selected_supporting = None

                with col_type1:
                    st.markdown("**Code Files** (multi-select supported)")
                    if code_files:
                        code_options = {
                            f"{f['name']} ({f['type_label']})": f for f in code_files
                        }
                        selected_code_names = st.multiselect(
                            "Select code files (for multi-step models):",
                            options=list(code_options.keys()),
                            default=list(code_options.keys())[:1] if code_options else None,
                            key="code_multiselect",
                        )
                        selected_codes = (
                            [code_options[name] for name in selected_code_names]
                            if selected_code_names
                            else []
                        )
                    else:
                        st.info("No code files detected")
                        selected_codes = []

                    st.markdown("**Data Files** (multi-select supported)")
                    if data_files:
                        data_options = {
                            f"{f['name']} ({f['size']} bytes)": f for f in data_files
                        }
                        selected_data_names = st.multiselect(
                            "Select data files:",
                            options=list(data_options.keys()),
                            default=list(data_options.keys()),
                            key="data_multiselect",
                        )
                        selected_datas = (
                            [data_options[name] for name in selected_data_names]
                            if selected_data_names
                            else []
                        )
                    else:
                        st.info("No data files detected")
                        selected_datas = []

                    st.markdown("**Constant Files** (feature-calculation constant tables, e.g. element property tables; NOT used in training)")
                    st.caption(
                        "Selected data files are injected into the conversion prompt as constant files; "
                        "the LLM embeds the constant tables into the plugin code so features can be "
                        "computed from the formula alone."
                    )
                    if data_files:
                        const_options = {
                            f"{f['name']} ({f['size']} bytes)": f for f in data_files
                        }
                        selected_const_names = st.multiselect(
                            "Select constant files (element property / lookup tables):",
                            options=list(const_options.keys()),
                            default=[],
                            key="const_multiselect",
                        )
                        selected_consts = (
                            [const_options[name] for name in selected_const_names]
                            if selected_const_names
                            else []
                        )
                    else:
                        selected_consts = []

                with col_type2:
                    # Combine paper and supporting into one PDF category
                    all_pdf_files = paper_files + supporting_files

                    st.markdown("**PDF Documents** (multi-select)")
                    if all_pdf_files:
                        pdf_options = {
                            f"{f['name']} ({f['type_label']})": f for f in all_pdf_files
                        }
                        selected_pdf_names = st.multiselect(
                            "Select PDF files (paper + supporting):",
                            options=list(pdf_options.keys()),
                            default=list(pdf_options.keys()),
                            key="pdf_multiselect",
                        )
                        selected_pdfs = (
                            [pdf_options[name] for name in selected_pdf_names]
                            if selected_pdf_names
                            else []
                        )
                    else:
                        st.info("No PDF files detected")
                        selected_pdfs = []

                    # Also show separate selection for convenience (but use selected_pdfs)
                    st.markdown("**Auto-detect as:**")
                    col_pdf1, col_pdf2 = st.columns(2)
                    with col_pdf1:
                        selected_paper = st.selectbox(
                            "Primary paper:",
                            options=[f["name"] for f in selected_pdfs]
                            if selected_pdfs
                            else ["None"],
                            key="paper_select",
                        )
                    with col_pdf2:
                        selected_supporting_names = st.multiselect(
                            "Supporting info:",
                            options=[
                                f["name"]
                                for f in selected_pdfs
                                if f["name"] != selected_paper
                            ]
                            if selected_pdfs
                            else [],
                            key="supp_multiselect",
                        )

                with col_type3:
                    st.markdown("**Property Definition**")
                    prop_key = st.text_input(
                        "Property Key",
                        value=selected_folder.split("_")[-1]
                        if "_" in selected_folder
                        else "property",
                        key="prop_key_folder",
                    )
                    prop_symbol = st.text_input(
                        "Display Symbol",
                        value=selected_folder.split("_")[-1].upper()
                        if "_" in selected_folder
                        else "P",
                        key="prop_symbol_folder",
                    )
                    prop_unit = st.text_input("Unit", value="unit", key="prop_unit_folder")

                with col_type3:
                    st.markdown("**Training Settings**")
                    n_ensembles = st.number_input(
                        "Ensemble iterations (for ensemble models):",
                        min_value=1,
                        max_value=2000,
                        value=100,
                        step=100,
                        key="n_ensembles_folder",
                    )
                    st.caption("e.g., 1000 for 1000-fold ensemble like the Tm model")

                    st.markdown("**Special Requirements**")
                    model_notes = st.text_area(
                        "Notes for model (special requirements):",
                        placeholder="e.g., Use ANN for stage2, specific hyperparameters, custom feature calculations, etc.",
                        key="model_notes_folder",
                        height=80,
                    )
                    st.caption(
                        "These notes will be included in the AI prompt to generate accurate code"
                    )

                confirm_btn = st.button(
                    "Confirm & Analyze with AI", type="primary", use_container_width=True
                )

                if confirm_btn:
                    if not selected_codes and not selected_pdfs:
                        st.error("Please select at least code or PDF file")
                    else:
                        with st.spinner("Processing..."):
                            user_code = ""
                            paper_context = ""
                            training_data_info = ""

                            # Handle multiple code files
                            if selected_codes:
                                for i, code_file in enumerate(selected_codes):
                                    with open(
                                        code_file["path"],
                                        "r",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as f:
                                        content = f.read()
                                        user_code += (
                                            f"\n\n=== FILE {i + 1}: {code_file['name']} ===\n"
                                            + content
                                        )
                                st.success(f"Loaded {len(selected_codes)} code file(s)")

                            # Handle multiple PDF files
                            if selected_pdfs:
                                with st.spinner(
                                    f"Extracting text from {len(selected_pdfs)} PDF(s)..."
                                ):
                                    for pdf_file in selected_pdfs:
                                        text = extract_text_from_pdf(pdf_file["path"])
                                        if text:
                                            label = pdf_file["type_label"]
                                            limit = 15000 if label == "Paper" else 10000
                                            paper_context += (
                                                f"\n\n=== [{label}] {pdf_file['name']} ===\n"
                                                + text[:limit]
                                            )
                                    st.success(
                                        f"Extracted text from {len(selected_pdfs)} PDF(s)"
                                    )

                            # Handle multiple data files - combine them
                            const_file_names = {
                                cf["name"] for cf in selected_consts
                            }
                            dfs = []
                            if selected_datas:
                                for data_file in selected_datas:
                                    if data_file["name"] in const_file_names:
                                        continue  # constant files do not participate in training-data merging
                                    try:
                                        if data_file["name"].endswith(".csv"):
                                            df_temp = pd.read_csv(data_file["path"])
                                        elif data_file["name"].endswith((".xlsx", ".xls")):
                                            df_temp = pd.read_excel(data_file["path"])
                                        else:
                                            continue

                                        df_temp["_source_file"] = data_file["name"]
                                        dfs.append(df_temp)
                                        st.success(
                                            f"Loaded data: {data_file['name']} ({df_temp.shape[0]} rows)"
                                        )
                                    except Exception as e:
                                        st.warning(
                                            f"Could not read {data_file['name']}: {e}"
                                        )

                                if dfs:
                                    if len(dfs) == 1:
                                        df = dfs[0]
                                    else:
                                        # Combine multiple dataframes
                                        try:
                                            df = pd.concat(dfs, ignore_index=True)
                                            st.success(
                                                f"Combined {len(dfs)} data files: {df.shape[0]} total rows"
                                            )
                                        except:
                                            df = dfs[0]
                                            st.warning(
                                                "Could not combine files, using first file"
                                            )

                                    training_data_info = f"""
Training Data (combined): {df.shape[0]} rows, {df.shape[1]} columns
Columns: {list(df.columns)}
First few rows:
{df.head().to_string()}
"""
                                else:
                                    df = None
                            else:
                                df = None

                            model_files_detection = detect_model_files(folder_path)
                            st.session_state.processing = {
                                "folder": selected_folder,
                                "folder_path": folder_path,
                                "code": user_code,
                                "paper": paper_context,
                                "training_data": training_data_info,
                                "prop_key": prop_key,
                                "prop_symbol": prop_symbol,
                                "prop_unit": prop_unit,
                                "data_files": selected_datas,
                                "constant_files": selected_consts,
                                "data_df": df,
                                "n_ensembles": n_ensembles,
                                "model_notes": model_notes,
                                "model_files_detection": model_files_detection,
                                "branch": model_files_detection["branch"],
                                "auto_config_done": False,
                            }

        if "processing" in st.session_state:
            proc = st.session_state.processing

            # Step 2.5: AI Auto-Configuration
            st.markdown("---")
            st.markdown("### Step 3: AI Auto-Configuration")
            st.markdown("*AI will analyze your files and suggest optimal configuration*")

            auto_config_col, preview_col = st.columns([1, 1])

            with auto_config_col:
                if not proc.get("auto_config_done", False):
                    if st.button(
                        "Analyze with AI", type="primary", use_container_width=True
                    ):
                        with st.spinner("AI is analyzing your model..."):
                            # Check if we have any input
                            has_code = bool(proc.get("code"))
                            has_paper = bool(proc.get("paper"))
                            has_data = bool(proc.get("training_data"))

                            if not (has_code or has_paper or has_data):
                                st.error(
                                    "Please upload at least one file (code or PDF) before AI analysis"
                                )
                                proc["auto_config_done"] = True
                            else:
                                suggested_config = analyze_model_config(
                                    proc.get("paper", ""),
                                    proc.get("code", ""),
                                    proc.get("training_data", ""),
                                )

                                if suggested_config:
                                    proc["ai_suggested_config"] = suggested_config
                                    proc["prop_key"] = suggested_config.get(
                                        "key", proc.get("prop_key")
                                    )
                                    proc["prop_symbol"] = suggested_config.get(
                                        "symbol", proc.get("prop_symbol")
                                    )
                                    proc["prop_unit"] = suggested_config.get(
                                        "unit", proc.get("prop_unit")
                                    )
                                    proc["target_column"] = suggested_config.get(
                                        "target_column", ""
                                    )
                                    proc["stage1_features"] = suggested_config.get(
                                        "stage1_features", []
                                    )
                                    proc["stage2_features"] = suggested_config.get(
                                        "stage2_features", []
                                    )
                                    proc["is_two_stage"] = suggested_config.get(
                                        "two_stage", False
                                    )
                                    proc["model_notes"] = suggested_config.get(
                                        "notes", proc.get("model_notes", "")
                                    )
                                    proc["scope"] = suggested_config.get("scope", "")
                                    proc["auto_config_done"] = True
                                    st.success(
                                        "AI analysis complete! Please review and adjust if needed."
                                    )
                                else:
                                    st.warning(
                                        "AI analysis failed. Please configure manually."
                                    )
                                    proc["auto_config_done"] = True

            # Show suggested configuration
            if proc.get("ai_suggested_config"):
                config = proc["ai_suggested_config"]
                st.markdown(
                    '<div class="info-box">**AI Suggested Configuration:**</div>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"- **Property**: {config.get('property', 'UNKNOWN')}")
                    st.markdown(f"- **Key**: `{config.get('key', 'UNKNOWN')}`")
                    st.markdown(f"- **Symbol**: `{config.get('symbol', 'UNKNOWN')}`")
                    st.markdown(f"- **Unit**: `{config.get('unit', 'UNKNOWN')}`")
                with col2:
                    st.markdown(
                        f"- **Target Column**: `{config.get('target_column', 'UNKNOWN')}`"
                    )
                    st.markdown(f"- **Two-Stage**: {config.get('two_stage', 'UNKNOWN')}")
                    if config.get("stage1_features"):
                        st.markdown(
                            f"- **Stage 1 Features**: {len(config['stage1_features'])} elements"
                        )
                    if config.get("stage2_features"):
                        st.markdown(
                            f"- **Stage 2 Features**: {', '.join(config['stage2_features'][:5])}..."
                        )

                # Show scope
                if config.get("scope"):
                    st.markdown(
                        f"- **Scope/Applicability**: {config.get('scope', 'UNKNOWN')}"
                    )

            # Allow manual override
            st.markdown("**Configuration (editable):**")

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                new_prop_key = st.text_input(
                    "Property Key", value=proc.get("prop_key", ""), key="edit_prop_key"
                )
                proc["prop_key"] = new_prop_key
            with edit_col2:
                new_prop_symbol = st.text_input(
                    "Symbol", value=proc.get("prop_symbol", ""), key="edit_prop_symbol"
                )
                proc["prop_symbol"] = new_prop_symbol

            edit_col3, edit_col4 = st.columns(2)
            with edit_col3:
                new_prop_unit = st.text_input(
                    "Unit", value=proc.get("prop_unit", ""), key="edit_prop_unit"
                )
                proc["prop_unit"] = new_prop_unit
            with edit_col4:
                new_scope = st.text_input(
                    "Scope/Applicability",
                    value=proc.get("scope", ""),
                    key="edit_scope",
                    placeholder="e.g., Oxides only, ABO3 perovskite, General",
                )
                proc["scope"] = new_scope
                st.caption("Describe which materials this model applies to")

            st.markdown("---")
            st.markdown("### Step 4: Convert & Train")

            col_conv, col_train = st.columns(2)

            with col_conv:
                convert_btn = st.button(
                    "Convert Code", type="primary", use_container_width=True
                )

                if convert_btn:
                    with st.spinner("AI is converting your model..."):
                        prop_key_val = proc.get("prop_key", "property")
                        prop_symbol_val = proc.get("prop_symbol", "P")
                        prop_unit_val = proc.get("prop_unit", "unit")
                        n_ensembles_val = proc.get("n_ensembles", 100)
                        model_notes_val = proc.get("model_notes", "")

                        props_info = f"""
## Output Properties
- Property: {prop_symbol_val} ({prop_unit_val})
- Key: {prop_key_val}_mean, {prop_key_val}_dev
- Ensemble iterations: {n_ensembles_val}
"""

                        if model_notes_val:
                            props_info += f"""
## Special Requirements (User Notes)
{model_notes_val}
"""

                        # Replace placeholders in template
                        prompt = CONVERSION_PROMPT_TEMPLATE.replace(
                            "{{training_data_info}}",
                            proc.get("training_data", "No training data provided"),
                        )
                        constants_info, const_err = build_constants_info(
                            proc.get("constant_files", []),
                            plugin_name=proc.get("folder", "plugin"),
                        )
                        if const_err:
                            st.warning(f"Constant file read error: {const_err}")
                        prompt = prompt.replace(
                            "{{constants_info}}",
                            constants_info or "No user-provided constant files.",
                        )
                        prompt = prompt.replace(
                            "{{user_code}}", proc.get("code", "No code provided")
                        )
                        prompt = prompt.replace("{{paper_context}}", proc.get("paper", ""))
                        prompt = prompt.replace("{{n_ensembles}}", str(n_ensembles_val))
                        prompt = prompt.replace(
                            "{{model_notes}}", model_notes_val or "None"
                        )
                        prompt = prompt.replace("{{prop_key}}", prop_key_val)
                        prompt = prompt.replace("{{prop_symbol}}", prop_symbol_val)
                        prompt = prompt.replace("{{prop_unit}}", prop_unit_val)

                        # Two-branch: probe for ready-made model files under the source folder external_models/<model_name>/
                        source_folder = proc.get("folder_path") or proc.get("path") or ""
                        target_model_dir = os.path.join(
                            current_dir,
                            "plugins",
                            "external",
                            proc.get("folder", "plugin"),
                        )
                        model_detection = proc.get(
                            "model_files_detection"
                        ) or detect_model_files(source_folder)
                        branch = model_detection["branch"]
                        plugin_name = proc.get("folder", "plugin")
                        branch_info = build_branch_info(
                            model_detection, source_folder, target_model_dir, plugin_name
                        )
                        model_files_info = build_model_files_info(
                            [source_folder, target_model_dir]
                        )
                        prompt = prompt.replace("{{branch_info}}", branch_info)
                        prompt = prompt.replace("{{model_files_info}}", model_files_info)

                        # Prepend the properties info to the prompt
                        prompt = props_info + "\n\n" + prompt

                        converted_code, error = call_deepseek(prompt)

                        if error:
                            st.markdown(
                                f'<div class="error-box">{error}</div>',
                                unsafe_allow_html=True,
                            )
                        elif converted_code:
                            clean_code = extract_python_code(converted_code)
                            st.session_state.converted_code = clean_code
                            st.session_state.branch = branch
                            training_config, cfg_error = extract_training_config(
                                converted_code
                            )
                            if branch == "B" and training_config is None:
                                st.session_state.training_config = None
                                st.markdown(
                                    f'<div class="error-box">Training config extraction failed (Branch B must output TRAINING_CONFIG JSON; default model fallback is forbidden): {cfg_error}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.session_state.training_config = training_config
                                st.markdown(
                                    f'<div class="success-box">Code conversion successful! Branch {branch} detected.</div>',
                                    unsafe_allow_html=True,
                                )

            if "converted_code" in st.session_state:
                with col_train:
                    data_df = proc.get("data_df")

                    # Two-stage training configuration
                    st.markdown("**Training Configuration**")

                    all_cols = list(data_df.columns) if data_df is not None else []
                    non_target_cols = [
                        c
                        for c in all_cols
                        if c.lower()
                        not in ["target", "property", "y", "label", "output", "source_file"]
                    ]
                    target_cols = [
                        c
                        for c in all_cols
                        if c.lower() in ["target", "property", "y", "label", "output"]
                    ]

                    # Use AI-suggested target if available
                    ai_target = proc.get("target_column", "")
                    if ai_target and ai_target in all_cols:
                        default_target = ai_target
                    else:
                        default_target = (
                            target_cols[0] if target_cols else non_target_cols[0]
                        )

                    # Let user select which is target
                    target_col_selected = st.selectbox(
                        "Target column:",
                        options=[default_target]
                        + [c for c in all_cols if c != default_target],
                        index=0,
                        key="target_col_select",
                    )

                    # Use AI-suggested features if available
                    ai_stage1 = proc.get("stage1_features", [])
                    ai_stage2 = proc.get("stage2_features", [])

                    # Default to AI suggestions if valid
                    if ai_stage1:
                        default_stage1 = [c for c in ai_stage1 if c in non_target_cols]
                    else:
                        default_stage1 = (
                            non_target_cols[:3]
                            if len(non_target_cols) > 3
                            else non_target_cols
                        )

                    if ai_stage2:
                        default_stage2 = [
                            c
                            for c in ai_stage2
                            if c in non_target_cols and c not in default_stage1
                        ]
                    else:
                        default_stage2 = []

                    # Let user specify stage 1 features (composition-based)
                    stage1_cols = st.multiselect(
                        "Stage 1 features (composition-based):",
                        options=non_target_cols,
                        default=default_stage1,
                        key="stage1_cols",
                    )

                    # Let user specify additional stage 2 features
                    stage2_additional_cols = st.multiselect(
                        "Additional Stage 2 features:",
                        options=[c for c in non_target_cols if c not in stage1_cols],
                        default=default_stage2,
                        key="stage2_cols",
                    )

                    # Auto-detect if this is a two-stage model (use AI suggestion if available)
                    ai_two_stage = proc.get("is_two_stage", len(stage2_additional_cols) > 0)
                    is_two_stage = st.checkbox(
                        "Two-stage model (cascaded prediction)",
                        value=ai_two_stage,
                        key="is_two_stage",
                    )

                    # Number of bootstrap iterations (train/test splits)
                    n_iterations = st.number_input(
                        "Bootstrap iterations (train/test splits):",
                        min_value=10,
                        max_value=2000,
                        value=100,
                        step=10,
                        key="n_iterations",
                    )
                    st.caption("Each iteration: random 70% train, 30% test, get one model")

                    train_btn = st.button(
                        "Train Model", type="primary", use_container_width=True
                    )

                    # Two-branch training entry:
                    # Branch A = reuse ready-made model files (copy into the plugin subdirectory, skip training)
                    # Branch B = config-driven retraining (training-config JSON extracted by the LLM)
                    branch_for_train = (
                        st.session_state.get("branch")
                        or proc.get("model_files_detection", {}).get("branch", "B")
                    )
                    if train_btn and branch_for_train == "A":
                        model_detection_a = proc.get(
                            "model_files_detection"
                        ) or detect_model_files(proc.get("folder_path") or "")
                        model_dir_a = os.path.join(
                            current_dir, "plugins", "external", proc["folder"]
                        )
                        copy_res = copy_existing_models(
                            model_detection_a.get("files", []), model_dir_a
                        )
                        if copy_res["errors"]:
                            st.error(
                                "Model file copy failed: " + "; ".join(copy_res["errors"])
                            )
                        else:
                            st.success(
                                f"Branch A: reused existing model files, training skipped. "
                                f"Copied {len(copy_res['copied'])} file(s) → {model_dir_a}"
                            )
                        st.session_state.trained = True
                    elif train_btn:
                        data_df = proc.get("data_df")
                        if data_df is None or data_df.empty:
                            st.error("No training data available")
                        else:
                            with st.spinner("Training model..."):
                                try:
                                    from features import FeatureCalculator
                                    from data_loader import DataLoader

                                    loader = DataLoader()
                                    calc = FeatureCalculator(loader)

                                    target_col = target_col_selected

                                    # Extract features using the same method as the plugin
                                    features_list = []
                                    targets = []

                                    if (
                                        "composition" in data_df.columns
                                        or "formula" in data_df.columns
                                    ):
                                        for idx, row in data_df.iterrows():
                                            try:
                                                formula = row.get(
                                                    "composition", row.get("formula", "")
                                                )
                                                if formula:
                                                    comp = comp_str_to_structure(
                                                        formula
                                                    )
                                                    structure = comp

                                                    # Calculate features the same way as the plugin
                                                    base_feats, _ = (
                                                        calc.calc_features_ordered(
                                                            structure
                                                        )
                                                    )

                                                    # Add elemental composition fractions (same as plugin)
                                                    el_amt_dict = comp.get_el_amt_dict()
                                                    total_atoms = sum(el_amt_dict.values())
                                                    element_cols = [
                                                        "H",
                                                        "Li",
                                                        "Be",
                                                        "B",
                                                        "C",
                                                        "N",
                                                        "O",
                                                        "Na",
                                                        "Mg",
                                                        "Al",
                                                        "Si",
                                                        "P",
                                                        "S",
                                                        "Cl",
                                                        "K",
                                                        "Ca",
                                                        "Ti",
                                                        "V",
                                                        "Cr",
                                                        "Mn",
                                                        "Fe",
                                                        "Co",
                                                        "Cu",
                                                        "Zn",
                                                        "As",
                                                        "Se",
                                                        "Br",
                                                        "Rb",
                                                        "Sr",
                                                        "Zr",
                                                        "Nb",
                                                        "Mo",
                                                        "Ag",
                                                        "Cd",
                                                        "Te",
                                                        "I",
                                                        "Cs",
                                                        "Ba",
                                                        "W",
                                                        "Re",
                                                        "Tl",
                                                        "Pb",
                                                    ]
                                                    for elem in element_cols:
                                                        if elem in el_amt_dict:
                                                            base_feats[elem] = (
                                                                el_amt_dict[elem]
                                                                / total_atoms
                                                            )
                                                        else:
                                                            base_feats[elem] = 0.0

                                                    # Add custom features
                                                    base_feats["na"] = (
                                                        structure.composition.num_atoms
                                                    )
                                                    base_feats["d"] = structure.density

                                                    # Add user-specified additional features
                                                    for col in stage2_additional_cols:
                                                        if col in data_df.columns:
                                                            base_feats[col] = row[col]

                                                    features_list.append(base_feats)
                                                    targets.append(row[target_col])
                                            except Exception as e:
                                                continue
                                    else:
                                        # Use existing features from data
                                        for idx, row in data_df.iterrows():
                                            feats = {
                                                col: row[col]
                                                for col in stage1_cols
                                                + stage2_additional_cols
                                                if col in data_df.columns
                                            }
                                            features_list.append(feats)
                                            targets.append(row[target_col])

                                    if not features_list:
                                        st.error("No valid samples found")
                                    else:
                                        features_df = pd.DataFrame(features_list)

                                        # Remove non-numeric columns
                                        numeric_cols = features_df.select_dtypes(
                                            include=[np.number]
                                        ).columns.tolist()
                                        features_df = features_df[numeric_cols]

                                        model_dir = os.path.join(
                                            current_dir,
                                            "plugins",
                                            "external",
                                            proc["folder"],
                                        )
                                        os.makedirs(model_dir, exist_ok=True)

                                        # Auto-detect stage 1 and stage 2 columns
                                        element_cols = [
                                            "H",
                                            "Li",
                                            "Be",
                                            "B",
                                            "C",
                                            "N",
                                            "O",
                                            "Na",
                                            "Mg",
                                            "Al",
                                            "Si",
                                            "P",
                                            "S",
                                            "Cl",
                                            "K",
                                            "Ca",
                                            "Ti",
                                            "V",
                                            "Cr",
                                            "Mn",
                                            "Fe",
                                            "Co",
                                            "Cu",
                                            "Zn",
                                            "As",
                                            "Se",
                                            "Br",
                                            "Rb",
                                            "Sr",
                                            "Zr",
                                            "Nb",
                                            "Mo",
                                            "Ag",
                                            "Cd",
                                            "Te",
                                            "I",
                                            "Cs",
                                            "Ba",
                                            "W",
                                            "Re",
                                            "Tl",
                                            "Pb",
                                        ]

                                        # Use elemental composition for stage 1
                                        stage1_X_cols = [
                                            c
                                            for c in element_cols
                                            if c in features_df.columns
                                        ]

                                        # Stage 2 = stage1 + custom features + stage1_pred
                                        custom_cols = [
                                            "na",
                                            "d",
                                            "blsd",
                                            "blnpv",
                                            "epa",
                                            "fepa",
                                        ]
                                        stage2_feats_cols = stage1_X_cols.copy()
                                        for c in custom_cols:
                                            if c in features_df.columns:
                                                stage2_feats_cols.append(c)

                                        if is_two_stage and len(stage2_additional_cols) > 0:
                                            # Add user-specified stage 2 additional features
                                            for c in stage2_additional_cols:
                                                if (
                                                    c not in stage2_feats_cols
                                                    and c in features_df.columns
                                                ):
                                                    stage2_feats_cols.append(c)

                                        all_stage2_cols = stage2_feats_cols + [
                                            "stage1_pred"
                                        ]

                                        if is_two_stage and len(stage2_additional_cols) > 0:
                                            # Two-stage training: save ALL models in ONE file
                                            # Branch B: config-driven (LLM extracts from original code/literature); default templates are forbidden
                                            import inspect as _inspect

                                            training_config = st.session_state.get(
                                                "training_config"
                                            ) or {}
                                            stage1_alg = (
                                                training_config.get("stage1_algorithm")
                                                or training_config.get("algorithm")
                                                or ""
                                            ).strip()
                                            stage2_alg = (
                                                training_config.get("stage2_algorithm")
                                                or stage1_alg
                                            ).strip()
                                            if not stage1_alg or not stage2_alg:
                                                st.error(
                                                    "Branch B: two-stage training missing algorithm config "
                                                    "(TRAINING_CONFIG algorithm/"
                                                    "stage1_algorithm is empty), "
                                                    "default template fallback is forbidden"
                                                )
                                                raise ValueError(
                                                    "Missing algorithm in "
                                                    "TRAINING_CONFIG for "
                                                    "two-stage training"
                                                )
                                            st.markdown(
                                                f"**Training two-stage model with "
                                                f"{n_iterations} bootstrap iterations "
                                                f"(config-driven: stage1={stage1_alg}, "
                                                f"stage2={stage2_alg})...**"
                                            )

                                            stage1_X = (
                                                features_df[stage1_X_cols].fillna(0).values
                                            )
                                            y = np.array(targets)

                                            def _make_estimator(alg_name, cfg_key):
                                                est_cls = resolve_sklearn_estimator(
                                                    alg_name
                                                )
                                                if est_cls is None:
                                                    raise ValueError(
                                                        f"Cannot resolve algorithm from sklearn: "
                                                        f"'{alg_name}' (training config extraction "
                                                        "unusable; default model fallback is forbidden)"
                                                    )
                                                raw_hp = training_config.get(
                                                    cfg_key
                                                ) or training_config.get(
                                                    "hyperparameters"
                                                ) or {}
                                                try:
                                                    sig = _inspect.signature(
                                                        est_cls.__init__
                                                    )
                                                    hparams = {
                                                        k: v
                                                        for k, v in raw_hp.items()
                                                        if k in sig.parameters
                                                        and k != "self"
                                                    }
                                                except Exception:
                                                    hparams = dict(raw_hp)
                                                pre = (
                                                    training_config.get(
                                                        "stage2_preprocessing"
                                                    )
                                                    if cfg_key
                                                    == "stage2_hyperparameters"
                                                    else training_config.get(
                                                        "preprocessing"
                                                    )
                                                )
                                                return (
                                                    build_preprocessing_pipeline(
                                                        pre or [], est_cls(**hparams)
                                                    ),
                                                    hparams,
                                                )

                                            # Store all models
                                            all_stage1_models = []
                                            all_stage2_models = []
                                            all_preds = []

                                            progress_bar = st.progress(0)

                                            for i in range(n_iterations):
                                                if i % 10 == 0:
                                                    progress_bar.progress(i / n_iterations)

                                                n_samples = len(y)
                                                train_idx = np.random.choice(
                                                    n_samples,
                                                    size=int(0.7 * n_samples),
                                                    replace=True,
                                                )

                                                # Stage 1 (config-driven)
                                                m1, hparams1 = _make_estimator(
                                                    stage1_alg, "hyperparameters"
                                                )
                                                m1.fit(stage1_X[train_idx], y[train_idx])

                                                # Get stage 1 predictions for all data
                                                sp_full = m1.predict(stage1_X)

                                                # Prepare stage 2 features
                                                features_temp = features_df.copy()
                                                features_temp["stage1_pred"] = sp_full
                                                X2_full = (
                                                    features_temp[all_stage2_cols]
                                                    .fillna(0)
                                                    .values
                                                )

                                                # Stage 2 (config-driven)
                                                m2, hparams2 = _make_estimator(
                                                    stage2_alg,
                                                    "stage2_hyperparameters",
                                                )
                                                m2.fit(X2_full[train_idx], y[train_idx])

                                                preds = m2.predict(X2_full)
                                                all_preds.append(preds)

                                                all_stage1_models.append(m1)
                                                all_stage2_models.append(m2)

                                            progress_bar.progress(1.0)

                                            # Calculate stats
                                            all_preds = np.array(all_preds)
                                            mean_pred = np.mean(all_preds, axis=0)
                                            std_pred = np.std(all_preds, axis=0)

                                            # Save EVERYTHING in ONE file
                                            import sklearn as _sklearn
                                            save_data = {
                                                "models_stage1": all_stage1_models,
                                                "models_stage2": all_stage2_models,
                                                "mean": mean_pred,
                                                "std": std_pred,
                                                "stage1_cols": stage1_X_cols,
                                                "stage2_cols": all_stage2_cols,
                                                "n_iterations": n_iterations,
                                                "prop_key": proc.get(
                                                    "prop_key", "Tm"
                                                ),  # User's input
                                                "prop_symbol": proc.get(
                                                    "prop_symbol", "Tm"
                                                ),
                                                "prop_unit": proc.get("prop_unit", "K"),
                                                "algorithm": stage1_alg,
                                                "stage2_algorithm": stage2_alg,
                                                "hyperparameters": hparams1,
                                                "stage2_hyperparameters": hparams2,
                                                "sklearn_version": _sklearn.__version__,
                                            }
                                            # sklearn version compatibility: add monotonic_cst to all trees before dump
                                            patch_tree_sklearn_compat(save_data)
                                            joblib.dump(
                                                save_data,
                                                os.path.join(
                                                    model_dir, "two_stage_model.pkl"
                                                ),
                                            )

                                            st.success(
                                                f"Saved all {n_iterations} models in ONE file: two_stage_model.pkl (stage1={stage1_alg}, stage2={stage2_alg})"
                                            )
                                        else:
                                            # Single-stage training: save ALL models in ONE file
                                            # Branch B: config-driven (LLM extracts from original code/literature); default templates are forbidden
                                            training_config = st.session_state.get(
                                                "training_config"
                                            ) or {}
                                            if not (training_config.get("algorithm") or "").strip():
                                                st.error(
                                                    "Branch B: single-stage training missing algorithm config "
                                                    "(TRAINING_CONFIG algorithm is empty), "
                                                    "default template fallback is forbidden"
                                                )
                                                raise ValueError(
                                                    "Missing algorithm in "
                                                    "TRAINING_CONFIG for single-stage "
                                                    "training"
                                                )
                                            st.markdown(
                                                f"**Training with {n_iterations} "
                                                f"bootstrap iterations "
                                                f"(config-driven: "
                                                f"{training_config.get('algorithm')})...**"
                                            )

                                            X = features_df.fillna(0).values
                                            y = np.array(targets)

                                            model_path = os.path.join(
                                                model_dir, "model.pkl"
                                            )
                                            progress_bar = st.progress(0)
                                            save_data = train_with_config(
                                                X,
                                                y,
                                                list(features_df.columns),
                                                str(target_col_selected),
                                                training_config,
                                                n_iterations,
                                                model_path,
                                                progress_cb=lambda f: progress_bar.progress(
                                                    f
                                                ),
                                            )

                                            st.success(
                                                f"Saved all {n_iterations} models in ONE file: model.pkl "
                                                f"(algorithm={save_data.get('algorithm')})"
                                            )
                                except Exception as e:
                                    st.error(f"Training failed: {str(e)}")

            if "converted_code" in st.session_state:
                st.markdown("---")
                st.markdown("### Generated Plugin Code")

                st.code(st.session_state.converted_code, language="python")

                save_btn = st.button("Save Plugin", type="primary")
                if save_btn:
                    try:
                        output_path = save_plugin_code(
                            st.session_state.converted_code,
                            proc.get("folder", "plugin"),
                            proc.get("constant_files", []),
                        )
                        st.success(
                            f"Saved to: plugins/external/{os.path.basename(output_path)} "
                            f"(smoke test PASSED)"
                        )
                    except Exception as e:
                        st.error(
                            f"Save failed - the generated plugin did not pass the programmatic smoke check; saving was blocked: {str(e)}"
                        )

    with tab2:
        st.markdown("### Upload Mode (Legacy)")
        st.info(
            "Use Folder Mode for better experience. This tab provides direct file upload."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Step 1: Upload Model Code")

            uploaded_code = st.file_uploader(
                "Drag & drop code file (Python, R, or text)",
                type=["py", "r", "txt"],
                key="code_main_legacy",
                help="Upload your model code file",
            )

            uploaded_extra_codes = st.file_uploader(
                "Upload additional model files",
                type=["py", "r", "txt"],
                accept_multiple_files=True,
                key="code_extra_legacy",
            )

            user_code = ""

            if uploaded_code is not None:
                try:
                    user_code = uploaded_code.getvalue().decode("utf-8")
                    st.success(f"Loaded: {uploaded_code.name} ({len(user_code)} chars)")
                except:
                    st.error("Failed to read file encoding")

            if uploaded_extra_codes:
                for f in uploaded_extra_codes:
                    try:
                        extra_content = f.getvalue().decode("utf-8")
                        user_code += f"\n\n=== FILE: {f.name} ===\n" + extra_content
                        st.success(f"Loaded: {f.name}")
                    except:
                        st.warning(f"Could not read: {f.name}")

            st.markdown("#### Step 2: Upload Reference Papers")

            uploaded_pdfs = st.file_uploader(
                "Drag & drop paper PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_legacy",
            )

            paper_context = ""
            if uploaded_pdfs:
                for pdf in uploaded_pdfs:
                    st.markdown(
                        f'<span class="pdf-tag">{pdf.name}</span>', unsafe_allow_html=True
                    )

                with st.spinner("Extracting text from PDFs..."):
                    all_text = []
                    for pdf in uploaded_pdfs:
                        text = extract_text_from_pdf(pdf)
                        if text:
                            all_text.append(f"=== {pdf.name} ===\n{text[:10000]}")

                    if all_text:
                        paper_context = "\n\n".join(all_text)
                        st.success(f"Extracted text from {len(uploaded_pdfs)} PDF(s)")

            st.markdown("#### Step 2.5: Upload Constant Files (feature-calculation constant tables)")
            uploaded_consts = st.file_uploader(
                "Drag & drop constant files (element property / lookup tables, e.g. Table_S2.csv)",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="const_legacy",
            )
            constant_files = []
            if uploaded_consts:
                os.makedirs("temp_uploads", exist_ok=True)
                for cf in uploaded_consts:
                    tmp_path = os.path.join("temp_uploads", cf.name)
                    with open(tmp_path, "wb") as f:
                        f.write(cf.getbuffer())
                    constant_files.append({"name": cf.name, "path": tmp_path})
                    st.markdown(
                        f'<span class="pdf-tag">Constant: {cf.name}</span>',
                        unsafe_allow_html=True,
                    )

        with col2:
            st.markdown("#### Step 3: Define Output Properties")

            col_prop1, col_prop2, col_prop3 = st.columns(3)
            with col_prop1:
                prop_key = st.text_input(
                    "Property Key", value="melt_temp", key="prop_key_legacy"
                )
            with col_prop2:
                prop_symbol = st.text_input(
                    "Display Symbol", value="Tm", key="prop_symbol_legacy"
                )
            with col_prop3:
                prop_unit = st.text_input("Unit", value="K", key="prop_unit_legacy")

            st.markdown("#### Step 4: Generate Plugin")
            plugin_name_input = st.text_input(
                "Plugin name:", value="my_predictor", key="plugin_name_legacy"
            )

            convert_btn = st.button(
                "Convert to Plugin",
                type="primary",
                use_container_width=True,
                key="convert_legacy",
            )

            if convert_btn:
                if not user_code.strip() and not paper_context.strip():
                    st.error("Please upload model code or paper PDF")
                else:
                    with st.spinner("AI is converting..."):
                        props_info = f"""
## Output Properties
- Property: {prop_symbol} ({prop_unit})
- Key: {prop_key}_mean, {prop_key}_dev
"""

                        prompt = (
                            CONVERSION_PROMPT_TEMPLATE.replace(
                                "{{training_data_info}}", "No training data"
                            )
                            .replace("{{user_code}}", user_code or "No code provided")
                            .replace("{{paper_context}}", paper_context)
                            .replace(
                                "{{model_files_info}}",
                                "No model files detected (legacy upload mode).",
                            )
                        )
                        constants_info, const_err = build_constants_info(
                            constant_files
                        )
                        if const_err:
                            st.warning(f"Constant file read error: {const_err}")
                        prompt = prompt.replace(
                            "{{constants_info}}",
                            constants_info or "No user-provided constant files.",
                        )

                        converted_code, error = call_deepseek(prompt)

                        if error:
                            st.markdown(
                                f'<div class="error-box">{error}</div>',
                                unsafe_allow_html=True,
                            )
                        elif converted_code:
                            clean_code = extract_python_code(converted_code)
                            st.session_state.converted_code_legacy = clean_code
                            st.markdown(
                                '<div class="success-box">Conversion successful!</div>',
                                unsafe_allow_html=True,
                            )

        if "converted_code_legacy" in st.session_state:
            st.markdown("---")
            st.markdown("### Generated Plugin Code")
            st.code(st.session_state.converted_code_legacy, language="python")

    with tab3:
        st.markdown("### Plugin Template")

        template_code = '''from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

metadata = ModelMetadata(
    name="Your Model Name",
    version="1.0.0",
    author="Author Name",
    description="Model description with retraining support",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

class YourFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata

    # HARD CONSTRAINT (a): features MUST be driven by the REAL feature_cols from the model/training config.
    # NEVER hardcode placeholder element-symbol names like ['H','Li','Ca','O'] as property columns.
    # HARD CONSTRAINT (b): map feature column names to Element_Property_Table.csv columns with an
    # explicit alias/fuzzy match (e.g. Density -> 'Density (g/mL)', ionization -> '1st ionization potential (kJ/mol)',
    # Bulk -> 'Bulk'); raise an explicit error if a column cannot be resolved - NEVER silently return 0.0.

    def __init__(self):
        self.feature_cols = []
        # read REAL feature_cols from the trained model (same path pattern as the Predictor)
        plugin_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_name, "model.pkl")
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            if isinstance(data, dict):
                self.feature_cols = list(data.get("feature_cols") or [])

    def calculate(self, structure, base_features):
        features = {}
        # 1) parse composition: comp = structure.composition; el_amt = comp.get_el_amt_dict()
        #    REAL API (pymatgen Structure): structure.composition -> Composition
        #      el_amt = structure.composition.get_el_amt_dict()  # {'Ba':1,'Ti':1,'O':3} (element symbol -> amount)
        #      total  = structure.composition.num_atoms           # number of atoms
        #      mole_frac = {el: amt / total for el, amt in el_amt.items()}  # element symbol -> mole fraction
        #    NEVER call structure.get_el_amt_dict() directly - Structure has NO such method.
        # 2) for each col in self.feature_cols: resolve table column (exact -> alias -> fuzzy),
        #    then compute composition-weighted average over elements from Element_Property_Table.csv
        # 3) MISSING-VALUE SEMANTICS (HARD CONSTRAINT b, training-side convention fillna(median)):
        #    - load table ONCE and pre-fill NaN with the column MEDIAN:
        #        table = pd.read_csv(feature_table_path)
        #        table = table.fillna(table.median(numeric_only=True))
        #    - per-element lookup is NaN-safe (dict.get with default / pd.isna check):
        #        row = table.loc[table['Symbol'] == el]
        #        val = row[col].iloc[0] if len(row) else None
        #        if val is None or pd.isna(val):
        #            continue  # skip this element in the weighted average - NEVER raise for a single element
        #    - if EVERY element is skipped for a column (weighted result is NaN / median was NaN):
        #        raise explicit error (case ii)
        #    - if the column NAME cannot be resolved to a table column (after alias/fuzzy):
        #        raise explicit error (case iii) - NEVER silently return 0.0
        #    NEVER raise "Element 'X' has no value for property 'Y'" for a single-element data gap.
        return features

    def get_required_base_features(self):
        return []

    def get_required_features(self):
        # HARD CONSTRAINT (a): return the REAL feature columns, never placeholders
        return self.feature_cols

class YourPredictor(BasePredictor):
    metadata = metadata

    # HARD CONSTRAINT (c): prop_key is the property id used for output keys <prop_key>_mean/_dev
    # (e.g. "tec", "er", "melt_temp"). NEVER use the deprecated "property_mean"/"property_dev".

    def __init__(self):
        self.prop_key = "your_prop"  # e.g. "tec"
        self.model = None
        self.feature_cols = []
        self.model_path = os.path.join(os.path.dirname(__file__), "your_model.pkl")
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                if isinstance(data, dict):
                    self.model = data.get("models") or data
                    self.feature_cols = list(data.get("feature_cols") or [])
                else:
                    self.model = data
            except Exception:
                self.model = None

    def train(self, features_df, target_values):
        """Train the model with provided data"""
        X = features_df.values
        y = np.array(target_values)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(
            {"models": [self.model], "feature_cols": list(features_df.columns), "target_col": self.prop_key},
            self.model_path,
        )

    def predict(self, features):
        if self.model is None:
            return {f"{self.prop_key}_mean": None, f"{self.prop_key}_dev": None, "note": "Model not loaded"}

        X = np.array([[features.get(f, 0.0) for f in self.get_required_features()]])
        prediction = self.model.predict(X)[0]

        # HARD CONSTRAINT (c): output keys MUST be <prop_key>_mean / <prop_key>_dev
        return {f"{self.prop_key}_mean": float(prediction), f"{self.prop_key}_dev": 0.0}

    def get_required_features(self):
        # HARD CONSTRAINT (a): return the REAL feature columns read from the model
        return self.feature_cols

register_plugin(Plugin(
    id="your_model_id",
    structure_type="general",
    feature_calculator=YourFeatureCalculator(),
    predictor=YourPredictor()
))
'''

    with tab4:
        st.markdown("### Installed Plugins")

        from plugins import global_registry

        plugins = global_registry.list_all()

        if plugins:
            for p in plugins:
                with st.expander(f"{p.id} ({p.structure_type})"):
                    meta = p.metadata
                    if meta:
                        st.markdown(f"""
                    - **Name**: {meta.name}
                    - **Version**: {meta.version}
                    - **Author**: {meta.author}
                    - **Description**: {meta.description}
                    """)

                    status = "Enabled" if p.enabled else "Disabled"
                    st.markdown(f"**Status**: {status}")
        else:
            st.info("No plugins installed yet.")

        st.markdown("---")
        st.markdown("#### Plugin Directory")

        plugin_dirs = ["plugins/builtin", "plugins/external"]
        for d in plugin_dirs:
            full_path = os.path.join(current_dir, d)
            if os.path.exists(full_path):
                files = [
                    f
                    for f in os.listdir(full_path)
                    if f.endswith(".py") and not f.startswith("_")
                ]
                st.markdown(f"**{d}/**")
                for f in files:
                    st.markdown(f"  - {f}")


# Smoke-test candidate formula pool: covers common systems such as ABO3 / A2B2O7 / binary oxides.
# Selection rule: every element of a candidate formula must be within the plugin constant-table element set,
# to avoid testing an A2B2O7 plugin with BaTiO3/SrTiO3 (whose constant tables lack Ba/Sr) and getting a false "missing element attribute" error.
_SMOKE_FORMULA_POOL = [
    "BaTiO3", "SrTiO3", "CaTiO3", "LaAlO3",
    "La2Zr2O7", "Y2Ti2O7", "Nd2Hf2O7", "Sm2Sn2O7",
    "Gd2Zr2O7", "Er2Ti2O7", "Yb2Hf2O7", "Lu2Sn2O7",
    "MgO", "Al2O3", "SiO2", "TiO2", "ZrO2", "HfO2", "SnO2",
# Multi-cation high-entropy samples: expose the "features insensitive to composition" degeneration (with a single A and single B, the weighted std is always 0,
# so a correctly weighted formula cannot be distinguished from an incorrect unweighted/placeholder formula). Kept pymatgen-parseable.
    "(Y0.2Gd0.2Er0.2Yb0.2Lu0.2)2Zr2O7",
    "(Sm0.25Eu0.25Gd0.25Yb0.25)2Zr2O7",
    "(Y0.5Gd0.5)2(Ti0.5Zr0.5)2O7",
]


def _extract_constant_table_elements(ns):
    """Extract the constant-table element set from the plugin module namespace after exec.

    Prefer the 'Element' column (when read_csv was not followed by set_index);
    otherwise take the non-RangeIndex object-type index (after set_index('Element')).
    Returns None if no constant table can be identified (the caller falls back to default samples).
    """
    import pandas as pd
    for name, val in ns.items():
        if not isinstance(val, pd.DataFrame):
            continue
        if "Element" in val.columns:
            return set(val["Element"].astype(str).str.strip())
        if (
            val.index.dtype == object
            and len(val.index) > 0
            and not isinstance(val.index, pd.RangeIndex)
        ):
            return set(val.index.astype(str).str.strip())
    return None


def _formula_elements(formula):
    from pymatgen.core import Composition
    return {el.symbol for el in Composition(formula).elements}


def _cation_count(formula):
    """Count the number of cation species (rough: number of element symbols in the formula - O count)"""
    from pymatgen.core import Composition
    try:
        comp = Composition(formula)
        return len([el for el in comp.elements if el.symbol != "O"])
    except Exception:
        return 1


def _select_smoke_formulas(ns, sample_formulas):
    """System-aware smoke-sample selection: prefer candidate formulas whose elements are all inside the constant table.

    When the caller explicitly passes sample_formulas, use them as-is (respecting the explicit choice);
    otherwise extract the element set from the plugin's module-level constant table and filter the candidate pool;
    prefer multi-cation (high-entropy) samples to expose feature degeneration, otherwise fall back to the default logic;
    fall back to the default ABO3 sample when nothing matches (the error message then points to a system/constant-table problem).
    """
    if sample_formulas is not None:
        return list(sample_formulas)
    elems = _extract_constant_table_elements(ns)
    if not elems:
        return ["BaTiO3", "SrTiO3"]
    matched = [f for f in _SMOKE_FORMULA_POOL if _formula_elements(f) <= elems]
    if len(matched) >= 2:
        # Prefer high-entropy samples with >= 3 cation species (at most 2),
        # top up with ordinary samples when insufficient: ensuring both discriminative power and multi-cation degeneration coverage.
        he = [f for f in matched if _cation_count(f) >= 3]
        base = he[:2]
        rest = [f for f in matched if f not in base]
        return (base + rest)[:2]
    if len(matched) == 1:
        return matched
    return ["BaTiO3", "SrTiO3"]


# Framework-level fallback function: physical-rule A/B-site assignment for formulas without parentheses (BaTiO3/SrTiO3/La2Zr2O7 etc.).
# If an LLM-generated plugin has not implemented the constraint-h fallback, then on smoke-test failure
# _auto_patch_no_parenthesis_fallback automatically injects this function and rewrites calculate().
_SPLIT_A_B_SITES_SNIPPET = '''def _split_a_b_sites(amt_dict):
    """Framework-injected fallback (constraint h): A/B-site physical rule for
    formulas WITHOUT parentheses (e.g. BaTiO3, La2Zr2O7). NEVER raises for any
    cation-containing oxide."""
    cations = {el: amt for el, amt in amt_dict.items() if el != "O"}
    if not cations:
        raise ValueError("Cannot parse A-site: no cation in composition")
    B_CANDIDATES = ("Zr", "Ti", "Hf", "Sn", "Nb", "Ta", "W", "Mo", "Mn")
    b_els = [el for el in cations if el in B_CANDIDATES]
    if b_els:
        b_amt = {el: cations[el] for el in b_els}
        a_amt = {el: amt for el, amt in cations.items() if el not in b_els}
        if not a_amt:  # single-cation case: put the smallest B cation into A
            min_el = min(b_amt, key=b_amt.get)
            a_amt = {min_el: b_amt.pop(min_el)}
    else:
        order = sorted(cations.items(), key=lambda kv: kv[1])
        if len(order) == 1:  # simple binary oxide MO
            a_amt = dict(order)
            b_amt = dict(order)
        else:
            b_el, b_v = order[0]
            b_amt = {b_el: b_v}
            a_amt = {el: amt for el, amt in order[1:]}

    def _norm(d):
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in d.items()}

    return _norm(a_amt), _norm(b_amt)


'''


# Match the typical bad branch inside calculate() that does "when no parentheses, use composition.formula and then run a parenthesis regex":
#   orig = getattr(structure, '_orig_formula', None)
#   if orig is not None and '(' in orig: formula = orig
#   else: formula = structure.composition.formula
#   total_elems, a_elems, b_elems = self._parse_composition(formula)
_PATCH_NO_PAREN_RE = re.compile(
    r"^(?P<ind>[ \t]+)orig = getattr\(structure, '_orig_formula', None\)"
    r"[\s\S]*?total_elems, a_elems, b_elems = self\._parse_composition\(formula\)",
    re.MULTILINE,
)

_PATCH_NO_PAREN_REPL = (
    "\\g<ind>orig = getattr(structure, '_orig_formula', None)\n"
    "\\g<ind>if orig is not None and '(' in orig:\n"
    "\\g<ind>    total_elems, a_elems, b_elems = self._parse_composition(orig)\n"
    "\\g<ind>else:\n"
    "\\g<ind>    _amt = {el.symbol: float(v) for el, v in structure.composition.get_el_amt_dict().items()}\n"
    "\\g<ind>    _a_amt, _b_amt = _split_a_b_sites(_amt)\n"
    "\\g<ind>    total_elems = dict(_a_amt)\n"
    "\\g<ind>    total_elems.update(_b_amt)\n"
    "\\g<ind>    total_elems['O'] = _amt.get('O', 0.0)\n"
    "\\g<ind>    a_elems = dict(_a_amt)\n"
    "\\g<ind>    b_elems = dict(_b_amt)"
)


def _auto_patch_no_parenthesis_fallback(plugin_path):
    """Automatically inject the physical A/B-site fallback for parenthesized formulas into the plugin (root-cause defense, not relying on LLM compliance).

    1) If the source lacks a _split_a_b_sites definition, inject the function before the FeatureCalculator class;
    2) If calculate() has the bad branch "no parentheses -> composition.formula -> parenthesis regex",
       replace it with a direct call to the _split_a_b_sites physical fallback.
    Returns True if the file was modified and written back; False if no modification was needed/possible.
    """
    with open(plugin_path, "r", encoding="utf-8") as f:
        src = f.read()
    orig_src = src

    if "def _split_a_b_sites" not in src:
        anchor = "class FeatureCalculator("
        if anchor not in src:
            return False
        src = src.replace(anchor, _SPLIT_A_B_SITES_SNIPPET + anchor, 1)

    new_src, n = _PATCH_NO_PAREN_RE.subn(_PATCH_NO_PAREN_REPL, src, count=1)
    if n == 0:
        return False

    with open(plugin_path, "w", encoding="utf-8") as f:
        f.write(new_src)
    return True


def run_plugin_smoke_test(plugin_path: str, sample_formulas=None):
    """Programmatic smoke test: import the just-generated plugin -> call predict with sample compositions -> verify the output keys are canonical and values are not constant-0/garbage.
    Raises RuntimeError (with fix guidance) on failure, which the caller must surface explicitly; returns a validation-summary dict on success.
    This is a root-cause defense: not relying only on LLM compliance, the generated plugin must pass this validation to be usable.
    """
    import importlib.util
    import re as _re
    import numpy as _np

    # Do not assign default samples up front: pass None to _select_smoke_formulas for system-aware selection,
    # to avoid false missing-attribute/feature-failure errors when the default BaTiO3/SrTiO3 is used for systems outside its constant table such as A2B2O7.
    if sample_formulas is None:
        sample_formulas = None

    if not os.path.exists(plugin_path):
        raise RuntimeError(f"Plugin file not found: {plugin_path}; smoke test cannot run")

    # ---- 1) Read the source; truncate the trailing register_plugin(...) call to avoid polluting the global registry ----
    with open(plugin_path, "r", encoding="utf-8") as f:
        src = f.read()
    idx = src.rfind("register_plugin(")
    if idx >= 0:
        src = src[:idx] + "pass\n"

    ns = {
        "__name__": "smoke_" + os.path.splitext(os.path.basename(plugin_path))[0],
        "__file__": plugin_path,
    }
    try:
        exec(compile(src, plugin_path, "exec"), ns)
    except Exception as e:
        raise RuntimeError(f"Plugin import failed (syntax/dependency error): {type(e).__name__}: {e}") from e

    # System-aware sample selection: avoid testing A2B2O7 plugins with BaTiO3/SrTiO3 (constant tables lack Ba/Sr) and falsely reporting missing attributes
    sample_formulas = _select_smoke_formulas(ns, sample_formulas)

    # Module-level constant-table absence diagnosis: if no DataFrame constant table is found at module level, and the source's
    # __init__ contains file IO (a typical symptom of the LLM violating constraints e/i), give explicit
    # guidance instead of silently falling back to default samples and ending with a confusing signal such as "outputs are completely identical".
    if _extract_constant_table_elements(ns) is None:
        import ast as _ast_smoke
        _bad_io = set()
        try:
            _tree = _ast_smoke.parse(src)
            for _node in _ast_smoke.walk(_tree):
                if isinstance(_node, _ast_smoke.FunctionDef) and _node.name == "__init__":
                    for _sub in _ast_smoke.walk(_node):
                        if not isinstance(_sub, _ast_smoke.Call):
                            continue
                        _fn = _sub.func
                        _nm = (
                            _fn.attr
                            if isinstance(_fn, _ast_smoke.Attribute)
                            else (_fn.id if isinstance(_fn, _ast_smoke.Name) else "")
                        )
                        if _nm.lower() in (
                            "read_csv", "read_excel", "read_table", "read_pickle",
                            "read_json", "load", "load_model", "load_weights", "open",
                        ):
                            _bad_io.add(_nm)
        except Exception:
            pass
        if _bad_io:
            raise RuntimeError(
                "Constant table is not defined at module level, and FeatureCalculator.__init__ performs file I/O "
                f"calling {sorted(_bad_io)} (violates converter constraint e/i). The constant table must be loaded with "
                "pd.read_csv at module level (top of file); __init__ may only reference it via a lightweight assignment"
                "(e.g. self.feature_table = CONSTANT_TABLE). Please regenerate the plugin or fix the code, "
                "and forbid any file reads inside __init__."
            )

    calc_cls = ns.get("FeatureCalculator")
    pred_cls = ns.get("Predictor")
    if calc_cls is None or pred_cls is None:
        raise RuntimeError(
            "Plugin is missing FeatureCalculator / Predictor classes; smoke test could not run."
            "Ensure the generated code defines both classes (inheriting the plugins.base base class)."
        )

    try:
        calc = calc_cls()
        pred = pred_cls()
    except Exception as e:
        raise RuntimeError(f"Plugin instantiation failed: {type(e).__name__}: {e}") from e

    # ---- 2) Feature-key alignment check: the keys produced by calculate must match the model's feature_cols ----
    # This is the most common root cause of "completely identical outputs for different compositions": the LLM defines custom placeholder keys feat1~feat8,
    # and predict does .get(k, 0.0) against model.pkl's feature_cols, so everything falls back to 0 -> constant output.
    try:
        required = list(calc.get_required_features() or [])
    except Exception:
        required = []
    model_feats = list(getattr(pred, "feature_cols", None) or [])
    if required and model_feats and set(required) != set(model_feats):
        raise RuntimeError(
            f"Feature names/count mismatch with model: FeatureCalculator.get_required_features()={required}, "
            f"model feature_cols loaded by Predictor={model_feats}."
            "Feature keys must strictly match model.pkl feature_cols (e.g. "
            "['Mv_Me','Sx_ca_minus_Sx_an','Sr_a_minus_Sr_b']); do NOT define custom "
            "feat1~featN placeholders - they make predict fall back to 0 and output constants."
        )

    # ---- 3) Build sample structures and predict ----

    results = {}
    for formula in sample_formulas:
        try:
            structure = comp_str_to_structure(formula)
            feats = calc.calculate(structure, {})
            if required:
                missing = [k for k in required if k not in feats]
                if missing:
                    raise RuntimeError(
                        f"Sample {formula} is missing feature keys {missing} (required={required})."
                        "calculate() must return all required feature keys; "
                        "missing ones make predict fall back to 0 and output constants."
                    )
        except Exception as e:
            if "Cannot parse A-site" in str(e):
                patched = _auto_patch_no_parenthesis_fallback(plugin_path)
                if patched:
                    print(
                        "[smoke-test] Detected parsing failure for bracket-free formula; auto-injected "
                        "_split_a_b_sites physics fallback and retried the smoke test"
                    )
                    return run_plugin_smoke_test(plugin_path, None)
            raise RuntimeError(
                f"Sample {formula} feature computation failed: {type(e).__name__}: {e}. "
                "If it is a missing-column/attribute error, check the mapping from feature columns to Element_Property_Table.csv; "
                "if it is 'Cannot parse A-site', implement the _split_a_b_sites physical-rule fallback from prompt constraint h for bracket-free formulas (e.g. BaTiO3/SrTiO3); "
                "never apply bracket regex to bracket-free strings; "
                "never silently return 0."
            ) from e
        try:
            result = pred.predict(feats)
        except Exception as e:
            raise RuntimeError(
                f"Sample {formula} prediction failed: {type(e).__name__}: {e}. "
                "Check that predict() reads the features dict consistently with the model feature order."
            ) from e
        results[formula] = (feats, result)

    # ---- 3) Validate output-key conventions (<prop>_mean / <prop>_dev) ----
    first_feats, first_result = results[sample_formulas[0]]
    if not isinstance(first_result, dict):
        raise RuntimeError(
            f"predict() returned wrong type: {type(first_result).__name__}; expected dict."
            "Output keys must follow <prop_key>_mean / <prop_key>_dev (e.g. tec_mean/tec_dev); do not use property_mean."
        )
    mean_keys = [k for k in first_result.keys() if k.endswith("_mean")]
    if not mean_keys:
        raise RuntimeError(
            f"predict() output keys invalid: no *_mean key found, actual keys={list(first_result.keys())}. "
            "Output keys must follow <prop_key>_mean / <prop_key>_dev (e.g. tec_mean/tec_dev); do not use property_mean/property_dev."
        )
    mkey = mean_keys[0]
    dkey = mkey.replace("_mean", "_dev")
    if dkey not in first_result:
        raise RuntimeError(
            f"predict() output keys invalid: found {mkey} but missing its counterpart {dkey}. "
            "Output keys must provide <prop_key>_mean paired with <prop_key>_dev."
        )
    if mkey == "property_mean":
        raise RuntimeError(
            "predict() used the deprecated property_mean/property_dev keys. "
            "Must switch to <prop_key>_mean / <prop_key>_dev (e.g. tec_mean/tec_dev)."
        )

    # ---- 4) Validate that values are not constant-0 / garbage ----
    means = []
    for formula in sample_formulas:
        _, result = results[formula]
        mean = result.get(mkey)
        dev = result.get(dkey)
        if mean is None or dev is None:
            note = result.get("note", "")
            raise RuntimeError(
                f"Sample {formula} returned None (mean={mean!r}, dev={dev!r}, note={note!r}). "
                "Check that the model loading and predict paths are correct."
            )
        try:
            mean = float(mean)
            dev = float(dev)
        except (TypeError, ValueError):
            raise RuntimeError(f"Sample {formula} output is non-numeric: mean={mean!r}, dev={dev!r}") from None
        if not _np.isfinite(mean) or not _np.isfinite(dev):
            raise RuntimeError(f"Sample {formula} output contains NaN/Inf: mean={mean!r}, dev={dev!r}")
        if abs(mean) < 1e-12:
            raise RuntimeError(
                f"Sample {formula} output is constantly 0 (mean={mean}). "
                "Feature computation seems to be constantly 0: check that features are driven by real feature_cols, that column names map to "
                "Element_Property_Table.csv, and that lookup failure does not silently return 0."
            )
        if abs(mean) > 1e15:
            raise RuntimeError(f"Sample {formula} output magnitude is abnormal: mean={mean} (suspected garbage value).")
        means.append(mean)

    if len(means) >= 2 and abs(means[0] - means[1]) < max(1e-9, abs(means[0]) * 0.01):
        raise RuntimeError(
            f"Different compositions produce identical outputs ({means}). This is a typical symptom of degenerate features (e.g. placeholder/constant-zero features fix the model output to constant garbage), "
            "check whether feature computation truly reflects composition differences."
        )

    return {
        "plugin_path": plugin_path,
        "output_key": mkey,
        "means": dict(zip(sample_formulas, means)),
        "ok": True,
    }


def save_plugin_code(
    plugin_code: str, plugin_name: str, constant_files=None
) -> str:
    """Save the plugin code to a file, auto-deploy the constant files, and run the programmatic smoke test.

    constant_files: the user-provided constant-file list (dict list, containing name/path),
    automatically copied on save to plugins/external/<plugin_name>_constants/,
    making the plugin self-contained (a module-level read_csv is enough; no manual placement by the user).
    Raises RuntimeError (with fix guidance) on validation failure; the caller must surface it explicitly and must not silently save an unusable plugin.
    """
    output_dir = os.path.join(current_dir, "plugins", "external")
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c for c in plugin_name if c.isalnum() or c in "_-")
    output_path = os.path.join(output_dir, f"{safe_name}.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plugin_code)
    # Auto-deploy constant files to the plugin-level <plugin_name>_constants/ directory
    deploy_info, deploy_err = deploy_constant_files(constant_files, output_path)
    if deploy_err:
        print(f"[deploy-constants] WARN: {deploy_err}")
    elif deploy_info:
        print(f"[deploy-constants] {deploy_info}")
    # Programmatic smoke test: import -> predict -> validate output keys + non-constant-0/non-garbage values
    smoke = run_plugin_smoke_test(output_path)
    print(f"[smoke-test] PASS: {smoke['output_key']} means={smoke['means']}")
    return output_path

if __name__ == "__main__":
    main()
