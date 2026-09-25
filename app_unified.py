import os

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================================
# [Console mute] Suppress three non-error messages from third-party libraries/frameworks (without modifying any files inside site-packages)
#   1) torch UserWarning：Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected ...
#      Source: mace/__init__.py sets this env var on import; then e3nn/o3/_wigner.py calls
#      torch.load during import without explicitly passing weights_only, so torch raises this warning.
#   2) mace/tools/cg.py outputs via a module-level print() when cuequivariance is unavailable:
#      "cuequivariance or cuequivariance_torch is not available. ..."
#   3) streamlit module-scan warning: "Examining the path of torch.classes raised: ..."
#      Source: streamlit/watcher/local_sources_watcher.py line 207. Every time the script finishes (or is interrupted and rerun)
#      streamlit rescans sys.modules, and its "namespace package path" extractor reads torch.classes.__path__._path,
#      torch's __path__ is a pseudo-namespace object, so the C++ layer raises RuntimeError (not AttributeError),
#      which is therefore recorded as a warning by the function's generic except clause. Diagnostic noise with no functional impact; suppressed by message prefix only.
# Must be placed before any statement that indirectly imports mace / e3nn (e.g. the import inverse_design below).
# =====================================================================
import warnings as _warnings

# (1) Precisely filter the torch warning above by message content; other warnings are unaffected
_warnings.filterwarnings(
    "ignore",
    message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected",
    category=UserWarning,
)

# (2) The cuequivariance notice is a print (not logging), so it can only be muted by redirecting stdout.
#     That print runs only on the first module import, so we perform a "silent pre-import" here once;
#     subsequent imports by the project (inverse_design.py -> mace.calculators) hit the sys.modules cache and emit nothing.
try:
    import contextlib as _contextlib
    import io as _io
    import importlib as _importlib

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with _contextlib.redirect_stdout(_io.StringIO()):
            _importlib.import_module("mace.tools.cg")
except Exception:
    pass

# (3) The streamlit module-scan warning is emitted via logging (stderr); filter exactly this one by message prefix;
#     other warnings on the same logger, as well as the project's own console output, are unaffected.
import logging as _logging

_logging.getLogger("streamlit.watcher.local_sources_watcher").addFilter(
    lambda record: not record.getMessage().startswith("Examining the path of torch.classes")
)

import streamlit as st
import pandas as pd
import time
import sys
import re
import requests
import shutil
import base64
import joblib
import numpy as np
import config
import warnings
import logging
import io

# Inverse design module (optional import)
try:
    import inverse_design

    HAS_INVERSE_DESIGN = True
except Exception as e:
    HAS_INVERSE_DESIGN = False


# Replace stderr immediately
class WarningFilter:
    def write(self, text):
        if "torch.classes" in text or "Examining the path" in text:
            return
        sys.__stdout__.write(text)

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        return 0


sys.stderr = WarningFilter()
sys.stdout = WarningFilter()

# Configure the root logging level
for logger_name in [
    "transformers",
    "torch",
    "sentence_transformers",
    "huggingface",
    "transformers.modeling_utils",
    "transformers.model",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

# Disable all warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Configure Streamlit logging
logging.basicConfig(level=logging.CRITICAL)

if "initialized" not in st.session_state:
    # Keep the necessary state, clear the rest
    keys_to_keep = ["initialized", "chat_history"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.initialized = True

# Ensure chat_history exists
if not hasattr(st.session_state, "chat_history"):
    st.session_state.chat_history = []

os.environ["HF_TOKEN"] = config.HF_TOKEN if hasattr(config, "HF_TOKEN") else ""
os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
os.chdir(current_dir)

import viz_component
from orchestrator import CentralOrchestrationAgent
from nlp_processor import NLPProcessor


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


DEEPSEEK_API_KEY = config.DS_API_KEY
DEEPSEEK_API_URL = config.DEEPSEEK_API_URL
DEEPSEEK_MODEL = config.DS_MODEL


def extract_text_from_pdf(pdf_file):
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


def call_deepseek_conversion(prompt, context=""):
    if not DEEPSEEK_API_KEY:
        return None, "[!] DeepSeek API Key not configured"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    full_prompt = prompt
    if context:
        full_prompt = f"""## Reference Paper Content
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

    for attempt in range(3):
        try:
            response = requests.post(
                f"{DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"], None
            elif response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None, f"[!] API Error: {response.status_code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return None, f"[!] Request failed: {str(e)}"
    return None, "[!] Max retries exceeded"


def extract_python_code(response_text):
    if not response_text:
        return ""
    if "```python" in response_text:
        start = response_text.find("```python") + 9
        end = response_text.rfind("```")
        return response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.rfind("```")
        return response_text[start:end].strip()
    return response_text.strip()


def create_pdf_export(content_lines, title="DiCerAgent Report"):
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
        if not line:
            story.append(Spacer(1, 4))
        elif "=" in line and len(line) < 50:
            story.append(Paragraph(line.replace("=", "").strip(), title_style))
        elif line.startswith("###"):
            story.append(Paragraph(line.replace("###", "").strip(), styles["Heading2"]))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _get_app_converter():
    import importlib

    return importlib.import_module("app_converter")


st.set_page_config(
    page_title="DiCerAgent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEEPSEEK_API_KEY = config.DS_API_KEY
DEEPSEEK_API_URL = config.DEEPSEEK_API_URL
DEEPSEEK_MODEL = config.DS_MODEL
EXTERNAL_MODELS_DIR = os.path.join(current_dir, "external_models")


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
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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


def _find_composition_col(data_df, target_col):
    """Locate the formula/composition text column in the training data. Returns the column name or None."""
    text_cols = []
    for c in data_df.columns:
        if c == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(data_df[c]):
            text_cols.append(c)
    for c in text_cols:
        cl = str(c).lower()
        if "composition" in cl or "formula" in cl or "chem" in cl:
            return c
    return text_cols[0] if text_cols else None


def _auto_compute_features_from_composition(converted_code, comp_col, data_df,
                                            cfg_feats, fake_file_path,
                                            const_files=None):
    """When the training CSV has only a composition column (no numeric feature columns),
    reuse the FeatureCalculator in the generated plugin code to compute features from formulas
    using the constant tables and the original code's calculation method.
    Returns (feat_df, feats_ok):
      feat_df : feature DataFrame aligned to cfg_feats (row order consistent with data_df)
      feats_ok: list of feature column names actually usable for training (numeric and in cfg_feats)
    Raises an exception with an explicit reason on failure.
    """
    import ast as _ast
    import re as _re

    # --- Auto-patch: feeding the composition string to a parenthesis regex for A/B-site parsing in a plugin will inevitably fail ---
    # pymatgen's formula / reduced_formula never contains parentheses, e.g.
    #   (Y0.2Gd0.2Er0.2Yb0.2Lu0.2)2Zr2O7 -> reduced_formula -> Yb2Gd2Y2Er2Lu2Zr10O35
    # so all such accesses are unified to "prefer structure._orig_formula (preserving the user's original parenthesized form)":
    #   getattr(structure, '_orig_formula', None) or structure.composition.reduced_formula
    # Covers common patterns: structure.composition.reduced_formula / .formula /
    # str(structure.composition) / bare composition.xxx / local-variable alias comp.xxx.
    # Order: handle the long forms (with .reduced_formula/.formula) first, then the short str(composition) form,
    # to avoid re-replacing structure.composition.xxx inside the replacement result.
    _orig_getter = (
        "getattr(structure, '_orig_formula', None) "
        "or structure.composition.reduced_formula"
    )
    _converted_code = converted_code
    for _pattern in (
        r"structure\.composition\.reduced_formula",
        r"structure\.composition\.formula",
        r"str\(structure\.composition\)",
        r"(?<!structure\.)composition\.reduced_formula",
        r"(?<!structure\.)composition\.formula",
        r"str\(composition\)",
        r"(?<![a-zA-Z_.])comp\.reduced_formula",
        r"(?<![a-zA-Z_.])comp\.formula",
        r"str\(comp\)",
        r"f\"\{structure\.composition\}\"",
        r"f'\{structure\.composition\}'",
    ):
        _converted_code = _re.sub(_pattern, _orig_getter, _converted_code)

    tree = _ast.parse(_converted_code)

    # --- Static validation of the module-level constant-table convention (generic root-cause defense, not relying on LLM compliance) ---
    # When LLM generation is unstable, it often writes file IO such as pd.read_csv / open / np.load into
    # FeatureCalculator.__init__ (violating prompt constraints e/i), which causes:
    #   1) In the training branch, after exec, instantiating FeatureCalculator() reads a file immediately - if it reads the training CSV
    #      (no Element column) it crashes with "None of ['Element'] are in the columns";
    #   2) Constant table not at module level -> smoke test _extract_constant_table_elements cannot find the
    #      DataFrame -> falls back to default samples -> features invalid -> identical outputs for different compositions.
    # Here, before exec, we use AST to detect file-IO calls inside __init__ and give explicit guidance,
    # instead of leaving a vague runtime crash to the user. Applies to all external model plugins.
    for _node in _ast.walk(tree):
        if isinstance(_node, _ast.FunctionDef) and _node.name == "__init__":
            _bad_io = set()
            for _sub in _ast.walk(_node):
                if not isinstance(_sub, _ast.Call):
                    continue
                _fn = _sub.func
                _nm = (
                    _fn.attr
                    if isinstance(_fn, _ast.Attribute)
                    else (_fn.id if isinstance(_fn, _ast.Name) else "")
                )
                if _nm.lower() in (
                    "read_csv", "read_excel", "read_table", "read_pickle",
                    "read_json", "load", "load_model", "load_weights", "open",
                ):
                    _bad_io.add(_nm)
            if _bad_io:
                raise ValueError(
                    "converted plugin code violates the module-level constants "
                    f"convention: FeatureCalculator.__init__ contains file I/O calls "
                    f"{sorted(_bad_io)}. Constant tables MUST be loaded AT MODULE "
                    "LEVEL (pd.read_csv at the top of the file, per converter "
                    "constraint e/i), and __init__ must only reference them via "
                    "lightweight assignment (e.g. self.feature_table = CONSTANT_TABLE). "
                    "Please re-generate the plugin or fix the generated code so no "
                    "read_csv/open/load appears inside __init__."
                )

    segs = []
    for node in tree.body:
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            segs.append(_ast.get_source_segment(_converted_code, node))
        elif isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            segs.append(_ast.get_source_segment(_converted_code, node))
        elif isinstance(node, _ast.FunctionDef):
            segs.append(_ast.get_source_segment(_converted_code, node))
        elif isinstance(node, _ast.ClassDef) and node.name == "FeatureCalculator":
            segs.append(_ast.get_source_segment(_converted_code, node))
    if not any("class FeatureCalculator" in s for s in segs):
        raise ValueError("converted plugin code does not contain a FeatureCalculator class")

    # --- Auto-deploy constant files (AUTO-DEPLOYED convention): the plugin reads CSVs at module level from
    # the <plugin_name>_constants/ directory; the training branch computes features automatically without going through Save Plugin,
    # so that directory may not exist yet. Here we copy const_files to the directory of __file__ before exec,
    # ensuring pd.read_csv can read them (same deployment path as Save Plugin; safe to overwrite repeatedly). ---
    if const_files:
        try:
            _deploy_base = os.path.dirname(os.path.abspath(fake_file_path))
            _deploy_dirs = set()
            for _node in _ast.walk(tree):
                if (
                    isinstance(_node, _ast.Constant)
                    and isinstance(_node.value, str)
                    and _node.value.endswith("_constants")
                ):
                    _deploy_dirs.add(_node.value)
            if not _deploy_dirs:
                _deploy_dirs.add(
                    os.path.splitext(os.path.basename(fake_file_path))[0]
                    + "_constants"
                )
            for _cd in _deploy_dirs:
                _target_dir = os.path.join(_deploy_base, _cd)
                os.makedirs(_target_dir, exist_ok=True)
                for _cf in const_files:
                    _p = _cf.get("path") if isinstance(_cf, dict) else _cf
                    if _p and os.path.exists(_p):
                        shutil.copy2(
                            _p, os.path.join(_target_dir, os.path.basename(_p))
                        )
        except Exception as _deploy_exc:
            print(f"[warn] auto-deploy const files failed: {_deploy_exc}")

    ns = {"__file__": fake_file_path, "__name__": "_auto_feature_calc"}
    exec("\n\n".join(segs), ns)

    # --- Constant-table fallback normalization: CONSTANT_TABLE in LLM-generated plugins is often written as
    # pd.DataFrame({'Element': [...], ...}) but misses set_index('Element')
    # (or writes CONSTANT_TABLE.set_index('Element') without assignment/inplace),
    # so .index is a RangeIndex and every lookup by element symbol misses,
    # reporting "missing element attributes: [...]". Here we uniformly rebuild constant-table DataFrames containing an 'Element' column
    # to be indexed by element symbol, compatible with both forms. ---
    for _k in list(ns.keys()):
        _v = ns[_k]
        if isinstance(_v, pd.DataFrame) and "Element" in _v.columns:
            try:
                if not (
                    _v.index.name == "Element"
                    or (
                        len(_v.index) > 0
                        and isinstance(_v.index[0], str)
                    )
                ):
                    ns[_k] = _v.set_index("Element")
            except Exception:
                pass

    # --- Fill in missing constant-table rows: LLM-generated plugins hard-code the constant table into the code (CONSTANT_TABLE
    # etc.) and, due to truncation/omission, often write only part of the element rows (e.g. missing trailing Hf/Eu/Sn), causing
    # "missing element attributes: ['Eu']" on element lookup. Here, if the constant table is missing elements of the composition and
    # the original constant file (const_files, element-attribute CSV) is provided, the missing rows are filled in from the CSV
    # instead of raising directly. Filling prefers column-name mapping (same name -> normalized-name match -> positional match). ---
    if const_files:
        try:
            _elem_in_comps = set()
            for _cs in data_df[comp_col].astype(str):
                _elem_in_comps.update(_re.findall(r"[A-Z][a-z]?", _cs))
            if _elem_in_comps:
                _csv_paths = []
                for _cf in const_files:
                    _p = _cf.get("path") if isinstance(_cf, dict) else None
                    if _p and os.path.exists(_p):
                        _csv_paths.append(_p)
                if _csv_paths:
                    for _k in list(ns.keys()):
                        _v = ns[_k]
                        if not isinstance(_v, pd.DataFrame):
                            continue
                        _idx = _v.index
                        if not (_idx.name == "Element" or (
                            len(_idx) > 0 and isinstance(_idx[0], str)
                        )):
                            continue
                        _missing = sorted(
                            e for e in _elem_in_comps
                            if e not in set(map(str, _idx))
                        )
                        if not _missing:
                            continue
                        for _cp in _csv_paths:
                            try:
                                _raw = None
                                for _enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
                                    try:
                                        _raw = pd.read_csv(_cp, encoding=_enc)
                                        break
                                    except Exception:
                                        continue
                                if _raw is None:
                                    continue
                                _elem_col = None
                                for _c in _raw.columns:
                                    _n = str(_c).strip().lower().replace(" ", "")
                                    if _n in ("element", "symbol", "elementsymbol"):
                                        _elem_col = _c
                                        break
                                if _elem_col is None:
                                    _elem_col = _raw.columns[0]
                                _raw_idx = _raw[_elem_col].astype(str).str.strip()
                                _raw2 = _raw.drop(columns=[_elem_col])
                                _raw2.index = _raw_idx
                                # Column-name mapping: same name -> normalized-name match -> positional match
                                _col_map = {}
                                for _c in _v.columns:
                                    if _c in _raw2.columns:
                                        _col_map[_c] = _c
                                if len(_col_map) < len(_v.columns):
                                    _norm_raw = {
                                        str(_c).lower().replace(" ", "").replace("(", "").replace(")", ""): _c
                                        for _c in _raw2.columns
                                    }
                                    for _c in _v.columns:
                                        if _c in _col_map:
                                            continue
                                        _nk = str(_c).lower().replace(" ", "").replace("(", "").replace(")", "")
                                        if _nk in _norm_raw:
                                            _col_map[_c] = _norm_raw[_nk]
                                if len(_col_map) < len(_v.columns) and len(_raw2.columns) == len(_v.columns):
                                    _raw_cols = list(_raw2.columns)
                                    for _j, _c in enumerate(_v.columns):
                                        if _c not in _col_map:
                                            _col_map[_c] = _raw_cols[_j]
                                _rows_new = []
                                _elements_found = []
                                for _el in _missing:
                                    if _el not in _raw2.index:
                                        continue
                                    _row = {}
                                    for _c in _v.columns:
                                        _src = _col_map.get(_c)
                                        _row[_c] = (
                                            _raw2.loc[_el, _src]
                                            if _src is not None and _src in _raw2.columns
                                            else float("nan")
                                        )
                                    _rows_new.append(_row)
                                    _elements_found.append(_el)
                                if _rows_new:
                                    _new_df = pd.DataFrame(
                                        _rows_new, index=_elements_found
                                    )
                                    ns[_k] = pd.concat([_v, _new_df])
                                    break
                            except Exception:
                                continue
        except Exception:
            pass

    # --- Column-name alias injection: FeatureCalculator generated by LLM often looks up tables with short column names
    # (e.g. 'velocity', 'young'), while the embedded constant-table column names are the original long CSV names
    # (e.g. 'Velocity Of Sound (m/s)'), causing KeyError: 'velocity'. Here we extract the string literals referenced
    # in the calculate source; if the constant table lacks that column but has a unique "containing match" (normalized
    # mutual substring, e.g. 'velocity' is a substring of 'velocity of sound (m/s)'), we copy that column and append it
    # as an alias column so both short/long names can be looked up. False-positive matches are harmless (unused unless referenced). ---
    try:
        _calc_node = None
        for _node in _ast.walk(tree):
            if isinstance(_node, _ast.ClassDef) and _node.name == "FeatureCalculator":
                _calc_node = _node
                break
        if _calc_node is not None:
            _lit_names = set()
            for _node in _ast.walk(_calc_node):
                if isinstance(_node, _ast.Constant) and isinstance(_node.value, str):
                    _s = _node.value.strip()
                    # Only consider string literals that look like column names: 3-40 chars, not purely numeric, not an element symbol (<=2 chars)
                    if 3 <= len(_s) <= 40 and not _s.replace(".", "").isdigit():
                        _lit_names.add(_s)
            for _k in list(ns.keys()):
                _v = ns[_k]
                if not isinstance(_v, pd.DataFrame):
                    continue
                _idx = _v.index
                if not (_idx.name == "Element" or (
                    len(_idx) > 0 and isinstance(_idx[0], str)
                )):
                    continue
                _existing = set(map(str, _v.columns))
                _col_norm = {
                    str(_c).lower().replace(" ", "").replace("(", "").replace(")", "").replace("'", ""): _c
                    for _c in _v.columns
                }
                _added = False
                for _cand in sorted(_lit_names):
                    if _cand in _existing:
                        continue
                    _cn = _cand.lower().replace(" ", "").replace("(", "").replace(")", "").replace("'", "")
                    if not _cn:
                        continue
                    _hits = [
                        _c for _n, _c in _col_norm.items()
                        if _n != _cn and (_cn in _n or _n in _cn)
                    ]
                    if len(_hits) == 1:
                        _v[_cand] = _v[_hits[0]]
                        _existing.add(_cand)
                        _added = True
                if _added:
                    ns[_k] = _v
    except Exception:
        pass

    calc = ns["FeatureCalculator"]()

    from pymatgen.core import Composition as _Composition, Structure as _Structure

    def _normalize_frac_formula(s):
        """Normalize fractional coefficients in composition text (e.g. La1/2, Sn1/3) into
        decimal forms parseable by pymatgen. pymatgen does not accept '1/2'/'1/3' fraction forms,
        and Composition('(La1/2Pr1/2)2...') raises '/2/2 is an invalid formula!'.
        Use full-precision floats (.17g) so the subsequent Fraction(float).limit_denominator
        can still exactly recover the minimal integer ratio (e.g. 1/3 -> 0.3333333333333333 -> Fraction(1,3)).

        In the data source, a fractional coefficient may be immediately followed by a redundant decimal
        (e.g. Eu1/30.33 is actually Eu1/3 followed by a redundant approximate value 0.33), so first match
        "fraction + decimal" as a whole, take the shortest denominator, and drop the redundant decimal
        """
        def _rep(m):
            num, den = int(m.group(2)), int(m.group(3))
            return m.group(1) + ("%.17g" % (num / den))

        # Parenthesis repair: the data source may drop a left parenthesis (e.g. La0.2Gd0.2Y0.2Yb0.2Er0.2)2(...)),
        # so restore the opening '(' of the group corresponding to the orphaned right ')', avoiding pymatgen's ')2 is an invalid formula!' error.
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

        # Consecutive writing where a fractional coefficient is immediately followed by a redundant decimal (Eu1/30.33 is actually Eu1/3 + redundant 0.33):
        # take the units digit of the denominator, and it must be immediately followed by a digit or decimal point (indicating it is glued to the trailing decimal),
        # normalize the fraction and then drop the glued redundant decimal segment.
        s = _re.sub(r"([A-Za-z])(\d+)/(\d)(?=[.\d])", _rep, s)
        # Normalize ordinary fractional coefficients (La1/2, Sn1/3, etc.)
        s = _re.sub(r"([A-Za-z])(\d+)/(\d+)", _rep, s)
        # Glued-digit fallback repair: two complete decimals directly concatenated (e.g. 0.33333333333333330.33) take the first;
        # malformed digit-pairs with a dot in between (e.g. 0.33.44) take the first.
        s = _re.sub(r"(\d+\.\d+)(\d+\.\d+)", r"\1", s)
        s = _re.sub(r"(\d+\.\d+)\.(\d+)", r"\1", s)
        return s

    def _comp_str_to_structure(comp_str):
        """Convert the composition text into a minimal pymatgen Structure (element ratios consistent with the formula).

        The plugin FeatureCalculator.calculate interface requires a pymatgen Structure object
        (internally it obtains the element composition via structure.composition.get_el_amt_dict()),
        so a string cannot be passed directly. Here we place atoms by the minimal integer ratio of the formula's elements,
        ensuring the element species and molar ratios of the composition match the original formula.
        Supports fractional coefficient forms (La1/2, Sn1/3, etc.), normalized to decimals before parsing.
        """
        comp = _Composition(_normalize_frac_formula(str(comp_str).strip()))
        _norm_str = _normalize_frac_formula(str(comp_str).strip())
        reduced = comp.reduced_composition
        # Convert each element amount to the minimal integer ratio (avoid round dropping decimals like 0.4 to 1)
        from fractions import Fraction
        import math
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
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        structure = _Structure(lattice, species, coords, coords_are_cartesian=True)
        # Attach the original composition string: plugins may prefer getattr(structure, '_orig_formula', None)
        # to parse via the original parenthesized form (pymatgen's formula has no parentheses; a bare composition cannot recover A/B grouping)
        try:
            structure._orig_formula = _norm_str
        except Exception:
            pass
        return structure

    class _OrigFormulaComposition:
        """Wrapper for Composition: formula/reduced_formula/str() all return the original parenthesized form;
        other methods (get_el_amt_dict etc.) are passed through unchanged."""
        def __init__(self, composition, orig_formula):
            self._comp = composition
            self._orig_formula = orig_formula
        @property
        def formula(self):
            return self._orig_formula
        @property
        def reduced_formula(self):
            return self._orig_formula
        def __str__(self):
            return self._orig_formula
        def __repr__(self):
            return self._orig_formula
        def __getattr__(self, name):
            return getattr(self._comp, name)

    class _OrigFormulaStructure:
        """Wrapper for Structure: composition returns a proxy Composition (stringified = original parenthesized form),
        other attributes/methods are passed through."""
        def __init__(self, structure, orig_formula):
            self._structure = structure
            self._orig_formula = orig_formula
        @property
        def composition(self):
            return _OrigFormulaComposition(
                self._structure.composition, self._orig_formula
            )
        def __getattr__(self, name):
            return getattr(self._structure, name)

    rows = {}
    for i, comp_str in enumerate(data_df[comp_col].astype(str)):
        try:
            structure = _comp_str_to_structure(comp_str)
        except Exception as _ce:
            raise ValueError(
                f"failed to parse composition '{comp_str}' into a pymatgen "
                f"Structure (the plugin's calculate() expects a Structure "
                f"object, not a raw string): {_ce}"
            ) from _ce
        try:
            d = calc.calculate(_OrigFormulaStructure(structure, comp_str), {})
        except KeyError as _ke:
            _tbl_desc = []
            for _tk, _tv in ns.items():
                if isinstance(_tv, pd.DataFrame):
                    _tbl_desc.append(
                        f"{_tk}(cols={list(_tv.columns)[:15]})"
                    )
            raise KeyError(
                f"{_ke} raised inside FeatureCalculator.calculate for "
                f"composition '{comp_str}'. In-embed constant tables: "
                + ("; ".join(_tbl_desc) if _tbl_desc else "none")
            ) from _ke
        if not isinstance(d, dict):
            raise ValueError(
                f"FeatureCalculator.calculate returned {type(d).__name__} "
                f"for composition '{comp_str}'"
            )
        rows[i] = {k: v for k, v in d.items() if k in cfg_feats}
    if not rows:
        raise ValueError("no rows to compute features from")
    feat_df = pd.DataFrame.from_dict(rows, orient="index")
    if feat_df.empty:
        raise ValueError("feature computation produced an empty feature table")
    feats_ok = [
        f for f in cfg_feats
        if f in feat_df.columns and pd.api.types.is_numeric_dtype(feat_df[f])
    ]
    if not feats_ok:
        raise ValueError(
            "computed features contain no numeric columns matching "
            "TRAINING_CONFIG.feature_cols; got keys: "
            + ", ".join(map(str, list(feat_df.columns)[:10]))
        )
    return feat_df, feats_ok


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

st.markdown(
    """
<style>
    /* 1. Base font & global background - white */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* reserve the vertical scrollbar gutter so the content width (and every control's
       right edge) stays identical across pages, no 5.6px shift when switching tabs */
    html { scrollbar-gutter: stable; }
    .stApp { background-color: #ffffff; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    /* 2. Hide Streamlit default header/footer */
    header[data-testid="stHeader"] {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* 3. Logo & sidebar styles */
    .logo-container { display: flex; align-items: center; gap: 16px; margin-bottom: 8px; margin-top: 0; }
    .logo-img { width: 46px !important; height: 46px !important; border-radius: 12px !important; display: block !important; }
    /* brand name: same size as page titles (30px) */
    .logo-text { font-size: 30px !important; font-weight: 700 !important; color: #1d1d1f !important; letter-spacing: -0.7px !important; line-height: 1.25 !important; display: inline-block !important; vertical-align: middle !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0px !important; }
    
    /* 4. Page container: compact top spacing so logo sits close to browser top */
    .block-container { 
        padding-top: 1.5rem !important;
    }
    
    .main-title { 
        font-size: 30px !important; 
        font-weight: 700 !important; 
        color: var(--ink) !important; 
        letter-spacing: -0.7px !important; 
        margin-top: 0px !important;    /* set to 0; controlled by container padding */
        margin-bottom: 8px !important; /* extra spacing below the subtitle */
    }
    
    .subtitle { 
        color: #8A9199 !important; 
        font-size: 15.5px !important; 
        font-weight: 400 !important; 
        letter-spacing: 0.1px !important;
        margin-top: 0px !important;    /* remove negative margin to fix crowding */
        margin-bottom: 24px !important; 
    }               
    
    /* 0. brand tokens - unified palette */
    :root {
        --brand: #1C6FB6;            /* ceramic blue primary */
        --brand-strong: #155894;     /* hover / pressed */
        --brand-soft: #E8F1FA;       /* light blue fill */
        --ink: #1d1d1f;              /* primary text */
        --muted: #86868b;            /* secondary text */
        --hairline: #e5e5e7;
    }
    .stApp a { color: var(--brand) !important; }
    
    /* 0b. unified card system (shared by pred-card & Settings/About) */
    .card,
    .pred-card {
        background-color: #ffffff !important;
        padding: 24px 26px !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 8px 24px rgba(16,24,40,0.045) !important;
        border: 1px solid #ECEEF2 !important;
        margin-bottom: 12px !important;
    }
    .pred-card { position: relative !important; }
    .card > div, .card > .stMarkdown, .card > p { 
        background-color: transparent !important;
    }
    
    /* 0c. page divider under page titles */
    .page-divider {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, var(--brand) 0%, rgba(28,111,182,0.35) 45%, rgba(28,111,182,0.06) 85%, transparent 100%) !important;
        border-radius: 2px !important;
        margin: 2px 0 22px 0 !important;
    }
    
    /* 0d. KPI big cards on prediction page */
    .kpi-row { display: flex; gap: 14px; margin: 4px 0 20px 0; flex-wrap: wrap; }
    .kpi-card {
        flex: 1 1 150px;
        background: linear-gradient(135deg, #F4F8FC 0%, #EAF2FA 100%) !important;
        border: 1px solid #DCE7F2 !important;
        border-left: 4px solid var(--brand) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        min-width: 120px;
    }
    .kpi-label { font-size: 12.5px; font-weight: 600; color: #5E6B7A; margin-bottom: 6px; letter-spacing: 0.2px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: var(--ink); line-height: 1.05; }
    .kpi-dev { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 6px; }
    
    /* 5. Tabs styling - lightweight underline navigation (top-level nav) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important; background: transparent !important;
        display: flex; align-items: center;
        border-bottom: 1px solid var(--hairline) !important;
        padding-bottom: 0 !important; margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        /* equal width for every top-level tab: all aligned to the longest label ("Settings") */
        height: 46px !important; width: auto !important; min-width: 96px !important;
        flex: 0 0 auto !important; padding: 0 18px !important;
        background-color: transparent !important;
        border: none !important; border-radius: 8px 8px 0 0 !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        box-shadow: none !important;
        transition: background-color 0.18s ease, color 0.18s ease !important;
    }
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] div {
        margin: 0 !important; padding: 0 !important; line-height: 1 !important;
        font-size: 15px !important; font-weight: 500 !important; color: #5E6B7A !important;
        letter-spacing: 0.1px !important;
        transition: color 0.18s ease !important;
    }
    .stTabs [aria-selected="true"] { background-color: transparent !important; box-shadow: none !important; }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] div { color: var(--brand) !important; font-weight: 600 !important; }
    /* glowing underline indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--brand) !important;
        height: 2px !important; border-radius: 2px !important;
        box-shadow: 0 0 8px rgba(28,111,182,0.55) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] * { background-color: var(--brand) !important; }
    .stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) { background-color: rgba(28,111,182,0.05) !important; }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) p, .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) div { color: var(--brand) !important; }
    
    /* Sub-tabs inside tab panels (MPC / IDO) - revert to default Streamlit style */
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0 !important; }
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"] {
        height: auto !important; width: auto !important; min-width: 0 !important; flex: none !important; padding: 8px 16px !important;
        background-color: transparent !important; border-radius: 0 !important; box-shadow: none !important;
        transition: none !important;
    }
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"] p,
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"] div {
        color: var(--ink) !important; font-size: 16px !important; font-weight: 400 !important;
    }
    div[role="tabpanel"] div[data-testid="stTabs"] [aria-selected="true"],
    div[role="tabpanel"] div[data-testid="stTabs"] [aria-selected="true"] p {
        background-color: transparent !important; color: var(--brand) !important; font-weight: 600 !important;
    }
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab-highlight"] * { background-color: var(--brand) !important; }
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"]:hover:not([aria-selected="true"]),
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"]:hover:not([aria-selected="true"]) p,
    div[role="tabpanel"] div[data-testid="stTabs"] [data-baseweb="tab"]:hover:not([aria-selected="true"]) div {
        background-color: transparent !important; box-shadow: none !important; color: var(--brand) !important;
    }
    
    /* core fix: unified white card style - use specific class names to avoid affecting other tabs */
    /* 1. core: apply styles only when .pred-card class exists */
    .pred-card {
        background-color: #ffffff !important;
        padding: 24px !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
    }

    /* 2. penetrate: target only containers with .pred-card class */
    .pred-card > div[data-testid="stHtml"], 
    .pred-card > div {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 3. forced protection for 3D model iframes */
    iframe[title="stmol.showmol"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
    }

    /* 4. literature data card: ensure margin alignment */
    .lit-card-wrapper {
        margin-top: 20px !important;
    }

    /* 10. table styling */
    .pred-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .pred-table th { background: #f5f5f7 !important; padding: 14px 16px; text-align: left; border-bottom: 2px solid #e5e5e7; font-weight: 600; color: #1d1d1f; }
    .pred-table td { padding: 14px 16px; border-bottom: 1px solid #f0f0f0; color: #1d1d1f; }
    .pred-table .val { font-size: 22px; font-weight: 700; color: var(--brand) !important; }
    .pred-table .dev { color: var(--muted); font-size: 14px; margin-left: 6px; }

    /* 11. inputs & buttons - crisp geometry + focus glow */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        height: 46px !important;
        border-radius: 10px !important;
        border: 1px solid #DFE3E9 !important;
        background-color: #FCFDFE !important;
        box-shadow: inset 0 1px 2px rgba(16,24,40,0.03) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div { background-color: transparent !important; border: none !important; }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: var(--brand) !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(28,111,182,0.14), 0 0 14px rgba(28,111,182,0.16) !important;
    }
    div[data-testid="stTextInput"] input { font-size: 15px !important; color: var(--ink) !important; }
    div[data-testid="stTextInput"] input::placeholder { color: #A7AEB8 !important; }
    /* primary action button: gradient blue, flat & technical */
    div.stButton > button {
        height: 46px !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important;
        background: linear-gradient(135deg, #2E86D6 0%, #1C6FB6 55%, #155894 100%) !important;
        color: #ffffff !important; border: none !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.08), 0 0 0 1px rgba(28,111,182,0.16) !important;
        transition: filter 0.18s ease, box-shadow 0.22s ease, transform 0.18s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #3B93E0 0%, #2478C0 55%, #175F9F 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(28,111,182,0.28) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active { transform: translateY(0) !important; filter: brightness(0.97); }
    
    /* 11b. secondary options card: light bordered container that directly holds option widgets.
       NOTE: Streamlit DOM is BorderWrapper > div(anon) > stVerticalBlock > element-container > widget,
       so the :has() chain is depth-limited to avoid styling page-level wrappers. */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div > div[data-testid="stCheckbox"]):not([data-testid="stExpander"] div[data-testid="stVerticalBlockBorderWrapper"]) {
        background-color: #F8FAFD !important;
        border: 1px solid #E4EBF3 !important;
        border-radius: 14px !important;
        padding: 4px 18px 14px 18px !important;
        margin: 8px 0 20px 0 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.03) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div > div[data-testid="stCheckbox"]) div[data-testid="stCheckbox"] label p {
        font-size: 13.5px !important; font-weight: 500 !important; color: #46536B !important;
    }
    /* expander: quiet flat surface */
    div[data-testid="stExpander"] {
        border: 1px solid var(--hairline) !important;
        border-radius: 12px !important;
        background-color: #FBFCFE !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary { padding: 9px 14px !important; }
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {
        font-size: 13.5px !important; font-weight: 500 !important; color: #55617A !important;
        transition: color 0.18s ease !important;
    }
    div[data-testid="stExpander"] summary:hover p,
    div[data-testid="stExpander"] summary:hover span { color: var(--brand) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div > div[data-testid="stCheckbox"]) div[data-testid="stExpander"] {
        background-color: #ffffff !important; border-color: #E4EBF3 !important; margin-bottom: 2px !important;
    }

    /* 11c. typography hierarchy: deep-gray body, light-gray secondary */
    div[data-testid="stMarkdownContainer"] h1 { font-size: 26px !important; font-weight: 600 !important; color: var(--ink) !important; letter-spacing: -0.5px !important; }
    div[data-testid="stMarkdownContainer"] h2 { font-size: 20px !important; font-weight: 600 !important; color: #1F2733 !important; letter-spacing: -0.25px !important; margin-top: 26px !important; }
    div[data-testid="stMarkdownContainer"] h3 { font-size: 16.5px !important; font-weight: 600 !important; color: #2B3445 !important; letter-spacing: -0.1px !important; margin-top: 22px !important; }
    div[data-testid="stMarkdownContainer"] h4 { font-size: 15px !important; font-weight: 600 !important; color: #55617A !important; margin-top: 18px !important; }
    div[data-testid="stCaptionContainer"] p,
    div[data-testid="stCaptionContainer"] span,
    small { font-size: 13px !important; color: #98A1AE !important; }

    /* 12. other UI components */
    .section-header { font-size: 20px !important; font-weight: 600 !important; color: #1d1d1f !important; margin-top: 24px !important; margin-bottom: 16px !important; display: block; }
    .report-box { background: white; padding: 32px; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.06); }
    .success-msg { background: linear-gradient(135deg, #34c759 0%, #30d158 100%); color: white; padding: 16px 20px; border-radius: 12px; }
    .error-msg { background: linear-gradient(135deg, #ff3b30 0%, #ff453a 100%); color: white; padding: 16px 20px; border-radius: 12px; }
    .footer { text-align: center; color: #86868b; font-size: 13px; padding: 24px; }
    /* 12b. small right-aligned Clear Cache footer button: pinned to the far right edge
       of the content area (this build emits neither .st-key-* classes nor
       stBaseButton-secondary, so scope by the only kind="secondary" button on the page) */
    div[data-testid="column"]:has(button[kind="secondary"]) { align-items: flex-end !important; }
    div[data-testid="column"]:has(button[kind="secondary"]) div[data-testid="stButton"] {
        display: flex !important; justify-content: flex-end !important; width: 100% !important;
    }
    div[data-testid="column"]:has(button[kind="secondary"]) div[data-testid="stButton"] > button { width: auto !important; }
    /* NOTE: this Streamlit build does NOT emit .st-key-* class nor
       data-testid="stBaseButton-secondary"; the real button is
       <button kind="secondary" data-testid="baseButton-secondary">.
       Select on kind so Clear Cache (the only secondary button) stays subdued. */
    div.stButton > button[kind="secondary"] {
        height: 30px !important; line-height: 30px !important;
        padding: 0 12px !important; font-size: 13px !important;
        border-radius: 8px !important; width: auto !important;
        font-weight: 500 !important;
        background: transparent !important; color: var(--muted) !important;
        border: 1px solid var(--hairline) !important;
        box-shadow: none !important;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: transparent !important; color: var(--ink) !important;
        border-color: var(--muted) !important;
    }
    div.stButton > button[kind="secondary"] p { font-size: 13px !important; margin: 0 !important; }
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e5e5e7, transparent); margin: 24px 0; }
    .pdf-tag { display: inline-block; background: #e74c3c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
    .error-box { background: linear-gradient(135deg, #ff3b30 0%, #ff453a 100%); color: white; padding: 16px 20px; border-radius: 12px; margin: 10px 0; }
    .success-box { background: linear-gradient(135deg, #34c759 0%, #30d158 100%); color: white; padding: 16px 20px; border-radius: 12px; margin: 10px 0; }
    div[data-testid="stHorizontalBlock"] { align-items: center; }
    /* no page/column specific button size override: every button keeps the unified 46px geometry */
    /* 13. force native Streamlit default-red (#FF4B4B) controls onto brand blue:
       slider thumb + value label, uploader/multiselect delete tag */
    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background-color: var(--brand) !important; }
    div[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--brand) !important; }
    div[data-testid="stFileUploader"] [data-baseweb="tag"],
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] { background-color: var(--brand) !important; border-color: var(--brand) !important; color: #ffffff !important; }
    div[data-testid="stFileUploader"] [data-baseweb="tag"] button,
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] button { background: transparent !important; color: #ffffff !important; }
    /* radio selected dot: Streamlit uses .st-c2 (default primary #FF4B4B) for the checked circle */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child { background-color: var(--brand) !important; border-color: var(--brand) !important; }
    /* blanket catch: any Streamlit primary-red element -> brand blue (radio dots, checkbox checks, etc.) */
    .st-c2 { background-color: var(--brand) !important; }

    /* ============================================================
       14. unified page header: all 7 pages share one structure
           (single block: .main-title + .subtitle) -> identical alignment
       ============================================================ */
    .page-header { display: block !important; margin: 0 !important; padding: 0 !important; }
    .page-header .main-title {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        letter-spacing: -0.7px !important;
        line-height: 1.25 !important;
        margin: 0 0 8px 0 !important;
        padding: 0 !important;
    }
    .page-header .subtitle {
        color: #8A9199 !important;
        font-size: 15.5px !important;
        font-weight: 400 !important;
        letter-spacing: 0.1px !important;
        line-height: 1.5 !important;
        margin: 0 0 22px 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stMarkdown"]:has(.page-header),
    div[data-testid="stMarkdownContainer"]:has(.page-header) { margin: 0 !important; padding: 0 !important; }

    /* brand header: name exactly as large as page titles */
    .brand-header { display: flex !important; align-items: center !important; gap: 16px !important; }

    /* ============================================================
       15. unified form controls (text / number / select / multiselect / textarea)
       ============================================================ */
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stDateInput"] div[data-baseweb="input"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
        min-height: 46px !important;
        border-radius: 10px !important;
        border: 1px solid #DFE3E9 !important;
        background-color: #FCFDFE !important;
        box-shadow: inset 0 1px 2px rgba(16,24,40,0.03) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div { background-color: transparent !important; border: none !important; }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] div { font-size: 15px !important; color: var(--ink) !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div:first-child {
        /* value container: keep the native .5rem inset (was stripped by "padding: 0"),
           otherwise the selected label sticks to the left border; keep the vertical centring */
        display: flex !important; align-items: center !important; height: 100% !important;
        line-height: 1 !important; margin: 0 !important; padding: 0 .5rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div:last-child:not(:first-child) {
        /* chevron container: span the full 46px row and centre its content,
           so the arrow keeps its .5rem right inset and stays vertically centred */
        display: flex !important; align-items: center !important; align-self: stretch !important;
        height: 100% !important; margin: 0 !important; padding: 0 .5rem 0 0 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[data-testid="stMarkdownContainer"] p {
        /* inner text wrapper only: normalise the line box, no padding stripping on its parents */
        display: flex !important; align-items: center !important; height: 100% !important;
        line-height: 1 !important; margin: 0 !important; padding: 0 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
        /* the value container is shorter than the 46px box -> centre it instead of top-aligned stretch */
        align-items: center !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] button,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] button {
        height: 44px !important; min-height: 44px !important; border: none !important;
        background: transparent !important; color: var(--brand) !important; border-radius: 8px !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stDateInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child:focus-within,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child:focus-within {
        border-color: var(--brand) !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(28,111,182,0.14), 0 0 14px rgba(28,111,182,0.16) !important;
    }
    div[data-testid="stTextArea"] textarea {
        min-height: 96px !important;
        border-radius: 10px !important;
        border: 1px solid #DFE3E9 !important;
        background-color: #FCFDFE !important;
        font-size: 15px !important;
        color: var(--ink) !important;
        box-shadow: inset 0 1px 2px rgba(16,24,40,0.03) !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--brand) !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(28,111,182,0.14), 0 0 14px rgba(28,111,182,0.16) !important;
    }

    /* 15b. st.chat_input (CADO page): identical geometry + focus glow as text inputs */
    div[data-testid="stChatInput"] div[data-baseweb="textarea"] {
        min-height: 46px !important;
        border-radius: 10px !important;
        border: 1px solid #DFE3E9 !important;
        background-color: #FCFDFE !important;
        box-shadow: inset 0 1px 2px rgba(16,24,40,0.03) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }
    div[data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within {
        border-color: var(--brand) !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(28,111,182,0.14), 0 0 14px rgba(28,111,182,0.16) !important;
    }
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
        /* keep the field body and the submit button on the same horizontal centre line */
        display: flex !important; align-items: center !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 15px !important;
        color: var(--ink) !important;
        background: transparent !important;
        box-sizing: border-box !important;
        min-height: 46px !important;
        padding: 12px 48px 12px 10px !important;
        margin: 0 !important;
        line-height: 22px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder { color: #A7AEB8 !important; }
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {
        height: 38px !important; min-height: 38px !important; width: 38px !important;
        align-self: center !important; margin: auto 0 !important;
        border-radius: 9px !important; color: #ffffff !important;
        background: linear-gradient(135deg, #2E86D6 0%, #1C6FB6 55%, #155894 100%) !important;
        border: none !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover {
        background: linear-gradient(135deg, #3B93E0 0%, #2478C0 55%, #175F9F 100%) !important;
        box-shadow: 0 3px 12px rgba(28,111,182,0.26) !important;
    }

    /* ============================================================
       16. unified button system - every button family, every page
           geometry identical; primary = brand gradient, others = light ghost
       ============================================================ */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stLinkButton"] > a {
        height: 46px !important;
        min-height: 46px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.2px !important;
        width: 100% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background 0.18s ease, background-color 0.18s ease, border-color 0.18s ease,
                    box-shadow 0.22s ease, transform 0.18s ease !important;
    }
    div.stButton > button p,
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stPopover"] > button p,
    div[data-testid="stLinkButton"] > a p { font-size: 15px !important; font-weight: 600 !important; margin: 0 !important; }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2E86D6 0%, #1C6FB6 55%, #155894 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.08), 0 0 0 1px rgba(28,111,182,0.16) !important;
    }
    div.stButton > button[kind="primary"] p,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] p,
    div[data-testid="stDownloadButton"] > button[kind="primary"] p { color: #ffffff !important; }
    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3B93E0 0%, #2478C0 55%, #175F9F 100%) !important;
        box-shadow: 0 4px 16px rgba(28,111,182,0.28) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button[kind="secondary"],
    div[data-testid="stFormSubmitButton"] > button[kind="secondary"],
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stLinkButton"] > a {
        background: #ffffff !important;
        color: var(--brand) !important;
        border: 1px solid #D8E2EC !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04) !important;
    }
    div.stButton > button[kind="secondary"] p,
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stPopover"] > button p,
    div[data-testid="stLinkButton"] > a p { color: var(--brand) !important; font-size: 15px !important; font-weight: 600 !important; }
    div.stButton > button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stPopover"] > button:hover,
    div[data-testid="stLinkButton"] > a:hover {
        background: #F1F7FC !important;
        color: var(--brand) !important;
        border-color: var(--brand) !important;
        box-shadow: 0 3px 10px rgba(28,111,182,0.14) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "analyzed_data" not in st.session_state:
    st.session_state.analyzed_data = None
if "generated_report" not in st.session_state:
    st.session_state.generated_report = ""
if "report_style" not in st.session_state:
    st.session_state.report_style = "concise"
if "report_context" not in st.session_state:
    st.session_state.report_context = None
if "report_formula" not in st.session_state:
    st.session_state.report_formula = None
if "references" not in st.session_state:
    st.session_state.references = []
if "kr_props" not in st.session_state:
    st.session_state.kr_props = {}


@st.cache_resource(show_spinner="⚡ Loading Neural Engines...")
def load_agent():
    nlp = NLPProcessor(use_llm=True)
    agent = CentralOrchestrationAgent()
    return nlp, agent


try:
    nlp_processor, agent = load_agent()
except Exception as e:
    st.error(f"❌ Critical Error Loading Agent: {e}")
    st.stop()


import base64


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# ============================================================
# Top branding area: LOGO + DiCerAgent (left-aligned)
# ============================================================
logo_path = os.path.join(current_dir, "logo", "DiCerAgent_logo.png")

if os.path.exists(logo_path):
    bin_str = get_base64_of_bin_file(logo_path)
    st.markdown(
        f"""
    <div class="brand-header" style="display: flex; justify-content: flex-start; align-items: center; gap: 16px; padding: 2px 0 22px 0;">
        <img src="data:image/png;base64,{bin_str}" class="logo-img" alt="logo">
        <span class="logo-text">DiCerAgent</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="text-align: left;"><span class="logo-text">DiCerAgent</span></div>',
        unsafe_allow_html=True,
    )

tab_pred, tab_report, tab_chat, tab_plugin, tab_inverse, tab_settings, tab_about = st.tabs(
    [
        "ARMPP",
        "HRKE",
        "CADO",
        "MPC",
        "IDO",
        "Settings",
        "About",
    ]
)


def show_references():
    if st.session_state.references:
        all_refs = st.session_state.references
        top_refs = all_refs[:20]
        more_refs = all_refs[20:]

# --- 1. Top-20 core references (logic unchanged; ensure abstracts fill the card) ---
        for i, doc in enumerate(top_refs):
            authors = doc.get("authors", ["Unknown"])
            display_authors = (
                ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
                if isinstance(authors, list)
                else str(authors)
            )
            title = doc.get("title", "Untitled Paper")
            journal = doc.get("journal", "Unknown Journal")
            year = doc.get("year", "n.d.")
            doi = doc.get("doi", "").strip()
            if not doi and "10." in str(doc.get("source", "")):
                doi = str(doc.get("source", ""))

            with st.expander(f"[{i + 1}] {title}", expanded=False):
                col_info, col_btn = st.columns([3.5, 1])
                with col_info:
                    st.markdown(
                        f"""
                    <div style="font-size: 14px; color: #424245;">
                        <span style="font-weight: 600;">Authors:</span> {display_authors}
                        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
                        <span style="font-weight: 600;">Journal:</span> <span style="font-style: italic;">{journal}</span>, {year}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if doi:
                        clean_doi = doi.replace("doi:", "").strip()
                        link = (
                            f"https://doi.org/{clean_doi}"
                            if "10." in clean_doi and "http" not in clean_doi
                            else clean_doi
                        )
                        st.link_button("🔗 Read Paper", link, use_container_width=True)

                if doc.get("content"):
                    st.markdown(
                        f"""
                    <div style="margin-top: 0px; margin-bottom: 3px; padding: 12px; background-color: #f8f9fa; border-radius: 6px; font-size: 13px; color: #444; line-height: 1.5; text-align: justify;">
                        {doc.get("content")[:1200]}...
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

# --- 2. More related references (heavily compacted and beautified for Figure 4) ---
        if more_refs:
            st.markdown("---")
            with st.expander(
                f"📚 View {len(more_refs)} More Related References", expanded=False
            ):
                # Use a two- or three-column layout to reduce vertical whitespace
                cols = st.columns(2)
                for i, doc in enumerate(more_refs):
                    # Place alternately in the left/right column
                    col_idx = i % 2
                    idx = i + 21

                    title = doc.get("title", "Untitled")
                    journal = doc.get("journal", "Unknown")
                    year = doc.get("year", "n.d.")
                    doi = doc.get("doi", "").strip() or (
                        str(doc.get("source", ""))
                        if "10." in str(doc.get("source", ""))
                        else ""
                    )

                    # Build a minimal HTML entry
                    target_url = (
                        f"https://doi.org/{doi.replace('doi:', '').strip()}"
                        if doi
                        else "#"
                    )
                    link_html = (
                        f"<a href='{target_url}' target='_blank' style='color:#007aff; text-decoration:none; margin-left:8px;'>🔗 Read</a>"
                        if doi
                        else ""
                    )

                    with cols[col_idx]:
                        st.markdown(
                            f"""
                        <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #f0f0f2; border-radius: 8px; background-color: #ffffff;">
                            <div style="font-size: 13.5px; font-weight: 600; color: #1d1d1f; line-height: 1.3;">
                                [{idx}] {title}
                            </div>
                            <div style="font-size: 12px; color: #86868b; margin-top: 4px;">
                                <span style="text-transform: uppercase;">{journal}</span>, {year} {link_html}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
    else:
        st.caption("No references loaded.")


with tab_pred:
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Adaptive-Routing Multi-Property Predictor</p>
        <p class="subtitle">AI-driven multi-property prediction under heterogeneous data conditions</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query_input = st.text_input(
            "Enter Formula or Material ID",
            placeholder="e.g., CaWO4, mp-19426",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("Analyze", type="primary", use_container_width=True)

    # ---- secondary options group: grouped inside a light bordered card ----
    with st.container(border=True):
        use_cod = st.checkbox("Prioritize COD Database", value=False)

# ---- Collapsible manual method selector (collapsed by default = auto mode) ----
        with st.expander("⚙︎ Manual Method Selection (optional)", expanded=False):
            st.caption(
                "Tick one or more methods to override auto‑routing. "
                "Methods lacking required data (ML / literature) will be silently skipped."
            )
            ALL_METHODS = [
                "ML", "KNN", "ML-KNN",
                "LLM-ZS", "LLM-ML", "LLM-Full",
                "LLM-ML-Prop", "LLM-ML-Full",
            ]
            cols = st.columns(len(ALL_METHODS))
            selected_methods = []
            for i, m in enumerate(ALL_METHODS):
                with cols[i]:
                    if st.checkbox(m, key=f"cb_method_{m}"):
                        selected_methods.append(m)

    if run_btn:
        if not query_input:
            st.toast("⚠️ Please enter a formula or material ID first!", icon="ℹ️")
        else:
            final_query = query_input + (" --cod" if use_cod else "")

# No checkbox -> None (auto mode); checkboxes -> list
            user_methods = selected_methods if selected_methods else None

            with st.status("🤖 Agent Running...", expanded=True) as status:
                st.write("🧠 Parsing Semantic Intent...")
                cmd = nlp_processor.process_query(final_query)
                targets = agent.parse_user_input(cmd)
                target = targets[0]
                q_str = target.get("query")

                with st.spinner("Reloading external plugins..."):
                    agent.reload_external_plugins()

                st.write(f"📈 Running Physics Prediction for '{q_str}'...")
                ml_data = agent.analyze_single(
                    q_str, target.get("manual_features"), db_pref=target.get("db_pref"),
                    user_context=query_input,
                    enabled_methods=user_methods,
                )
                st.session_state.analyzed_data = ml_data

                st.write("📚 Running Hybrid Knowledge Retrieval...")
                if agent.scholar:
                    aliases = agent.extractor.brainstorm_keywords(q_str)
                    st.write(f"🔍 Keywords Identified: {', '.join(aliases)}")

                    csv_res = agent.csv_engine.search(q_str, aliases=aliases, limit=50)
                    rag_res = agent.rag_engine.search(q_str, aliases=aliases, k=50)

                    merged_docs = {}
                    for doc in csv_res:
                        key = agent.csv_engine._get_fingerprint(doc["title"])
                        if key:
                            merged_docs[key] = doc

                    for doc in rag_res:
                        full_path = doc.get("filename_raw", "")
                        fraw = os.path.basename(full_path)
                        meta = agent.csv_engine.match_metadata(
                            fraw, rag_title=doc.get("title")
                        )

                        if meta is not None:
                            doc["title"] = str(meta.get("Title", doc["title"])).strip()
                            doc["authors"] = str(
                                meta.get("Authors", doc.get("authors", ""))
                            )
                            if ";" in doc["authors"]:
                                doc["authors"] = [
                                    a.strip() for a in doc["authors"].split(";")
                                ]
                            doc["journal"] = str(
                                meta.get(
                                    "Source Title",
                                    meta.get("Journal Abbreviation", "Local Repo"),
                                )
                            )
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
                            doi_val = str(meta.get("DOI", meta.get("doi", ""))).strip()
                            if doi_val.lower() != "nan" and doi_val:
                                doc["doi"] = doi_val

                        title_key = agent.csv_engine._get_fingerprint(
                            doc.get("title", "")
                        )
                        if title_key in merged_docs:
                            existing = merged_docs[title_key]
                            existing["content"] = doc["content"]
                            existing["has_fulltext"] = True
                            existing["score"] = (
                                max(existing["score"], doc["score"]) * 1.3
                            )
                            if "doi" not in existing and "doi" in doc:
                                existing["doi"] = doc["doi"]
                            if "filename_raw" in doc:
                                existing["filename_raw"] = fraw
                        else:
                            doc["filename_raw"] = fraw
                            merged_docs[title_key] = doc

                    # === Channel 3: OpenAlex online retrieval ===
                    if agent.scholar:
                        try:
                            raw_papers = agent.scholar.search_literature(q_str, limit=10)
                            if raw_papers:
                                agent.scholar.enrich_papers_with_fulltext(raw_papers, max_fulltext=10)
                                scholar_docs = agent.scholar.process_papers_to_dicts(raw_papers)
                                for doc in scholar_docs:
                                    title_key = agent.csv_engine._get_fingerprint(doc.get("title", ""))
                                    if not title_key:
                                        continue
                                    # Re-score via ChemicalScorer for fair comparison with local engines
                                    doc["score"] = agent.csv_engine.scorer.evaluate(
                                        doc.get("title", ""), q_str, aliases=aliases
                                    )
                                    if title_key in merged_docs:
                                        existing = merged_docs[title_key]
                                        if "doi" not in existing and doc.get("doi"):
                                            existing["doi"] = doc["doi"]
                                    else:
                                        merged_docs[title_key] = doc
                                st.write(f"🌐 OpenAlex: {len(scholar_docs)} supplementary results")
                        except Exception as e:
                            st.write(f"⚠️ OpenAlex unavailable: {e}")

                    final_docs = sorted(
                        list(merged_docs.values()),
                        key=lambda x: x["score"],
                        reverse=True,
                    )
                    st.session_state.references = final_docs
                    st.write(f"📄 Found {len(final_docs)} relevant documents.")

                    st.write("🧪 Synthesizing Technical Report (Concise)...")
                    context = "=== ABSTRACTS & CONTENT ===\n"
                    llm_docs = final_docs[:20]
                    # Fallback: ensure top-20 contains 5-10 Local PDF full texts (if insufficient, backfill by score from below)
                    pdf_in_top = [
                        d for d in llm_docs
                        if d.get("type") == "Local PDF" or d.get("has_fulltext")
                    ]
                    if len(pdf_in_top) < 5:
                        pdf_rest = [
                            d for d in final_docs[20:]
                            if d.get("type") == "Local PDF" or d.get("has_fulltext")
                        ]
                        need = min(5 - len(pdf_in_top), len(pdf_rest))
                        if need > 0:
                            non_pdf = [d for d in llm_docs if d not in pdf_in_top]
                            non_pdf.sort(key=lambda x: x.get("score", 0))
                            for j in range(need):
                                llm_docs.remove(non_pdf[j])
                                llm_docs.append(pdf_rest[j])
                            llm_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                    for i, doc in enumerate(llm_docs):
                        context += f"Ref [{i + 1}] Title: {doc.get('title')}\nContent: {doc.get('content', '')[:1000]}\n\n"

                    # Save context for later regeneration
                    st.session_state.report_context = context
                    st.session_state.report_formula = q_str
                    st.session_state.report_style = "concise"

                    guide = agent.extractor.generate_report(q_str, context, style="concise")
                    st.session_state.generated_report = guide

                    st.write("📊 Extracting Literature Data Points...")
                    kr_props = agent.extractor.extract_property_values(
                        q_str, context, summary=guide
                    )
                    st.session_state.kr_props = kr_props
                else:
                    st.warning("Knowledge Engine not loaded.")

                status.update(
                    label="✨ Analysis Complete!", state="complete", expanded=False
                )

    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        kr_data = st.session_state.kr_props

        def format_kr_str(key):
            items = kr_data.get(key, [])
            if not items:
                return "No Knowledge Retrieval Data"
            lines = []
            for item in items:
                val = item.get("value", "?")
                src = item.get("source", "")
                cond = item.get("condition", "")
                if cond and cond != "Pure":
                    val += f" ({cond})"
                lines.append(f"• {val} {src}")
            return "\n".join(lines)

        col1, col2 = st.columns([1, 1.2])

        def get_display_name(key):
            from plugins.base import get_property_info

            base_key = key.replace("_mean", "").replace("_dev", "")
            info = get_property_info(base_key)
            if key.endswith("_dev"):
                return None
            if info.unit:
                return f"{info.symbol} ({info.unit})"
            return info.symbol if info.symbol else base_key

        def fmt_val(val, dev, key):
            from plugins.base import get_property_info

            if val is None:
                return "", ""
            try:
                val = float(val)
            except:
                return str(val), ""
            try:
                dev = float(dev) if dev is not None else 0
                val_str = f"{val:.2f}"
                if dev != 0:
                    dev_str = f"± {dev:.2f}"
                else:
                    dev_str = ""
                return val_str, dev_str
            except:
                return f"{val:.2f}", ""

        col1, col2 = st.columns([1, 1.2])

        with col1:
            # --- Keep all logic processing unchanged ---
            exclude_keys = {
                "Input_Query",
                "Material_ID",
                "Structure_Type",
                "Formula",
                "Error",
                "_structure_obj",
            }
            pred_keys = [
                k for k in data.keys() if k not in exclude_keys and k.endswith("_mean")
            ]

            seen = set()
            normalized_keys = []
            for k in pred_keys:
                norm = k.replace("_mean", "").lower()
                if norm not in seen:
                    seen.add(norm)
                    normalized_keys.append((k, norm))

            order = {"er": "0", "qxf": "1", "tcf": "2"}

            def get_sort_order(item):
                k, norm = item
                return order.get(norm, "z" + norm)

            normalized_keys = sorted(normalized_keys, key=get_sort_order)

            # Fix the three dielectric property columns (er/qxf/tcf); do not vary with the *_mean keys of prediction results
            # (to avoid extra keys such as tm_mean producing extra columns in the references table)
            lit_keys = []
            from plugins.base import get_property_info
            for norm in ["er", "qxf", "tcf"]:
                info = get_property_info(norm)
                if info.unit:
                    lit_keys.append((norm, f"{info.symbol} ({info.unit})"))
                else:
                    lit_keys.append((norm, info.symbol))

            st_type = data.get("Structure_Type", "Unknown")
            mat_id = data.get("Material_ID", "N/A")

            if normalized_keys:
                # Group enhanced results by target
                # UI's norm (qxf) and the enhanced result's target (qf) have a mapping difference
                TARGET_ALIASES = {"qxf": "qf"}
                enhanced = data.get("enhanced_results") or []
                enhanced_by_prop = {}
                for r in enhanced:
                    t = r.get("target", "").lower()
                    enhanced_by_prop.setdefault(t, []).append(r)

                kpi_slots = {}
                table_html = """<table class="pred-table"><tr><th>Property</th><th>Predicted Value</th></tr>"""
                for key, norm in normalized_keys:
                    dev_key = key.replace("_mean", "_dev")
                    val = data.get(key)
                    dev = data.get(dev_key)
                    display_name = get_display_name(key)
                    val_str, dev_str = fmt_val(val, dev, key)
                    if val_str:
                        kpi_slots[norm] = (display_name, val_str, dev_str)

                    # Collect predicted values of all methods for this property (ML + enhanced)
                    all_vals = []

                    if val_str:
                        ml_str = f"{val_str} (ML)"
                        if dev_str:
                            ml_str += f" <span class='dev'>{dev_str}</span>"
                        all_vals.append(ml_str)  # always use the full ML (incl. dev); if an enhanced result already has ML, it is skipped below

                    # Enhanced results: skip already-processed ML methods (add only LLM-enhanced methods)
                    for check_t in {norm, TARGET_ALIASES.get(norm, "")}:
                        if not check_t:
                            continue
                        for er in enhanced_by_prop.get(check_t, []):
                            if er.get("method") == "ML":
                                continue  # ML already processed above
                            ev = er.get("value")
                            em = er.get("method", "?").replace("-", "_")
                            if ev is not None:
                                all_vals.append(f"{ev:.2f} ({em})")
                    # Flat-injected _mean keys (e.g. er_LLM_ML_Full_mean)
                    alt_prefixes = {norm}
                    for _alias, _mapped in TARGET_ALIASES.items():
                        if _mapped == norm:
                            alt_prefixes.add(_alias)
                    for flat_key in sorted(data.keys()):
                        if not flat_key.endswith("_mean") or flat_key == key:
                            continue
                        for pfx in alt_prefixes:
                            if flat_key.startswith(f"{pfx}_"):
                                fv = data[flat_key]
                                if fv is not None:
                                    fm = flat_key[len(pfx)+1:-5]
                                    all_vals.append(f"{fv:.2f} ({fm})")
                                break

                    seen_v = set()
                    dedup = []
                    for v in all_vals:
                        v_lower = v.lower()
                        if v_lower not in seen_v:
                            seen_v.add(v_lower)
                            dedup.append(v)

                    cell = ", ".join(dedup)
                    table_html += f"<tr><td><b>{display_name}</b></td><td class='val'>{cell}</td></tr>"
                table_html += "</table>"
            else:
                table_html = "<p>No prediction results available.</p>"
                kpi_slots = {}

            # --- KPI big cards: fixed display of er/qxf/tcf main predictions ---
            kpi_html = ""
            if kpi_slots:
                kpi_html = '<div class="kpi-row">'
                for norm in ["er", "qxf", "tcf"]:
                    slot = kpi_slots.get(norm)
                    if slot is None:
                        continue
                    label, v, d = slot
                    kpi_html += (
                        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                        f'<div class="kpi-value">{v}<span class="kpi-dev">{d}</span></div></div>'
                    )
                kpi_html += "</div>"

            # --- Key rendering change: wrap all content with unified-card ---
            st.markdown(
                f"""
            <div class="pred-card">
                <h3 style="margin-top:0; color:#1d1d1f; font-size: 20px; margin-bottom: 16px;">📈 Prediction Results</h3>
                <div style="margin-bottom: 15px; color: #86868b; font-size: 14px;">
                    <strong>Structure:</strong> {st_type} | <strong>ID:</strong> {mat_id}
                </div>
                {kpi_html}
                {table_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Right side uses border mode alignment; CSS ensures its border/shadow matches the left side
            with st.container(border=True):
                st.markdown(
                    '<h3 style="margin-top:0; color:#1d1d1f; font-size: 20px; margin-bottom: 16px;">🧊 Crystal Structure</h3>',
                    unsafe_allow_html=True,
                )
                struct_obj = data.get("_structure_obj")
                if struct_obj:
                    # Render the model directly
                    viz_component.render_structure_card(
                        struct_obj, key_suffix="main_pred"
                    )
                else:
                    st.info("No structure visualization available.")

        if lit_keys:
            # --- 1. Build the complete HTML string ---
            # Merge all content (container opening + title + table + container closing)
            lit_html = '<div class="pred-card" style="margin-top: 20px;">'
            lit_html += '<h3 style="margin-top:0; color:#1d1d1f; font-size: 20px;">📊 Literature Data</h3>'
            lit_html += '<table class="pred-table"><tr>'

            # Table header
            for lit_key, lit_name in lit_keys:
                lit_html += f"<th>{lit_name}</th>"
            lit_html += "</tr><tr>"

            # Content rows
            for lit_key, lit_name in lit_keys:
                items = kr_data.get(lit_key, [])
                if items:
                    vals = []
                    for idx, item in enumerate(items, 1):
                        val = item.get("value", "?")
                        src = item.get("source", "").strip("[] ")
                        cond = item.get("condition", "")
                        display_val = (
                            f"{val} ({cond})" if cond and cond != "Pure" else val
                        )
                        vals.append(f"({idx}) {display_val} [{src}]")
                    lit_html += f'<td style="vertical-align: top; padding: 12px; line-height: 1.6;">{"<br>".join(vals)}</td>'
                else:
                    lit_html += '<td style="color:#86868b; vertical-align: top; padding: 12px;">No data</td>'

            lit_html += "</tr></table>"
            lit_html += "</div>"  # close pred-card

            # --- 2. Render once ---
            st.markdown(lit_html, unsafe_allow_html=True)

        # === Export functionality ===
        if st.session_state.analyzed_data:
            # Generate the prediction report
            export_parts = []
            export_parts.append("=" * 60)
            export_parts.append("DiCerAgent ARMPP (Forward Prediction) Report")
            export_parts.append("=" * 60)

            formula = st.session_state.analyzed_data.get("Formula", "Unknown")
            structure_type = st.session_state.analyzed_data.get(
                "Structure_Type", "Unknown"
            )
            material_id = st.session_state.analyzed_data.get("Material_ID", "N/A")

            export_parts.append(f"\nMaterial: {formula}")
            export_parts.append(f"Structure Type: {structure_type}")
            export_parts.append(f"Materials Project ID: {material_id}")
            export_parts.append("\n" + "-" * 60)
            export_parts.append("\n### Predicted Properties")

            for k, v in st.session_state.analyzed_data.items():
                if k.endswith("_mean") and v is not None:
                    prop = k.replace("_mean", "")
                    from plugins.base import get_property_info

                    info = get_property_info(prop)
                    symbol = info.symbol if info.symbol else prop
                    unit = info.unit if info.unit else ""
                    dev = st.session_state.analyzed_data.get(
                        k.replace("_mean", "_dev"), 0
                    )
                    if unit:
                        export_parts.append(f"{symbol}: {v:.2f} ± {dev:.2f} {unit}")
                    else:
                        export_parts.append(f"{symbol}: {v:.2f} ± {dev:.2f}")

            # References
            refs = st.session_state.references
            if refs:
                export_parts.append("\n" + "-" * 60)
                export_parts.append(f"\n### References ({len(refs)} papers)")
                for i, ref in enumerate(refs, 1):
                    title = ref.get("title", "Unknown")
                    authors = ref.get("authors", [])
                    year = ref.get("year", "n.d.")
                    journal = ref.get("journal", "Unknown")
                    url = ref.get("url", "")
                    export_parts.append(f"\n[{i}] {title}")
                    export_parts.append(f"    Year: {year} | Journal: {journal}")
                    if authors:
                        export_parts.append(
                            f"    Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}"
                        )
                    if url:
                        export_parts.append(f"    URL: {url}")

            export_text = "\n".join(export_parts)

            # HTML version
            html_parts = [f"<h1>DiCerAgent Prediction Report</h1>"]
            html_parts.append(f"<p><b>Material:</b> {formula}</p>")
            html_parts.append(f"<p><b>Structure:</b> {structure_type}</p>")
            html_parts.append(f"<p><b>MP ID:</b> {material_id}</p>")
            html_parts.append("<h2>Predicted Properties</h2>")

            for k, v in st.session_state.analyzed_data.items():
                if k.endswith("_mean") and v is not None:
                    prop = k.replace("_mean", "")
                    from plugins.base import get_property_info

                    info = get_property_info(prop)
                    symbol = info.symbol if info.symbol else prop
                    unit = info.unit if info.unit else ""
                    dev = st.session_state.analyzed_data.get(
                        k.replace("_mean", "_dev"), 0
                    )
                    html_parts.append(
                        f"<p><b>{symbol}</b>: {v:.2f} ± {dev:.2f} {unit}</p>"
                    )

            if refs:
                html_parts.append("<h2>References</h2>")
                for i, ref in enumerate(refs, 1):
                    title = ref.get("title", "Unknown")
                    html_parts.append(f"<p>[{i}] {title}</p>")

            html_content = (
                "<html><head><meta charset='utf-8'></head><body>"
                + "".join(html_parts)
                + "</body></html>"
            )

            # Export button
            st.markdown(
                """
                <style>
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
            with st.popover("📥 Export"):
                pdf_data = create_pdf_export(export_parts)
                st.download_button(
                    "📄 TXT",
                    export_text,
                    f"{formula}_prediction.txt",
                    "text/plain",
                    use_container_width=True,
                )
                st.download_button(
                    "📝 HTML",
                    html_content,
                    f"{formula}_prediction.html",
                    "text/html",
                    use_container_width=True,
                )
                st.download_button(
                    "📑 PDF",
                    pdf_data,
                    f"{formula}_prediction.pdf",
                    "application/pdf",
                    use_container_width=True,
                )

        st.markdown(
            '<p class="section-header">📖 References</p>', unsafe_allow_html=True
        )
        show_references()

with tab_report:
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Hybrid-Retrieval Knowledge Extractor</p>
        <p class="subtitle">Extracts chemistry-relevant knowledge from literature and databases to synthesize technical reports</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Style selection + regenerate
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_style = st.radio(
            "Report Style",
            ["concise", "detailed"],
            format_func=lambda s: "Concise" if s == "concise" else "Detailed",
            horizontal=True,
            index=0 if st.session_state.report_style == "concise" else 1,
            key="report_style_radio",
        )
    with col2:
        if st.button("🔄 Regenerate", use_container_width=True, type="primary",
                     disabled=not st.session_state.report_context):
            with st.spinner(f"Generating {new_style} report..."):
                st.session_state.report_style = new_style
                st.session_state.generated_report = agent.extractor.generate_report(
                    st.session_state.report_formula,
                    st.session_state.report_context,
                    style=new_style,
                )
            st.rerun()
    with col3:
        if st.session_state.report_style:
            badge = "🟢 Concise" if st.session_state.report_style == "concise" else "🔵 Detailed"
            st.caption(f"Current: {badge}")

    display_text = st.session_state.generated_report

    if display_text:
        with st.container(border=True):
            st.markdown(display_text)
    else:
        st.info("No report generated yet. Run analysis in the ARMPP tab first.")

    # === Export functionality ===
    if st.session_state.generated_report:
        formula = (
            st.session_state.analyzed_data.get("Formula", "Unknown")
            if st.session_state.analyzed_data
            else "Unknown"
        )

        # Generate report text
        export_parts = []
        export_parts.append("=" * 60)
        export_parts.append("DiCerAgent Technical Report")
        export_parts.append("=" * 60)
        export_parts.append(f"\nMaterial: {formula}")

        if st.session_state.generated_report:
            export_parts.append("\n" + "-" * 60)
            export_parts.append("\n### Technical Report")
            export_parts.append(st.session_state.generated_report)

        # References
        refs = st.session_state.references
        if refs:
            export_parts.append("\n" + "-" * 60)
            export_parts.append(f"\n### References ({len(refs)} papers)")
            for i, ref in enumerate(refs, 1):
                title = ref.get("title", "Unknown")
                authors = ref.get("authors", [])
                year = ref.get("year", "n.d.")
                journal = ref.get("journal", "Unknown")
                url = ref.get("url", "")
                export_parts.append(f"\n[{i}] {title}")
                export_parts.append(f"    Year: {year} | Journal: {journal}")
                if authors:
                    export_parts.append(
                        f"    Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}"
                    )
                if url:
                    export_parts.append(f"    URL: {url}")

        export_text = "\n".join(export_parts)

        # HTML version
        html_parts = [f"<h1>DiCerAgent Technical Report</h1>"]
        html_parts.append(f"<p><b>Material:</b> {formula}</p>")
        html_parts.append("<h2>Report</h2>")
        html_parts.append(f"<p>{st.session_state.generated_report}</p>")

        if refs:
            html_parts.append("<h2>References</h2>")
            for i, ref in enumerate(refs, 1):
                title = ref.get("title", "Unknown")
                html_parts.append(f"<p>[{i}] {title}</p>")

        html_content = (
            "<html><head><meta charset='utf-8'></head><body>"
            + "".join(html_parts)
            + "</body></html>"
        )

        # Generate PDF
        pdf_data = create_pdf_export(export_parts)

        # Export button
        st.markdown(
            """
            <style>
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
        with st.popover("📥 Export"):
            pdf_data = create_pdf_export(export_parts)
            st.download_button(
                "📄 TXT",
                export_text,
                f"{formula}_report.txt",
                "text/plain",
                use_container_width=True,
            )
            st.download_button(
                "📝 HTML",
                html_content,
                f"{formula}_report.html",
                "text/html",
                use_container_width=True,
            )
            st.download_button(
                "📑 PDF",
                pdf_data,
                f"{formula}_report.pdf",
                "application/pdf",
                use_container_width=True,
            )

    st.markdown('<p class="section-header">📖 References</p>', unsafe_allow_html=True)
    show_references()

# ==================== CADO Chat Tab ====================
from chat_module import render_chat_tab

with tab_chat:
    render_chat_tab(
        current_data=st.session_state.analyzed_data,
        current_refs=st.session_state.references,
        agent=agent,
        csv_engine=getattr(agent, "csv_engine", None),
        rag_engine=getattr(agent, "rag_engine", None),
        extractor=getattr(agent, "extractor", None),
        scholar=getattr(agent, "scholar", None),
    )

with tab_plugin:
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Model-to-Plugin Converter</p>
        <p class="subtitle">Converts heterogeneous external ML models into interoperable platform-ready plugins</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
        ["📁 Folder Mode", "📤 Upload Mode", "📝 Template", "📦 Installed Plugins"]
    )

    def extract_python_code(response: str) -> str:
        return _get_app_converter().extract_python_code(response)

    def extract_text_from_pdf(pdf_file):
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
        return _get_app_converter().call_deepseek(prompt, context, retry_count)

    def call_deepseek_for_analysis(prompt, retry_count=2):
        return _get_app_converter().call_deepseek_for_analysis(prompt, retry_count)

    def analyze_model_config(paper_context, user_code, training_data_info):
        return _get_app_converter().analyze_model_config(
            paper_context, user_code, training_data_info
        )

    def detect_file_type(filename, content_preview=""):
        return _get_app_converter().detect_file_type(filename, content_preview)

    def scan_folder(folder_path):
        return _get_app_converter().scan_folder(folder_path)

    with sub_tab1:
        st.markdown("### Step 1: Select Model Folder")
        st.markdown(f"Place your model files in: `external_models/<your_model_name>/`")

        os.makedirs(EXTERNAL_MODELS_DIR, exist_ok=True)
        folders = [
            d
            for d in os.listdir(EXTERNAL_MODELS_DIR)
            if os.path.isdir(os.path.join(EXTERNAL_MODELS_DIR, d))
            and not d.startswith(".")
        ]

        if not folders:
            st.info("No model folders found.")
            st.markdown("""
            **Expected folder structure:**
            ```
            external_models/
            └── oxide_MeltTemp/
                ├── model.py
                ├── train_data.csv
                └── paper.pdf
            ```
            """)
            selected_folder = None
        else:
            col_select, col_create = st.columns([2, 1])
            with col_select:
                selected_folder = st.selectbox("Select model folder:", options=folders)
            with col_create:
                new_folder_name = st.text_input(
                    "New folder:", key=f"new_folder_{selected_folder}"
                )
                if st.button("Create", key=f"create_folder_{selected_folder}"):
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
                st.warning("No files found.")
            else:
                col_type1, col_type2, col_type3 = st.columns(3)

                code_files = [f for f in files_info if f["type"] == "code"]
                data_files = [f for f in files_info if f["type"] == "data"]
                paper_files = [f for f in files_info if f["type"] == "paper"]
                supporting_files = [f for f in files_info if f["type"] == "supporting"]

                with col_type1:
                    st.markdown("**Code Files**")
                    if code_files:
                        code_options = {
                            f"{f['name']} ({f['type_label']})": f for f in code_files
                        }
                        selected_code_names = st.multiselect(
                            "Select code files:",
                            options=list(code_options.keys()),
                            default=list(code_options.keys())[:1],
                            key=f"code_multiselect_{selected_folder}",
                        )
                        selected_codes = (
                            [code_options[name] for name in selected_code_names]
                            if selected_code_names
                            else []
                        )
                    else:
                        st.info("No code files")
                        selected_codes = []

                    st.markdown("**Data Files**")
                    if data_files:
                        data_options = {
                            f"{f['name']} ({f['size']} bytes)": f for f in data_files
                        }
                        selected_data_names = st.multiselect(
                            "Select data files:",
                            options=list(data_options.keys()),
                            default=list(data_options.keys()),
                            key=f"data_multiselect_{selected_folder}",
                        )
                        selected_datas = (
                            [data_options[name] for name in selected_data_names]
                            if selected_data_names
                            else []
                        )
                    else:
                        st.info("No data files")
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
                            key=f"const_multiselect_{selected_folder}",
                        )
                        selected_consts = (
                            [const_options[name] for name in selected_const_names]
                            if selected_const_names
                            else []
                        )
                    else:
                        selected_consts = []

                with col_type2:
                    all_pdf_files = paper_files + supporting_files
                    st.markdown("**PDF Documents**")
                    if all_pdf_files:
                        pdf_options = {
                            f"{f['name']} ({f['type_label']})": f for f in all_pdf_files
                        }
                        selected_pdf_names = st.multiselect(
                            "Select PDFs:",
                            options=list(pdf_options.keys()),
                            default=list(pdf_options.keys()),
                            key=f"pdf_multiselect_{selected_folder}",
                        )
                        selected_pdfs = (
                            [pdf_options[name] for name in selected_pdf_names]
                            if selected_pdf_names
                            else []
                        )
                    else:
                        st.info("No PDF files")
                        selected_pdfs = []

                    selected_paper = st.selectbox(
                        "Primary paper:",
                        options=[f["name"] for f in selected_pdfs]
                        if selected_pdfs
                        else ["None"],
                        key=f"paper_select_{selected_folder}",
                    )

                with col_type3:
                    st.markdown("**Property Definition**")
                    prop_key = st.text_input(
                        "Property Key",
                        value=selected_folder.split("_")[-1]
                        if "_" in selected_folder
                        else "property",
                        key=f"prop_key_{selected_folder}",
                    )
                    prop_symbol = st.text_input(
                        "Display Symbol",
                        value=selected_folder.split("_")[-1].upper()
                        if "_" in selected_folder
                        else "P",
                        key=f"prop_symbol_{selected_folder}",
                    )
                    prop_unit = st.text_input(
                        "Unit", value="unit", key=f"prop_unit_{selected_folder}"
                    )

                    st.markdown("**Training Settings**")
                    n_ensembles = st.number_input(
                        "Ensemble iterations:",
                        min_value=1,
                        max_value=2000,
                        value=100,
                        step=100,
                        key=f"n_ensembles_{selected_folder}",
                    )

                    model_notes = st.text_area(
                        "Notes:",
                        placeholder="e.g., Use ANN for stage2...",
                        key=f"model_notes_{selected_folder}",
                        height=80,
                    )

                confirm_btn = st.button(
                    "⚡ Confirm & Analyze with AI",
                    type="primary",
                    use_container_width=True,
                )

                if confirm_btn:
                    if not selected_codes and not selected_pdfs:
                        st.error("Select at least code or PDF file")
                    else:
                        with st.spinner("Processing..."):
                            user_code = ""
                            paper_context = ""
                            training_data_info = ""

                            if selected_codes:
                                for i, code_file in enumerate(selected_codes):
                                    with open(
                                        code_file["path"],
                                        "r",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as f:
                                        user_code += (
                                            f"\n\n=== FILE {i + 1}: {code_file['name']} ===\n"
                                            + f.read()
                                        )
                                st.success(f"Loaded {len(selected_codes)} code file(s)")

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

                            if selected_datas:
                                const_file_names = {cf["name"] for cf in selected_consts}
                                dfs = []
                                all_dfs = []
                                for data_file in selected_datas:
                                    if data_file["name"] in const_file_names:
                                        continue  # constant files do not participate in training-data merging
                                    try:
                                        if data_file["name"].endswith(".csv"):
                                            df = pd.read_csv(data_file["path"])
                                        elif data_file["name"].endswith(
                                            (".xlsx", ".xls")
                                        ):
                                            df = pd.read_excel(data_file["path"])
                                        else:
                                            continue
                                        dfs.append((data_file["name"], df))
                                        all_dfs.append(df)
                                    except:
                                        pass
                                if dfs:
                                    for name, df in dfs:
                                        training_data_info += (
                                            f"\n\n=== Data: {name} ===\n"
                                        )
                                        training_data_info += (
                                            f"Columns: {list(df.columns)}\n"
                                        )
                                        training_data_info += f"Rows: {len(df)}\n"
                                combined_df = (
                                    pd.concat(all_dfs, ignore_index=True)
                                    if all_dfs
                                    else None
                                )
                            else:
                                combined_df = None
                            st.session_state.folder_analysis = {
                                "user_code": user_code,
                                "paper_context": paper_context,
                                "training_data_info": training_data_info,
                                "constant_files": selected_consts,
                                "prop_key": prop_key,
                                "prop_symbol": prop_symbol,
                                "prop_unit": prop_unit,
                                "n_ensembles": n_ensembles,
                                "model_notes": model_notes,
                                "data_df": combined_df,
                                "folder_path": folder_path,
                                "model_files_detection": detect_model_files(folder_path),
                            }

    if "folder_analysis" in st.session_state:
        st.markdown("---")
        st.markdown("### Step 3: Model Configuration & Conversion")

        analysis = st.session_state.folder_analysis

        ai_config = analyze_model_config(
            analysis["paper_context"],
            analysis["user_code"],
            analysis["training_data_info"],
        )

        st.markdown("#### 📋 Model Configuration (Editable)")

        if ai_config:
            c1, c2, c3 = st.columns(3)
            with c1:
                prop_key = st.text_input(
                    "Property Key",
                    value=ai_config.get("key", analysis.get("prop_key", "property")),
                    key="edit_prop_key",
                )
            with c2:
                prop_symbol = st.text_input(
                    "Symbol",
                    value=ai_config.get("symbol", analysis.get("prop_symbol", "P")),
                    key="edit_prop_symbol",
                )
            with c3:
                prop_unit = st.text_input(
                    "Unit",
                    value=ai_config.get("unit", analysis.get("prop_unit", "")),
                    key="edit_prop_unit",
                )

            c4, c5 = st.columns(2)
            with c4:
                scope = st.text_input(
                    "Scope/Applicability",
                    value=ai_config.get("scope", ""),
                    key="edit_scope",
                )
            with c5:
                is_two_stage = st.checkbox(
                    "Two-Stage Model",
                    value=ai_config.get("two_stage", False),
                    key="edit_two_stage",
                )

            st.caption("👆 Please modify as needed")

        st.markdown("---")
        st.markdown("### Step 4: Feature Selection & Conversion")

        analysis = st.session_state.folder_analysis
        data_df = analysis.get("data_df")

        feature_folder = os.path.join(current_dir, "feature")
        feature_info = ""
        if os.path.exists(feature_folder):
            feature_info += (
                "## Available Feature Constants Files (in feature/ folder)\n"
            )
            for f in os.listdir(feature_folder):
                fpath = os.path.join(feature_folder, f)
                if f.endswith(".csv"):
                    try:
                        df = pd.read_csv(fpath, nrows=5)
                        feature_info += f"- **{f}**: columns = {list(df.columns)}\n"
                    except:
                        feature_info += f"- **{f}**\n"
                elif f.endswith(".ini"):
                    feature_info += f"- **{f}** (config file)\n"
                elif f.endswith(".xlsx"):
                    feature_info += f"- **{f}** (Excel file)\n"
            feature_info += "\nIMPORTANT: When calculating features, check these files in the feature/ folder for any constants or lookup tables you may need!\n"

        CONVERSION_PROMPT = """You are a professional materials science ML engineer. Convert the external model into DiCerAgent plugin format.

## Requirements
1. Output ONLY the Python code - no explanations
2. MUST include get_required_features() method - THIS IS REQUIRED BY THE PLUGIN SYSTEM
3. The training system saves ALL models in ONE file (two_stage_model.pkl or model.pkl)
4. Your predict() should load models from: model.pkl or two_stage_model.pkl
5. In predict(): use all loaded models for ensemble prediction (mean + std)
6. Return format: {{ "property_mean": value, "property_dev": std }}
7. IMPORTANT: read MP_API_KEY from settings.ini using configparser with utf-8 encoding
8. Include get_required_features() method that returns the feature list
9. End with register_plugin() call

## Training Data Available
{training_data_info}

{feature_info}

## Model Configuration
- Property Key: {prop_key}
- Property Symbol: {prop_symbol}
- Property Unit: {prop_unit}
- Ensemble Iterations: {n_ensembles}

## Special Requirements from User
{model_notes}

## Original Model Code (may contain multiple files for multi-step model)
{user_code}

## Paper/Supporting Information
{paper_context}

## IMPORTANT: Feature Column Detection
Analyze the training data columns to identify:
- stage1_feature_cols: features used in step 1 (composition-based, like H, Li, Ca, O)
- stage2_add_feature_cols: additional features in step 2 (like na, d, fepa)
- target column: the final property to predict

## How to Calculate Features in calculate()
Based on training data columns, implement:
- Element columns: Composition(structure.composition.reduced_formula).get_el_amt_dict()[elem] (atomic numbers; see constraint j - MUST use the reduced formula, NEVER the cell's absolute counts)
- na: Composition(structure.composition.reduced_formula).num_atoms (one formula unit - see constraint j)
- d: structure.density
- fepa: Fetch from Materials Project API using mp-api

"""

    if "folder_analysis" in st.session_state:
        analysis = st.session_state.folder_analysis
        data_df = analysis.get("data_df")

        if data_df is not None and not data_df.empty:
            all_cols = list(data_df.columns)
            non_target_cols = [
                c
                for c in all_cols
                if c.lower()
                not in ["target", "property", "y", "label", "output", "source_file"]
            ]

            if "step4_target" not in st.session_state:
                _suggested = analysis.get("target_col")
                if _suggested and _suggested in non_target_cols:
                    st.session_state.step4_target = _suggested
                else:
                    _numeric_cands = [
                        c for c in non_target_cols
                        if pd.api.types.is_numeric_dtype(data_df[c])
                    ]
                    st.session_state.step4_target = (
                        _numeric_cands[0] if _numeric_cands
                        else (non_target_cols[0] if non_target_cols else None)
                    )
            if "step4_n_iters" not in st.session_state:
                st.session_state.step4_n_iters = 100
            if "step4_stage1" not in st.session_state:
                st.session_state.step4_stage1 = (
                    [
                        c for c in non_target_cols
                        if c != st.session_state.step4_target
                        and pd.api.types.is_numeric_dtype(data_df[c])
                    ][:5]
                    if non_target_cols else []
                )
            if "step4_stage2" not in st.session_state:
                st.session_state.step4_stage2 = []

            _numeric_cols_in_data = [
                c for c in non_target_cols
                if c != st.session_state.step4_target
                and pd.api.types.is_numeric_dtype(data_df[c])
            ]
            _text_cols_in_data = [
                c for c in non_target_cols
                if c != st.session_state.step4_target
                and not pd.api.types.is_numeric_dtype(data_df[c])
            ]
            if not _numeric_cols_in_data and _text_cols_in_data:
                if analysis.get("constant_files"):
                    st.info(
                        "Training data has no precomputed numeric feature columns "
                        "(only composition + target). Features will be auto-computed "
                        "from the chemical formula using the constant tables and the "
                        "original code's calculation logic when you train (Branch B). "
                        "You can leave Stage 1 / Stage 2 empty."
                    )
                else:
                    st.warning(
                        "Training data has no precomputed numeric feature columns. "
                        "Please select the constant files (element property tables) in "
                        "the previous step; otherwise features cannot be auto-computed "
                        "from the formula at training time."
                    )

            st.markdown("#### Select Target & Features")
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                target_col = st.selectbox(
                    "Target Column",
                    options=non_target_cols,
                    index=non_target_cols.index(st.session_state.step4_target)
                    if st.session_state.step4_target in non_target_cols
                    else 0,
                    key="train_target",
                )
            with col_feat2:
                n_iterations = st.number_input(
                    "Bootstrap Iterations",
                    min_value=10,
                    max_value=500,
                    value=st.session_state.step4_n_iters,
                    step=10,
                    key="n_iters",
                )

            c1, c2 = st.columns([9, 1])
            with c1:
                stage1_cols = st.multiselect(
                    "Stage 1 Features",
                    options=non_target_cols,
                    default=st.session_state.step4_stage1,
                    key="stage1_feats",
                )
            with c2:
                if st.button("All", key="sa_s1"):
                    st.session_state.step4_stage1 = non_target_cols
                    st.rerun()

            c3, c4 = st.columns([9, 1])
            with c3:
                stage2_cols = st.multiselect(
                    "Stage 2 Additional Features",
                    options=non_target_cols,
                    default=st.session_state.step4_stage2,
                    key="stage2_feats",
                )
            with c4:
                if st.button("All", key="sa_s2"):
                    st.session_state.step4_stage2 = non_target_cols
                    st.rerun()

            is_two_stage = st.checkbox(
                "Two-Stage Model", value=len(stage2_cols) > 0, key="train_two_stage"
            )

            st.session_state.step4_target = target_col
            st.session_state.step4_n_iters = n_iterations
            st.session_state.step4_stage1 = stage1_cols
            st.session_state.step4_stage2 = stage2_cols

            st.markdown("---")
            st.markdown("#### Convert to Plugin Code")
            convert_btn = st.button(
                "🔄 Convert to Plugin Code", type="primary", use_container_width=True
            )

            if convert_btn:
                st.session_state.folder_analysis["prop_key"] = prop_key
                st.session_state.folder_analysis["prop_symbol"] = prop_symbol
                st.session_state.folder_analysis["prop_unit"] = prop_unit
                st.session_state.folder_analysis["scope"] = scope
                st.session_state.folder_analysis["is_two_stage"] = is_two_stage
                st.session_state.folder_analysis["target_col"] = (
                    st.session_state.step4_target
                )
                st.session_state.folder_analysis["stage1_cols"] = (
                    st.session_state.step4_stage1
                )
                st.session_state.folder_analysis["stage2_cols"] = (
                    st.session_state.step4_stage2
                )
                st.session_state.folder_analysis["n_iterations"] = (
                    st.session_state.step4_n_iters
                )

                feature_folder = os.path.join(current_dir, "feature")
                feature_info = ""
                if os.path.exists(feature_folder):
                    feature_info += (
                        "## Available Feature Constants Files (in feature/ folder)\n"
                    )
                    for f in os.listdir(feature_folder):
                        fpath = os.path.join(feature_folder, f)
                        if f.endswith(".csv"):
                            try:
                                df = pd.read_csv(fpath, nrows=5)
                                feature_info += (
                                    f"- **{f}**: columns = {list(df.columns)}\n"
                                )
                            except:
                                feature_info += f"- **{f}**\n"
                        elif f.endswith(".ini"):
                            feature_info += f"- **{f}** (config file)\n"
                        elif f.endswith(".xlsx"):
                            feature_info += f"- **{f}** (Excel file)\n"
                    feature_info += "\nIMPORTANT: When calculating features, check these files in the feature/ folder for any constants or lookup tables you may need!\n"

                CONV_PROMPT = """You are a professional materials science ML engineer. Convert the external model into DiCerAgent plugin format.

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
g) EXACT reproduction of the original feature formula: when 'Original Model Code' below defines a specific feature-engineering recipe (e.g. A/B-site separated weighting like Sr_a - Sr_b, cation-only standard deviation, specific elemental property columns and data sources), the generated FeatureCalculator.calculate() MUST reproduce THAT EXACT recipe - same element grouping (A-site vs B-site vs cation vs total), same weightings, same standard-deviation denominators, same property table column mapping. Do NOT silently substitute a generic whole-composition weighted average for a site-separated formula. If the original code parses the formula into A/B sites (e.g. (A)2(B)2O7), replicate that parsing logic inside calculate() from the composition (e.g. via reduced_formula string parsing or an explicit site-assignment rule) so the numeric features match the original code's output for the same formula.
h) A/B-site parsing from the Structure object: pymatgen's structure.composition.formula / reduced_formula NEVER contains parentheses, so a regex like r'\\((.*?)\\)(\\d*\\.?\\d*)' applied to those strings will ALWAYS fail and raise 'Cannot parse A-site'. In calculate(), to recover the original A/B grouping, FIRST try: orig = getattr(structure, '_orig_formula', None); if orig is not None and '(' in orig, parse THAT string (it keeps the user's original parentheses, e.g. (Y0.2Gd0.2Er0.2Yb0.2Lu0.2)2Zr2O7) with the original regex. If no _orig_formula is available, derive the sites from structure.composition.get_el_amt_dict() with an explicit, physically-motivated site-assignment rule (e.g. for A2B2O7: B-site = Zr/Ti/Hf/Sn (or the element that pairs with O in the oxide), A-site = the remaining cations; normalize A/B-site fractions separately). NEVER apply a parenthesis regex to structure.composition.formula / reduced_formula strings.

i) Scalar input features MUST support dual-source resolution (base_features first, structure attribute second, default last): when the model has scalar input features that are NOT derived from composition (e.g. temperature, pressure, time), calculate(structure, base_features) MUST resolve them in this order: (1) from the base_features dict if the key is present (e.g. 'temperature'); (2) from a structure attribute if present (e.g. hasattr(structure, 'temperature') -> float(structure.temperature)); (3) a physically sensible default (e.g. 300.0 K) ONLY as the last resort. NEVER implement a single-source lookup: callers disagree on where scalars live - some pass them via base_features, some attach them to the structure object. A single-source implementation silently forces every sample to the default (e.g. all samples evaluated at 300K while the dataset spans 5K-1832K), which breaks temperature-dependent predictions WITHOUT raising any error and can only be caught by full-dataset feature-level comparison. Single-source scalar resolution is a conversion FAILURE.

j) Count-type composition features MUST be derived from the REDUCED formula, NEVER from the input structure's absolute cell size: features that count atoms per formula (e.g. na = number of atoms, and elemental composition columns expressed as absolute atom counts like Ca=1, O=1) MUST be computed from Composition(structure.composition.reduced_formula) - i.e. one formula unit - and NOT from structure.composition (the actual supercell/primitive cell), because the SAME material can be passed as Ca2O2 / Ca4O4 / mp-2605's 4-atom cell etc., and using the cell's absolute atom count silently changes na (e.g. 2 -> 40) and the element columns with cell size, breaking the trained model's input distribution without raising any error. This failure is only catchable by full-dataset feature-level comparison (e.g. all samples pinned to a wrong cell size). ALWAYS: red_comp = Composition(structure.composition.reduced_formula); use red_comp.get_el_amt_dict() for element columns and red_comp.num_atoms for na. Cell-dependent properties that ARE legitimately cell-based (density, volume) keep using the structure. Count-from-cell is a conversion FAILURE.

l) Materials-Project-sourced features (fepa, density, band_gap, etc.) MUST be fetched by REAL NETWORK CALLS at calculate/predict time - caching is NEVER a substitute:
   (i) The generated plugin MUST include a runtime MP-API fetch path (mp_api.client.MPRester) for EVERY external-property feature present in the training columns, with the API key read from settings.ini (configparser, utf-8) - NEVER hardcode 'YOUR_MP_KEY' or any literal key.
   (ii) Fetch by material_id FIRST when an mp_id is available; otherwise by reduced_formula. When multiple candidates match, pick the most STABLE one (min energy_above_hull) - request fields=['formation_energy_per_atom','energy_above_hull', ...] so the sort key is actually present. NEVER blindly take docs[0] without sorting.
   (iii) EMPTY result list / missing key / API exception MUST NOT be silently mapped to 0.0: 0.0 is a VALID-looking but WRONG value for fepa/density and silently corrupts predictions. Return None and let the caller treat the feature as MISSING (stage-degrade or explicit error), or raise an explicit error. Silent 0.0 fallback is a conversion FAILURE.
   (iv) A local cache is ALLOWED only as an optional speed-up BEHIND a global switch (e.g. USE_LOCAL_CACHE) that defaults to OFF in validation/testing. NEVER make the cache the primary or only source. Hardcoding cached values into the plugin source is a conversion FAILURE.
   (v) The self-check (constraint d) MUST exercise the REAL network path at least once (e.g. BaTiO3 -> fetch fepa/density from MP API), not just a mocked/stubbed value.

## STEP 0: BRANCH DECISION (MANDATORY)
{branch_info}

Follow the branch decided above:

- BRANCH A (existing model file(s) detected): the model files listed above will be copied verbatim into the plugin subdirectory. Generate plugin code that ONLY loads and adapts those files (joblib.load / torch.load / h5 / onnx as appropriate to the ACTUAL format shown by the probe). Do NOT include any training code. The Predictor MUST adapt to the ACTUAL top-level layout revealed by the probe (dict / plain list / single estimator / native format). If loading fails or the file is missing, predict() must return a dict with a "note" field carrying the reason; never silently return 0.0.
- BRANCH B (no model file): read 'Original Model Code' and 'Paper/Supporting Information' below, extract the REAL algorithm / hyperparameters / preprocessing / feature engineering / data split / target column, and output a structured TRAINING_CONFIG JSON (see the TRAINING_CONFIG OUTPUT section below). The plugin code MUST NOT contain any hardcoded default model (NO RandomForest fallback). If the algorithm cannot be determined, raise an explicit error instead of silently training a default model.

## Requirements
1. Output ONLY the Python code - no explanations
2. MUST import: from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
3. MUST import: from plugins.registry import register_plugin
4. MUST define metadata = ModelMetadata(name="...", version="1.0.0", author="...", description="...", supported_structures=["ABO3"], model_type=ModelType.PYTHON)
5. MUST create FeatureCalculator class that extends BaseFeatureCalculator with:
   - metadata = metadata
   - get_required_base_features() -> returns []
   - calculate(self, structure, base_features) -> returns features dict
   - get_required_features() -> returns list of feature names
6. MUST create Predictor class that extends BasePredictor with:
   - metadata = metadata
   - __init__() -> load models from model.pkl in the SUBDIRECTORY named after the plugin (see Requirement 8 for the exact path)
   - predict(self, features) -> receives features dict (NOT composition!), returns {{"property_mean": float, "property_dev": float}}
   - get_required_features() -> returns list of feature names
7. At the END of the file, call: register_plugin(Plugin(id="plugin_id", structure_type="ABO3", feature_calculator=FeatureCalculator(), predictor=Predictor(), metadata=metadata))
8. Model file path: the trained model is saved at plugins/external/<plugin_name>/model.pkl, i.e. a SUBDIRECTORY named after the plugin under the plugin file's directory. Derive the plugin name from __file__ and load with joblib.load:
   import joblib
   plugin_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
   model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_name, 'model.pkl')
   model_data = joblib.load(model_path)
9. Model file format: the model file is produced by the training system. Branch A: it is the verbatim copy of the detected source model file whose probe info is shown in the Branch info above; the Predictor MUST adapt to its ACTUAL layout. Branch B: it is the dict produced by TRAINING_CONFIG-driven retraining with keys: models (list of trained estimators, possibly wrapped in sklearn Pipelines), feature_cols, target_col, algorithm, hyperparameters, preprocessing, n_iterations, train_ratio, sklearn_version. NEVER hardcode "plain list of 100 RandomForestRegressor" unless the probe / actual file really is that; read the ACTUAL layout and adapt:
   - dict with 'models': model_data = joblib.load(model_path); models = model_data['models']; feature_cols = model_data.get('feature_cols', [])
   - plain list: model_data is a list of estimators, self.models = list(model_data)
   - in predict(): for each estimator (or pipeline) in models: pred = est.predict(X)[0] (a Pipeline needs no scaler.transform)
   - aggregate all predictions with np.mean / np.std
10. Model load failure: if the model file is missing or loading fails, predict() MUST NOT silently return 0.0; return a dict with a "note" field carrying the reason instead, e.g. {{"property_mean": None, "property_dev": None, "note": "<reason>"}}.
11. Feature table path: the feature constants file lives in the PROJECT ROOT feature/ folder (i.e. <project_root>/feature/Element_Property_Table.csv). Since the plugin file is at <project_root>/plugins/external/<plugin_name>.py, derive the project root with THREE os.path.dirname calls: feature_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'feature'); feature_table_path = os.path.join(feature_dir, 'Element_Property_Table.csv'); if os.path.exists(feature_table_path): self.feature_table = pd.read_csv(feature_table_path)
12. sklearn version compatibility: the model file may be trained with an older sklearn (e.g. 1.3.2) whose trees lack the monotonic_cst attribute required by newer sklearn (e.g. 1.8.0) during unpickling. After joblib.load, the Predictor MUST apply a compatibility patch before predicting: walk all tree estimators (dict/list nesting supported) and set monotonic_cst=None when missing. Standard pattern:
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
13. Feature alignment: the training data must use ONLY numeric feature columns that exactly match get_required_features(). NEVER mix text/non-numeric columns (e.g. Composition/formula strings) into features even after fillna(0) — that changes n_features and breaks predict(). If a column cannot be converted to numeric (all NaN after pd.to_numeric(errors='coerce')), exclude it from training features. In calculate(), produce EXACTLY the features listed in get_required_features(), with the SAME ORDER and SAME COUNT as the trained model (model.n_features_in_).

## TRAINING_CONFIG OUTPUT (Branch B ONLY)
When Branch B applies, after the plugin code, output the training configuration JSON between these markers:
<!--TRAINING_CONFIG_START-->
{{"algorithm": "SVR", "hyperparameters": {{"kernel": "rbf", "C": 3.1623, "gamma": 0.001, "epsilon": 0.01}}, "preprocessing": ["standard_scaler"], "feature_cols": ["H", "Li", "Ca", "O"], "target_col": "tec", "split": {{"train_ratio": 0.9, "random_state": 42}}}}
<!--TRAINING_CONFIG_END-->
Rules:
- algorithm: the REAL algorithm name used in the original code / paper (e.g. SVR, RandomForestRegressor, GradientBoostingRegressor, MLPRegressor, PLSRegression...).
- hyperparameters: the REAL hyperparameters from the original code / paper (e.g. for SVR: kernel, C, gamma, epsilon). If the original code performs a grid search, include the chosen best values (or the grid values).
- preprocessing: list of preprocessing steps applied in the original pipeline (e.g. ["standard_scaler"]); [] if none.
- feature_cols: the numeric feature column names used for training.
- target_col: the target column name.
- split: data split (train_ratio and random_state) used by the original code.
- NEVER emit a RandomForest config when the original model is not RandomForest. If you cannot determine the algorithm, do NOT invent one; the plugin code must raise an explicit error instead.

## Training Data Available
{training_data_info}

{feature_info}

{constants_info}

## Model Configuration
- Property Key: {prop_key}
- Property Symbol: {prop_symbol}
- Property Unit: {prop_unit}
- Ensemble Iterations: {n_ensembles}

## Special Requirements from User
{model_notes}

## Original Model Code (may contain multiple files for multi-step model)
{user_code}

## Paper/Supporting Information
{paper_context}

## IMPORTANT: Feature Column Detection
Analyze the training data columns to identify:
- stage1_feature_cols: features used in step 1 (composition-based, like H, Li, Ca, O)
- stage2_add_feature_cols: additional features in step 2 (like na, d, fepa)
- target column: the final property to predict

## How to Calculate Features in calculate()
Based on training data columns, implement:
- Element columns: Composition(structure.composition.reduced_formula).get_el_amt_dict()[elem] (atomic numbers; see constraint j - MUST use the reduced formula, NEVER the cell's absolute counts)
- na: Composition(structure.composition.reduced_formula).num_atoms (one formula unit - see constraint j)
- d: structure.density
- fepa: Fetch from Materials Project API using mp-api

## MP API Integration (if fepa / density / other MP-sourced features are in data)
REAL network calls only - caching is an optional speed-up behind USE_LOCAL_CACHE (default OFF). See constraint l.
Use this pattern:
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
    '''fepa: mp_id first, else reduced_formula. Stable candidate first (min energy_above_hull).
    Returns float, or None when MISSING - NEVER 0.0 (0.0 silently corrupts predictions).'''
    from mp_api.client import MPRester
    api_key = self._get_mp_api_key()
    if not api_key:
        return None
    with MPRester(api_key) as mpr:
        docs = (mpr.materials.summary.search(material_ids=[mp_id], fields=["formation_energy_per_atom", "energy_above_hull"])
                if mp_id else
                mpr.materials.summary.search(formula=composition.reduced_formula, fields=["formation_energy_per_atom", "energy_above_hull"]))
        if not docs:
            return None
        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
        val = getattr(doc, 'formation_energy_per_atom', None)
        return float(val) if val is not None else None

def _fetch_density(self, composition):
    '''density from MP API (needed when the input structure is a default skeleton, not a real cell).'''
    from mp_api.client import MPRester
    api_key = self._get_mp_api_key()
    if not api_key:
        return None
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(formula=composition.reduced_formula, fields=["density", "energy_above_hull"])
        if not docs:
            return None
        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
        val = getattr(doc, 'density', None)
        return float(val) if val is not None else None
```
Then in calculate(): if the caller passes only a skeleton structure and the training features include density,
resolve d as: explicit density arg -> MP API density (real network) -> structure.density.
"""
                # Two-branch: probe for ready-made model files under the source folder external_models/<model_name>/
                model_detection = analysis.get("model_files_detection") or detect_model_files(
                    analysis.get("folder_path", "")
                )
                branch = model_detection["branch"]
                plugin_name_branch = selected_folder if selected_folder else "plugin"
                plugin_dir_branch = os.path.join(
                    current_dir, "plugins", "external", plugin_name_branch
                )
                branch_info = build_branch_info(
                    model_detection,
                    analysis.get("folder_path", ""),
                    plugin_dir_branch,
                    plugin_name_branch,
                )

                with st.spinner("Generating plugin code..."):
                    converted_code, error = call_deepseek(
                        CONV_PROMPT.format(
                            prop_key=st.session_state.folder_analysis.get(
                                "prop_key", ""
                            ),
                            prop_symbol=st.session_state.folder_analysis.get(
                                "prop_symbol", ""
                            ),
                            prop_unit=st.session_state.folder_analysis.get(
                                "prop_unit", ""
                            ),
                            training_data_info=analysis.get("training_data_info", ""),
                            feature_info=feature_info,
                            constants_info=(
                                _get_app_converter()
                                .build_constants_info(
                                    analysis.get("constant_files", []),
                                    plugin_name=selected_folder or "plugin",
                                )[0]
                                or "No user-provided constant files."
                            ),
                            paper_context=analysis.get("paper_context", "")[:10000],
                            user_code=analysis.get("user_code", "")[:5000],
                            model_notes=analysis.get("model_notes", ""),
                            n_ensembles=analysis.get("n_ensembles", 100),
                            branch_info=branch_info,
                        )
                    )

                if error:
                    st.markdown(
                        f'<div class="error-msg">{error}</div>', unsafe_allow_html=True
                    )
                elif converted_code:
                    clean_code = extract_python_code(converted_code)
                    st.session_state.converted_code = clean_code
                    st.session_state.branch = branch
                    training_config, cfg_error = extract_training_config(converted_code)
                    if branch == "B" and training_config is None:
                        st.session_state.training_config = None
                        st.markdown(
                            f'<div class="error-msg">Training config extraction failed (Branch B must output TRAINING_CONFIG JSON; default model fallback is forbidden): {cfg_error}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.session_state.training_config = training_config
                        st.markdown(
                            f'<div class="success-msg">✨ Conversion successful! Branch {branch} detected.</div>',
                            unsafe_allow_html=True,
                        )

    if "converted_code" in st.session_state:
        st.markdown("---")
        st.markdown("### Step 5: Train Model")

        analysis = st.session_state.get("folder_analysis", {})

        target_col = st.session_state.get(
            "step4_target", analysis.get("target_col", "Unknown")
        )
        n_iterations = st.session_state.get(
            "step4_n_iters", analysis.get("n_iterations", 100)
        )
        stage1_cols = st.session_state.get(
            "step4_stage1", analysis.get("stage1_cols", [])
        )
        stage2_cols = st.session_state.get(
            "step4_stage2", analysis.get("stage2_cols", [])
        )

        st.info(
            f"Target: {target_col} | Stage1: {len(stage1_cols)} features | Stage2: {len(stage2_cols)} features"
        )

        train_btn = st.button(
            "🚀 Train Model", type="primary", use_container_width=True
        )

        if train_btn:
            # Two-branch training entry (faithful to the original model; hard-coded default templates are forbidden)
            branch = st.session_state.get("branch", "B")
            model_detection = analysis.get(
                "model_files_detection"
            ) or detect_model_files(analysis.get("folder_path", ""))
            plugin_dir = os.path.join(
                current_dir,
                "plugins",
                "external",
                selected_folder if selected_folder else "",
            )
            os.makedirs(plugin_dir, exist_ok=True)

            if branch == "A" or model_detection["branch"] == "A":
                # Branch A: reuse ready-made model files; only format/interface adaptation, no retraining
                with st.spinner("Copying existing model files (Branch A)..."):
                    try:
                        copy_result = copy_existing_models(
                            model_detection["files"], plugin_dir
                        )
                        if copy_result["errors"]:
                            st.error(
                                "Some model files failed to copy: "
                                + "; ".join(copy_result["errors"])
                            )
                        if copy_result["copied"]:
                            st.success(
                                "✅ Branch A: reused existing model files, training skipped.\n"
                                + "\n".join(
                                    f"- {p}" for p in copy_result["copied"]
                                )
                            )
                            st.session_state.trained = True
                        else:
                            st.error(
                                "Branch A: no reusable model files found. "
                                "Please check the external_models/<model_name>/ directory"
                            )
                    except Exception as e:
                        st.error(f"Branch A copy failed: {e}")
            else:
                # Branch B: retrain driven by the training config extracted by the LLM (default RandomForest fallback is forbidden)
                training_config = st.session_state.get("training_config")
                if not training_config:
                    st.error(
                        "Branch B: missing training config (TRAINING_CONFIG). Run "
                        "'🔄 Convert to Plugin Code' to extract the training config first; "
                        "default RandomForest template fallback is forbidden."
                    )
                else:
                    with st.spinner("Training model (Branch B, config-driven)..."):
                        try:
                            import numpy as np
                            import json as _json
                            import sklearn as _sklearn

                            all_feats = stage1_cols + stage2_cols
                            numeric_cols = []
                            for col in all_feats:
                                if col not in data_df.columns:
                                    continue
                                if pd.api.types.is_numeric_dtype(data_df[col]):
                                    numeric_cols.append(col)
                                else:
                                    # Only include a column when its content can be successfully converted to numeric;
                                    # plain-text columns (e.g. Composition) become all-NaN after conversion and are excluded directly,
                                    # to avoid fillna(0) mixing into training and causing n_features to mismatch the plugin's feature count.
                                    converted = pd.to_numeric(
                                        data_df[col], errors="coerce"
                                    )
                                    if converted.notna().any():
                                        data_df[col] = converted
                                        numeric_cols.append(col)

                            if not numeric_cols:
                                const_files = analysis.get("constant_files") or []
                                converted_code = st.session_state.get(
                                    "converted_code"
                                )
                                comp_col = _find_composition_col(data_df, target_col)
                                cfg_feats = list(
                                    training_config.get("feature_cols") or []
                                )
                                if (
                                    const_files
                                    and comp_col
                                    and converted_code
                                    and cfg_feats
                                ):
                                    fake_plugin_path = os.path.join(
                                        plugin_dir,
                                        (selected_folder or "plugin") + ".py",
                                    )
                                    try:
                                        feat_df, feats_ok = _auto_compute_features_from_composition(
                                            converted_code,
                                            comp_col,
                                            data_df,
                                            cfg_feats,
                                            fake_plugin_path,
                                            const_files,
                                        )
                                        if feats_ok:
                                            for col in feats_ok:
                                                data_df[col] = feat_df[col].values
                                            numeric_cols = feats_ok
                                            st.success(
                                                "Auto-computed "
                                                + str(len(feats_ok))
                                                + " numeric features from composition + constant tables: "
                                                + ", ".join(feats_ok[:8])
                                                + (
                                                    "..."
                                                    if len(feats_ok) > 8
                                                    else ""
                                                )
                                            )
                                    except Exception as _fe:
                                        import traceback as _tb
                                        st.warning(
                                            "Auto feature computation from composition failed: "
                                            + str(_fe)
                                            + "\n\n```\n"
                                            + _tb.format_exc()[-2500:]
                                            + "\n```"
                                        )
                                if not numeric_cols:
                                    if const_files:
                                        st.error(
                                            "No usable numeric feature columns in the training data. "
                                            "Constant files were detected - confirm the training data has a "
                                            "composition/formula column (e.g. Composition/Formula), "
                                            "and ensure Stage1/Stage2 feature columns are numeric features "
                                            "computed from the constant tables "
                                            "(e.g. composition-weighted averages of elemental properties), "
                                            "not raw text columns. "
                                            "If your training CSV only contains composition + target, "
                                            "the converter should auto-compute features at training time; "
                                            "if that failed, check the warning above, or upload a CSV that "
                                            "already contains the numeric feature columns "
                                            "(e.g. computed_features.csv)."
                                        )
                                    else:
                                        st.error(
                                            "No numeric features found! "
                                            "The training data lacks numeric feature columns. "
                                            "You can upload constant files "
                                            "(element property tables) and configure feature columns "
                                            "computed from formula + constants."
                                        )
                            else:
                                # Feature/target columns in the training config take priority (faithful to the original model)
                                cfg_feats = training_config.get("feature_cols") or []
                                use_feats = (
                                    [c for c in cfg_feats if c in numeric_cols]
                                    if cfg_feats
                                    else numeric_cols
                                )
                                if use_feats:
                                    numeric_cols = use_feats
                                cfg_target = training_config.get("target_col")
                                if cfg_target and cfg_target in data_df.columns:
                                    target_col = cfg_target
                                X = data_df[numeric_cols].fillna(0).values
                                y = pd.to_numeric(
                                    data_df[target_col], errors="coerce"
                                ).values

                                model_path = os.path.join(plugin_dir, "model.pkl")
                                progress_bar = st.progress(0)
                                save_data = train_with_config(
                                    X,
                                    y,
                                    numeric_cols,
                                    target_col,
                                    training_config,
                                    n_iterations,
                                    model_path,
                                    progress_cb=lambda f: progress_bar.progress(f),
                                )
                                # Record the sklearn version and feature info to help diagnose version-compatibility issues in the plugin
                                try:
                                    meta = {
                                        "sklearn_version": _sklearn.__version__,
                                        "algorithm": training_config.get("algorithm"),
                                        "n_models": len(save_data["models"]),
                                        "n_features": len(numeric_cols),
                                        "feature_cols": numeric_cols,
                                        "target_col": str(target_col),
                                        "training_config": training_config,
                                    }
                                    meta_path = os.path.join(
                                        plugin_dir, "model_meta.json"
                                    )
                                    with open(
                                        meta_path, "w", encoding="utf-8"
                                    ) as _f:
                                        _json.dump(
                                            meta, _f, ensure_ascii=False, indent=2
                                        )
                                except Exception:
                                    pass
                                st.success(
                                    f"✅ Model trained with "
                                    f"{training_config.get('algorithm')} "
                                    f"and saved to: {model_path}"
                                )
                                st.session_state.trained = True
                        except Exception as e:
                            st.error(f"Training failed: {e}")
        else:
            st.info("No training data available. Please select data files in Step 2.")

        st.markdown("---")
        st.markdown("### Step 6: Save Plugin")

        col_save1, col_save2 = st.columns([2, 1])
        with col_save1:
            plugin_name = st.text_input(
                "Plugin Name",
                value=selected_folder if selected_folder else "my_plugin",
                key=f"plugin_name_{selected_folder}",
            )
        with col_save2:
            save_btn = st.button(
                "💾 Save Plugin", type="primary", use_container_width=True
            )

        if save_btn:
            try:
                output_path = _get_app_converter().save_plugin_code(
                    st.session_state.converted_code,
                    plugin_name,
                    st.session_state.get("folder_analysis", {}).get(
                        "constant_files", []
                    ),
                )
                st.success(f"✅ Saved to: {output_path} (smoke test PASSED)")
            except Exception as e:
                st.error(
                    f"Save failed - the generated plugin did not pass the programmatic smoke check; saving was blocked: {e}"
                )

        st.markdown("---")
        st.markdown("### Generated Plugin Code")
        st.code(st.session_state.converted_code, language="python")

    with sub_tab2:
        st.markdown("### Upload Mode")
        st.info("Upload model code or papers to convert to plugin format.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Step 1: Upload Model Code")

            uploaded_code = st.file_uploader(
                "Drag & drop code file (Python, R, or text)",
                type=["py", "r", "txt"],
                key="upload_code_main",
                help="Upload your model code file",
            )

            uploaded_extra_codes = st.file_uploader(
                "Upload additional model files",
                type=["py", "r", "txt"],
                accept_multiple_files=True,
                key="upload_code_extra",
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
                key="upload_pdfs",
            )

            paper_context = ""
            if uploaded_pdfs:
                for pdf in uploaded_pdfs:
                    st.markdown(
                        f'<span class="pdf-tag">{pdf.name}</span>',
                        unsafe_allow_html=True,
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
                key="upload_consts",
            )
            upload_constant_files = []
            if uploaded_consts:
                os.makedirs("temp_uploads", exist_ok=True)
                for cf in uploaded_consts:
                    tmp_path = os.path.join("temp_uploads", cf.name)
                    with open(tmp_path, "wb") as f:
                        f.write(cf.getbuffer())
                    upload_constant_files.append({"name": cf.name, "path": tmp_path})
                    st.markdown(
                        f'<span class="pdf-tag">Constant: {cf.name}</span>',
                        unsafe_allow_html=True,
                    )

        with col2:
            st.markdown("#### Step 3: Define Output Properties")

            col_prop1, col_prop2, col_prop3 = st.columns(3)
            with col_prop1:
                prop_key = st.text_input(
                    "Property Key", value="melt_temp", key="upload_prop_key"
                )
            with col_prop2:
                prop_symbol = st.text_input(
                    "Display Symbol", value="Tm", key="upload_prop_symbol"
                )
            with col_prop3:
                prop_unit = st.text_input("Unit", value="K", key="upload_prop_unit")

            st.markdown("#### Step 4: Generate Plugin")
            plugin_name_input = st.text_input(
                "Plugin name:", value="my_predictor", key="upload_plugin_name"
            )

            convert_btn = st.button(
                "Convert to Plugin",
                type="primary",
                use_container_width=True,
                key="upload_convert_btn",
            )

            if convert_btn:
                if not user_code.strip() and not paper_context.strip():
                    st.error("Please upload model code or paper PDF")
                else:
                    with st.spinner("AI is converting..."):
                        upload_constants_info = (
                            _get_app_converter().build_constants_info(
                                upload_constant_files
                            )[0]
                            or "No user-provided constant files."
                        )
                        CONVERSION_PROMPT = f'''You are a materials science ML expert. Convert external prediction models into DiCerAgent plugin format.

CRITICAL: OUTPUT ONLY THE PYTHON CODE. NO EXPLANATIONS.

## Output Properties
- Property: {prop_symbol} ({prop_unit})
- Key: {prop_key}_mean, {prop_key}_dev

## Model Code:
{user_code or "No code provided"}

## Paper Context:
{paper_context or "No paper provided"}

## User-Provided Constant Files (MUST embed into plugin code):
{upload_constants_info}

IMPORTANT: If constant files are listed above, embed their content into the generated plugin code
as module-level constants (e.g. dict/DataFrame). Compute features from the chemical formula alone
using these constants (e.g. composition-weighted average of elemental properties). NEVER let the
plugin raise "No numeric features found!" when the input is only a chemical formula.

Generate a complete plugin with:
1. ModelMetadata with name="{plugin_name_input}", author="User", supported_structures=["general"], model_type=ModelType.PYTHON
2. FeatureCalculator class that calculates features from structure
3. Predictor class that loads model and makes predictions
4. Plugin registration

Use these imports:
from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin, register_property_info
from plugins.registry import register_plugin
import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
'''

                        converted_code, error = call_deepseek_conversion(
                            CONVERSION_PROMPT, paper_context
                        )

                        if error:
                            st.markdown(
                                f'<div class="error-box">{error}</div>',
                                unsafe_allow_html=True,
                            )
                        elif converted_code:
                            clean_code = extract_python_code(converted_code)
                            st.session_state.converted_code_upload = clean_code
                            st.markdown(
                                '<div class="success-box">Conversion successful!</div>',
                                unsafe_allow_html=True,
                            )

        if "converted_code_upload" in st.session_state:
            st.markdown("---")
            st.markdown("### Generated Plugin Code")
            st.code(st.session_state.converted_code_upload, language="python")

            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                save_name = st.text_input(
                    "Plugin filename:", value=plugin_name_input, key="upload_save_name"
                )
            with col_save2:
                if st.button("Save Plugin", key="upload_save_btn"):
                    try:
                        os.makedirs("plugins/external", exist_ok=True)
                        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", save_name)
                        with open(
                            f"plugins/external/{safe_name}.py", "w", encoding="utf-8"
                        ) as f:
                            f.write(st.session_state.converted_code_upload)
                        st.success(f"Saved to: plugins/external/{safe_name}.py")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

    with sub_tab3:
        st.markdown("### Plugin Template (Two-Stage Model)")
        template_code = """from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin
import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Elemental composition columns
ELEMENT_COLS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 
                'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
                'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
                'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm']

metadata = ModelMetadata(
    name="Your Model Name",
    version="1.0.0",
    author="Author Name",
    description="Two-stage ML model with MP integration",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

class YourFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    
    def __init__(self):
        self.feature_cols = []  # Loaded from model
    
    def calculate(self, structure, base_features):
        features = base_features.copy()
        composition = structure.composition
        
        # Stage 1: Elemental composition (atomic numbers)
        # REAL API: structure.composition -> Composition; el_amt = structure.composition.get_el_amt_dict()
        #   -> {'Ba':1,'Ti':1,'O':3} (element symbol -> amount); num_atoms for total; /total for mole fraction.
        # NEVER call structure.get_el_amt_dict() directly - Structure has NO such method.
        el_amt = composition.get_el_amt_dict()
        
        # MISSING-VALUE SEMANTICS (HARD CONSTRAINT b, training-side convention fillna(median)):
        # If a feature comes from Element_Property_Table.csv weighted averages:
        #   - load table ONCE and pre-fill NaN with the column MEDIAN:
        #       table = pd.read_csv(feature_table_path).fillna(pd.read_csv(feature_table_path).median(numeric_only=True))
        #     or in two steps: table = pd.read_csv(feature_table_path); table = table.fillna(table.median(numeric_only=True))
        #   - per-element lookup is NaN-safe (dict.get with default / pd.isna check):
        #       row = table.loc[table['Symbol'] == el]
        #       val = row[col].iloc[0] if len(row) else None
        #       if val is None or pd.isna(val): continue  # skip element - NEVER raise "Element 'X' has no value for property 'Y'"
        #   - if EVERY element is skipped for a column (weighted result is NaN): raise explicit error (case ii)
        #   - if the column NAME cannot be resolved (after alias/fuzzy): raise explicit error (case iii)
        for elem in ELEMENT_COLS:
            features[elem] = el_amt.get(elem, 0)
        
        # Stage 2: Custom features (if in training data)
        if 'na' in self.feature_cols:
            features['na'] = composition.num_atoms
        if 'd' in self.feature_cols:
            features['d'] = structure.density
        if 'fepa' in self.feature_cols:
            # Get formation energy from Materials Project - REAL network call (constraint l)
            try:
                from mp_api.client import MPRester
                comp_str = composition.reduced_formula
                with MPRester(api_key=self._get_mp_api_key()) as mpr:
                    docs = mpr.materials.summary.search(formula=comp_str, fields=['formation_energy_per_atom', 'energy_above_hull'])
                    if docs:
                        doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
                        features['fepa'] = doc.formation_energy_per_atom
                    else:
                        features['fepa'] = None  # MISSING - NEVER 0.0
            except:
                features['fepa'] = None  # MISSING - NEVER 0.0 (silent 0.0 corrupts predictions)
        
        return features
    
    def _get_mp_api_key(self):
        import configparser
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.ini')
        with open(config_path, 'r', encoding='utf-8') as f:
            config.read_file(f)
        return config.get('DEFAULT', 'MP_API_KEY', fallback='')
    
    def get_required_base_features(self):
        return ["Mass", "Va", "P_MLR_pv"]

class YourPredictor(BasePredictor):
    metadata = metadata
    
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_path = os.path.join(self.model_dir, "two_stage_model.pkl")
        self.models_stage1 = None
        self.models_stage2 = None
        self.stage1_cols = ELEMENT_COLS
        self.stage2_cols = []
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage2 = StandardScaler()
        self._load_models()
    
    def _load_models(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.models_stage1 = data["models_stage1"]
                self.models_stage2 = data["models_stage2"]
                self.stage1_cols = data.get("stage1_cols", ELEMENT_COLS)
                self.stage2_cols = data.get("stage2_cols", [])
                self.scaler_stage1 = data.get("scaler_stage1", StandardScaler())
                self.scaler_stage2 = data.get("scaler_stage2", StandardScaler())
            except:
                self.models_stage1 = None
                self.models_stage2 = None
    
    def _extract_stage_features(self, df, feature_cols):
        return df[feature_cols].values
    
    def train(self, features_df, target_values, stage1_feature_cols, stage2_add_feature_cols=None):
        # Stage 1: Train on composition features
        X1 = self._extract_stage_features(features_df, stage1_feature_cols)
        y = np.array(target_values)
        X1_scaled = self.scaler_stage1.fit_transform(X1)
        self.models_stage1 = [RandomForestRegressor(n_estimators=100, random_state=42+i) for i in range(3)]
        for m in self.models_stage1:
            m.fit(X1_scaled, y)
        
        # Get stage 1 predictions
        stage1_pred = np.mean([m.predict(X1_scaled) for m in self.models_stage1], axis=0)
        
        # Stage 2: Train on base features + stage1 prediction
        features_df_copy = features_df.copy()
        features_df_copy['stage1_pred'] = stage1_pred
        all_stage2_cols = stage1_feature_cols + (stage2_add_feature_cols or []) + ['stage1_pred']
        X2 = features_df_copy[all_stage2_cols].values
        X2_scaled = self.scaler_stage2.fit_transform(X2)
        self.models_stage2 = [RandomForestRegressor(n_estimators=100, random_state=42+i) for i in range(3)]
        for m in self.models_stage2:
            m.fit(X2_scaled, y)
        
        # Save all models
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump({
            "models_stage1": self.models_stage1,
            "models_stage2": self.models_stage2,
            "stage1_cols": stage1_feature_cols,
            "stage2_cols": all_stage2_cols,
            "scaler_stage1": self.scaler_stage1,
            "scaler_stage2": self.scaler_stage2
        }, self.model_path)
    
    def predict(self, features):
        if self.models_stage1 is None:
            return {"property_mean": 0.0, "property_dev": 0.0}
        
        # Stage 1
        stage1_X = np.array([[features.get(f, 0) for f in self.stage1_cols]])
        stage1_scaled = self.scaler_stage1.transform(stage1_X)
        stage1_pred = np.mean([m.predict(stage1_scaled) for m in self.models_stage1], axis=0)[0]
        
        # Stage 2
        all_features = self.stage1_cols + self.stage2_cols
        stage2_X = np.array([[features.get(f, 0) for f in all_features]])
        stage2_X[0][len(all_features)-1] = stage1_pred
        stage2_scaled = self.scaler_stage2.transform(stage2_X)
        final_pred = np.mean([m.predict(stage2_scaled) for m in self.models_stage2], axis=0)[0]
        
        return {"property_mean": float(final_pred), "property_dev": 0.0}
    
    def get_required_features(self):
        return self.stage1_cols + self.stage2_cols
    
    def get_stage1_features(self):
        return self.stage1_cols
    
    def get_stage2_features(self):
        return self.stage1_cols + self.stage2_cols

register_plugin(Plugin(
    id="your_model_id",
    structure_type="general",
    feature_calculator=YourFeatureCalculator(),
    predictor=YourPredictor()
))"""
        st.code(template_code, language="python")

        st.markdown("---")
        st.markdown("### Simple Template (Single Stage)")
        simple_template = """from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin
import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

metadata = ModelMetadata(
    name="Your Model Name",
    version="1.0.0",
    author="Author Name",
    description="Simple single-stage model",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

class YourFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    def calculate(self, structure, base_features):
        return base_features.copy()
    def get_required_base_features(self):
        return ["Mass", "Va", "P_MLR_pv"]

class YourPredictor(BasePredictor):
    metadata = metadata
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(self.model_path):
            try: self.model = joblib.load(self.model_path)
            except: self.model = None
    
    def train(self, features_df, target_values):
        X = features_df.values
        y = np.array(target_values)
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def predict(self, features):
        if self.model is None:
            return {"property_mean": 0.0, "property_dev": 0.0}
        X = np.array([[features.get(f, 0) for f in self.get_required_features()]])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        return {"property_mean": float(prediction), "property_dev": 0.0}
    
    def get_required_features(self):
        return []

register_plugin(Plugin(
    id="your_model_id",
    structure_type="general",
    feature_calculator=YourFeatureCalculator(),
    predictor=YourPredictor()
))"""
        st.code(simple_template, language="python")

    with sub_tab4:
        st.markdown("### Installed Plugins")

        try:
            from plugins import global_registry, load_all_plugins

            load_all_plugins()

            plugins = global_registry.list_all()

            if plugins:
                for p in plugins:
                    with st.expander(f"📦 {p.id} ({p.structure_type})"):
                        meta = p.metadata
                        if meta:
                            st.markdown(f"**Name:** {meta.name}")
                            st.markdown(f"**Version:** {meta.version}")
                            st.markdown(f"**Author:** {meta.author}")
                            st.markdown(f"**Description:** {meta.description}")
                        status = "✅ Enabled" if p.enabled else "❌ Disabled"
                        st.markdown(f"**Status:** {status}")
            else:
                st.info("No plugins registered yet.")
        except Exception as e:
            st.warning(f"Could not load plugin registry: {e}")

        st.markdown("---")
        st.markdown("#### Plugin Files (plugins/external)")
        plugin_dir = os.path.join(current_dir, "plugins", "external")
        if os.path.exists(plugin_dir):
            py_files = [
                f
                for f in os.listdir(plugin_dir)
                if f.endswith(".py") and not f.startswith("_")
            ]
            if py_files:
                for f in py_files:
                    st.markdown(f"📄 {f}")
            else:
                st.info("No plugin files found.")
        else:
            st.info("Plugin directory does not exist.")

with tab_settings:
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Settings</p>
        <p class="subtitle">Configuration and help information</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔑 API Configuration")

    col_api1, col_api2 = st.columns(2)
    with col_api1:
        ds_key = st.text_input(
            "DeepSeek API Key", value=DEEPSEEK_API_KEY, type="password"
        )
        st.caption("Used for report generation and model conversion")
    with col_api2:
        ds_url = st.text_input("DeepSeek API URL", value=DEEPSEEK_API_URL)
        st.caption("Default: https://api.deepseek.com")

    col_model1, col_model2 = st.columns(2)
    with col_model1:
        ds_model = st.selectbox(
            "DeepSeek Model",
            options=["deepseek-chat", "deepseek-v4-pro"],
            index=0 if DEEPSEEK_MODEL == "deepseek-chat" else 1,
            key="ds_model_setting",
        )
    st.markdown("---")

    if st.button("💾 Save Settings"):
        import configparser
        cfg = configparser.ConfigParser()
        cfg_path = config.settings_path
        if os.path.exists(cfg_path):
            cfg.read(cfg_path, encoding='utf-8')
        if 'DEFAULT' not in cfg:
            cfg['DEFAULT'] = {}
        cfg['DEFAULT']['DS_API_KEY'] = ds_key
        cfg['DEFAULT']['DEEPSEEK_API_URL'] = ds_url
        cfg['DEFAULT']['DS_MODEL'] = ds_model
        with open(cfg_path, 'w', encoding='utf-8') as f:
            cfg.write(f)
        st.session_state.ds_key_saved = ds_key
        st.session_state.ds_url_saved = ds_url
        st.session_state.ds_model_saved = ds_model
        st.success("Settings saved! Restart to apply model change.")

    st.markdown("---")
    st.markdown("### ℹ️ Help & Documentation")

    with st.expander("🚀 Quick Start"):
        st.markdown("""
        **1. Set up the environment**
        - Use the conda environment `MWDCsAI` (Python 3.10+)
        - Install R 4.3+ and make sure `Rscript.exe` is available
        - Run: `streamlit run app_unified.py`

        **2. Configure your API keys**
        - Open `settings.ini` in the project root
        - Fill in `MP_API_KEY` (Materials Project), `DS_API_KEY` (DeepSeek), and `HF_TOKEN` (Hugging Face)
        - Replace `YOUR_RSCRIPT_PATH` with the real path to `Rscript.exe` on your machine

        **3. Launch**
        - Start the app and switch to any tab to begin
        - If results look stale, use the "🧹 Clear Cache" button at the bottom of the page
        """)

    with st.expander("🧭 Module Guides"):
        st.markdown("""
        **ARMPP (Forward Prediction)**
        - Enter a material formula or ID, e.g. `CaWO4`, `mp-19426`, or `BaTiO3 --cod`
        - Click Analyze: the system predicts dielectric properties, retrieves relevant literature, and generates a technical report
        - View predicted values, literature comparison, and full references in the HRKE tab

        **HRKE (Hybrid-Retrieval Knowledge Extractor)**
        - Searches a local knowledge base combining literature entries and material property tables
        - Use the search box to query by material, property, or keyword
        - Retrieved references are shared with other tabs for report generation

        **CADO (Context-Aware Dialogue Orchestrator)**
        - Ask questions in natural language, e.g. "predict BaTiO3", "find CaWO4 literature", or "convert my model"
        - The system routes your request to the appropriate agent automatically
        - Session context is preserved within a conversation

        **MPC (Model-to-Plugin Converter)**
        - Folder Mode: put model code, training data, and a reference paper in a folder under `external_models/`, select it, and confirm
        - Upload Mode: upload your files directly in the browser
        - Template: download a template folder to learn the required structure
        - Installed Plugins: review and manage converted plugins in `plugins/external/`
        - Converted plugins become available immediately after installation

        **IDO (Inverse Design Optimizer)**
        - Optimizes compositions/structures toward target dielectric properties
        - Requires `mace`, `faiss`, and `pymatgen` in the active environment
        - Outputs candidate structures with predicted properties
        """)

    with st.expander("⚙️ Settings & Configuration"):
        st.markdown("""
        All configuration lives in `settings.ini`:

        | Key | Description |
        |---|---|
        | `R_EXEC_PATH` | Full path to `Rscript.exe` (required for R-based models) |
        | `MP_API_KEY` | Materials Project API key, apply at https://next-gen.materialsproject.org/api |
        | `DS_API_KEY` | DeepSeek API key |
        | `DS_MODEL` | DeepSeek model name, e.g. `deepseek-chat` |
        | `HF_TOKEN` | Hugging Face token (required for some model downloads) |

        Placeholder values starting with `YOUR_` must be replaced with your own credentials before first use.
        """)

    with st.expander("📦 Data & Models"):
        st.markdown("""
        - The bundled database is a **10% random sample** (seed 42) of the original dataset for redistribution; `CaWO4` is kept as a complete case study with all properties, literature, vectors, and CIF files
        - If the large FAISS index is missing, the knowledge engine silently falls back to the CSV-based retrieval channel
        - `external_models/` contains example conversion cases (e.g. high-entropy ceramic thermal conductivity, oxide melting temperature)
        """)

    with st.expander("📁 File Structure"):
        st.markdown("""
        ```
        DiCerAgent/
        ├── app_unified.py        # Main Streamlit entry (all tabs)
        ├── app_converter.py      # Model-to-plugin converter
        ├── chat_module.py        # Chat & intent routing (CADO)
        ├── orchestrator.py       # Central orchestration
        ├── main.py               # Standalone entry
        ├── config.py             # Configuration loader
        ├── settings.ini          # User configuration (API keys, R path)
        ├── features.py / extractor.py / data_loader.py   # Feature engineering
        ├── forward_prediction.py # ARMPP prediction engine
        ├── knowledge_engines.py / literature.py / nlp_processor.py  # HRKE
        ├── inverse_design.py     # IDO
        ├── structure_classifier.py / viz_component.py / get_cod_cif.py
        ├── plugins/
        │   ├── builtin/          # Built-in property plugins
        │   └── external/         # Converted external plugins
        ├── external_models/      # Model folders for MPC conversion
        ├── model/                # Pretrained ML models
        ├── feature/              # Feature tables
        ├── inverse_db/           # Material database (properties + vectors + FAISS index)
        ├── faiss_index/          # FAISS indexes for knowledge retrieval
        ├── enhanced_shared/      # Shared knowledge-base indexes
        ├── prediction/           # Prediction outputs & logs
        ├── llm_enhanced_ml/      # LLM-assisted ML utilities
        ├── LLM/                  # Local LLM/hub files
        └── logo/                 # UI assets
        ```
        """)

    with st.expander("🛠️ Troubleshooting"):
        st.markdown("""
        - **"Rscript not found"** — check `R_EXEC_PATH` in `settings.ini` and confirm R 4.3+ is installed
        - **401 / invalid key errors** — verify `MP_API_KEY`, `DS_API_KEY`, and `HF_TOKEN` are valid and active
        - **Knowledge engine not loaded** — this is expected if the large FAISS index is absent; the CSV fallback is used automatically
        - **IDO fails to load** — install `mace`, `faiss`, and `pymatgen` in the active environment
        - **Stale results / unexpected states** — click the "🧹 Clear Cache" button at the bottom of the page
        """)

# ========== IDO (Inverse Design Optimizer) Tab ==========
with tab_inverse:
    if HAS_INVERSE_DESIGN:
        try:
            inverse_design.render()
        except Exception as e:
            st.error(f"Inverse design module failed to load: {e}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc())
    else:
        st.info(
            "Inverse design module not installed. Make sure `inverse_design.py` and its dependencies (mace, faiss, pymatgen) are ready."
        )

# ==================== About Tab ====================
with tab_about:
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">About DiCerAgent</p>
        <p class="subtitle">Paper citation, author profiles and research team</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "**DiCerAgent** is an "
        "AI-driven integrated multi-agent research framework for dielectric ceramics. "
        "It integrates five specialized agents — the Hybrid-Retrieval Knowledge Extractor (HRKE), "
        "the Model-to-Plugin Converter (MPC), the Adaptive-Routing Multi-Property Predictor (ARMPP), "
        "the Inverse Design Optimizer (IDO), and the Context-Aware Dialogue Orchestrator (CADO) — "
        "covering hybrid knowledge retrieval, model capability orchestration, multi-property "
        "prediction, inverse design, and context-aware scientific reasoning."
    )

    st.markdown("#### 📌 Citation")
    st.markdown(
        "Jincheng Qin¹\*, Liangyu Mo¹˒², Mingyue Yang¹˒², Faqiang Zhang¹, "
        "Mingsheng Ma¹˒², Yongxiang Li³, Zhifu Liu¹˒²\*  \n"
        "**DiCerAgent: A Multi-Agent Research Platform for Dielectric Ceramics**  \n"
        "1. The State Key Laboratory of High Performance Ceramics and Superfine Structure, "
        "Shanghai Institute of Ceramics, Chinese Academy of Sciences, Shanghai 201899, China  \n"
        "2. Center of Materials Sciences and Optoelectronics Engineering, "
        "University of Chinese Academy of Sciences, Beijing 100049, China  \n"
        "3. School of Engineering, RMIT University, Melbourne, VIC 3000, Australia"
    )

    st.markdown("#### 👤 Author Profiles")
    st.markdown(
        "- **Jincheng Qin**: [qinjcsiccas.github.io/cv](https://qinjcsiccas.github.io/cv/)\n"
        "- **Zhifu Liu**: [english.sic.cas.cn — research profile](https://english.sic.cas.cn/sourcedb/rck/fs/202311/t20231107_611735.html)\n"
        "- **Mingsheng Ma**: [english.sic.cas.cn — research profile](https://english.sic.cas.cn/sourcedb/rck/fs/201704/t20170418_653509.html)"
    )

    st.markdown("#### 🏛️ Research Team")
    st.markdown(
        "Functional Ceramics Research Group, Shanghai Institute of Ceramics, CAS — "
        "team introduction (Chinese):  \n"
        "[www.sic.cas.cn - Group Introduction](https://www.sic.cas.cn/kybm/kybm4/yjly/lyx/gk/)"
    )

    st.markdown("#### ✉️ Contact")
    st.markdown(
        "For questions, feedback, or collaboration, please contact the developer:  \n"
        "**qinjccas@gmail.com**"
    )

# ============================================================
# Global footer area: Clear Cache + copyright (below all tabs, always visible)
# ============================================================
st.markdown("---")
col_cr_main, col_cc_main = st.columns([4, 1])
with col_cr_main:
    st.markdown("© 2026 Jincheng Qin. All rights reserved.")
with col_cc_main:
    if st.button("🧹 Clear Cache", type="secondary", key="clear_cache_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
