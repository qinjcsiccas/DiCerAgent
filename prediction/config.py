"""
Enhanced Prediction Module Configuration
----------------
Central management of: element tables, property parameters, LLM configuration, model parameters, paths
Merged from llm_enhanced_ml/config.py and the project-global config.py
"""

import os
import configparser

# ============================================================
# Project paths
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCED_SHARED_DIR = os.path.join(PROJECT_DIR, "enhanced_shared")

# Knowledge retrieval data
KR_DATA_DIR = os.path.join(PROJECT_DIR, "data", "kg_deepseek_data")  # optional raw data dir for rebuilding the knowledge retrieval index (not shipped in the slim version)
KR_INDEX_PATH = os.path.join(ENHANCED_SHARED_DIR, "kr_index_118.pkl")

# Source project data
SRC_MAPPING = os.path.join(PROJECT_DIR, "llm_enhanced_ml", "data", "mapping")
SRC_CIF = os.path.join(PROJECT_DIR, "llm_enhanced_ml", "data", "cif_files")

# Feature data (project global)
FEATURE_DIR = os.path.join(PROJECT_DIR, "feature")

# ============================================================
# API configuration
# ============================================================
_settings_path = os.path.join(PROJECT_DIR, "settings.ini")
DS_API_KEY = ""
DEEPSEEK_URL = "https://api.deepseek.com"
if os.path.exists(_settings_path):
    conf = configparser.ConfigParser()
    conf.read(_settings_path, encoding="utf-8")
    DS_API_KEY = conf["DEFAULT"].get("DS_API_KEY", "")
    DEEPSEEK_URL = conf["DEFAULT"].get("DEEPSEEK_API_URL", DEEPSEEK_URL)

LLM_MODEL = conf["DEFAULT"].get("DS_MODEL", "deepseek-chat") if os.path.exists(_settings_path) else "deepseek-chat"
LLM_TIMEOUT = 90
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.2
LLM_MAX_RETRIES = 2

# ============================================================
# Periodic table (118-element full table, used for formula vectorization)
# ============================================================
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# ============================================================
# Target property parameters
# ============================================================
PROPERTY_CONFIG = {
    "er": {
        "name": "Dielectric Constant",
        "unit": "",
        "range": (1.0, 500.0),
        "kr_field": "er",
        "default_pred": 20.0,
        "ml_model": "svr",
        "ml_params": {"C": 100, "gamma": 0.1, "kernel": "rbf"},
        "feat_cols": ["Mass", "Va", "P_MLR_pv", "ASD", "Nbr_dist_var", "Bond_abs_dev"],
        "data_type": "cif",
    },
    "qf": {
        "name": "Quality Factor",
        "unit": "GHz",
        "range": (100.0, 200000.0),
        "kr_field": "qxf",
        "default_pred": 20000.0,
    },
    "tcf_abo3": {
        "name": "TCF ABO3",
        "unit": "ppm/degC",
        "range": (-300.0, 300.0),
        "kr_field": "tau_f",
        "default_pred": 0.0,
    },
    "tcf_abo4": {
        "name": "TCF ABO4",
        "unit": "ppm/degC",
        "range": (-300.0, 300.0),
        "kr_field": "tau_f",
        "default_pred": 0.0,
    },
    "tm": {
        "name": "Melting Temperature",
        "unit": "K",
        "range": (500.0, 4000.0),
        "kr_field": None,  # no literature data yet, forced to use LLM-ML
        "default_pred": 2000.0,
    },
}

# ============================================================
# Retrieval parameters
# ============================================================
K_NEIGHBORS = 8
KR_SEARCH_MULTIPLIER = 3
MIN_KR_NEIGHBORS = 3          # minimum number of relevant neighbors
MIN_NEIGHBOR_SIMILARITY = 0.5  # minimum neighbor similarity threshold
