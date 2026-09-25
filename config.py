import os
import configparser

# ================= 1. Auto-detect project root =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_DIR = os.path.join(BASE_DIR, "model")
BASE_FEATURE_DIR = os.path.join(BASE_DIR, "feature")

# ================= 2. User configuration (supports settings.ini) =================
DEFAULT_R_EXEC = r"YOUR_RSCRIPT_PATH"
DEFAULT_MP_KEY = "YOUR_MP_API_KEY"
DEFAULT_DS_KEY = ""
DEFAULT_DS_URL = "https://api.deepseek.com"
DEFAULT_DS_MODEL = "deepseek-chat"  # flash (saves tokens), can be changed to deepseek-v4-pro
DEFAULT_HF_TOKEN = ""

settings_path = os.path.join(BASE_DIR, "settings.ini")
conf = configparser.ConfigParser()

if os.path.exists(settings_path):
    conf.read(settings_path, encoding='utf-8')
    R_SCRIPT_EXEC = conf['DEFAULT'].get('R_EXEC_PATH', DEFAULT_R_EXEC)
    MP_API_KEY = conf['DEFAULT'].get('MP_API_KEY', DEFAULT_MP_KEY)
    DS_API_KEY = conf['DEFAULT'].get('DS_API_KEY', DEFAULT_DS_KEY)
    DEEPSEEK_API_URL = conf['DEFAULT'].get('DEEPSEEK_API_URL', DEFAULT_DS_URL)
    DS_MODEL = conf['DEFAULT'].get('DS_MODEL', DEFAULT_DS_MODEL)
    HF_TOKEN = conf['DEFAULT'].get('HF_TOKEN', DEFAULT_HF_TOKEN)
else:
    R_SCRIPT_EXEC = DEFAULT_R_EXEC
    MP_API_KEY = DEFAULT_MP_KEY
    DS_API_KEY = DEFAULT_DS_KEY
    DEEPSEEK_API_URL = DEFAULT_DS_URL
    DS_MODEL = DEFAULT_DS_MODEL
    HF_TOKEN = DEFAULT_HF_TOKEN

# ================= 3. Auto-generated file paths =================

# --- Inverse design (inverse_db) data paths ---
INVERSE_DB_DIR = os.path.join(BASE_DIR, "inverse_db")
PATH_FAISS_INDEX = os.path.join(INVERSE_DB_DIR, "faiss_index.bin")
PATH_MATERIAL_IDS = os.path.join(INVERSE_DB_DIR, "material_ids.npy")
PATH_MATERIAL_PROPERTIES = os.path.join(INVERSE_DB_DIR, "material_properties.csv")
PATH_FINAL_CRYSTAL_VECTORS = os.path.join(INVERSE_DB_DIR, "final_crystal_vectors.npy")
PATH_MACE_MODEL = os.path.join(INVERSE_DB_DIR, "2024-01-07-mace-128-L2_epoch-199.model")

# --- MatterGen ---
MATTERGEN_DIR = r"YOUR_MATTERGEN_DIR"  # external model dir, points to real MatterGen checkout
PATH_MATTERGEN_CHECKPOINT = os.path.join(MATTERGEN_DIR, "checkpoints", "mattergen_base")
PATH_MATTERGEN_CHECKPOINT_SG = os.path.join(MATTERGEN_DIR, "checkpoints", "space_group")
PATH_MATTERGEN_OUTPUT = os.path.join(INVERSE_DB_DIR, "mattergen_output")

# --- Physical property tables ---
PATH_MLR_TABLE      = os.path.join(BASE_FEATURE_DIR, "MLR_polarizability.csv")
PATH_ML_TABLE       = os.path.join(BASE_FEATURE_DIR, "ML_polarizability.csv")
PATH_ZHANG_EN       = os.path.join(BASE_FEATURE_DIR, "Zhang_electronegativity.csv")
PATH_BOND_LENGTH    = os.path.join(BASE_FEATURE_DIR, "Bond_length.csv")
PATH_BOND_VALENCE   = os.path.join(BASE_FEATURE_DIR, "Bond_valence.csv")
PATH_BOND_COVALENCY = os.path.join(BASE_FEATURE_DIR, "Bond_covalency.csv")

# --- Enhanced prediction ---
ENHANCED_SHARED_DIR = os.path.join(BASE_DIR, "enhanced_shared")
PATH_ENHANCED_KG_INDEX = os.path.join(ENHANCED_SHARED_DIR, "kr_index_118.pkl")

# --- Machine learning prediction models (Python) ---
PATH_SVR_MODEL = os.path.join(BASE_MODEL_DIR, "SVR_k.pkl")
PATH_ABO3_TCF_MODEL = os.path.join(BASE_MODEL_DIR, "ABO3_TCF_XGBoost.pkl")

# --- R prediction scripts (all under model folder) ---
PATH_R_PREDICT_QF       = os.path.join(BASE_MODEL_DIR, "predict_ABO4_qf.R")
PATH_R_PREDICT_ABO4_TCF = os.path.join(BASE_MODEL_DIR, "predict_ABO4_tcf.R")
# PATH_R_PREDICT_ABO3_TCF = os.path.join(BASE_MODEL_DIR, "predict_ABO3_tcf.R") # deprecated
