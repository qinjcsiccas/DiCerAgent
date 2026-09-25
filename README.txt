================================================================
DiCerAgent — AI-Driven Multi-Agent Research Platform for Dielectric Ceramics
================================================================

DiCerAgent is an integrated multi-agent framework for dielectric ceramics research. It combines five specialized agents: ARMPP (forward prediction), HRKE (hybrid knowledge retrieval), MPC (model-to-plugin conversion), IDO (inverse design), and CADO (dialogue orchestration).

---------------------------------------------------------------
1. System Requirements
---------------------------------------------------------------
- Windows 10/11 (or Linux/macOS with equivalent setup)
- Python 3.10+ (developed with Python 3.11)
- R 4.3+ (required for R-based models, e.g. ABO4_tcf.rds)
- Internet connection for API calls (Materials Project, DeepSeek, Hugging Face)
- About 2 GB free disk space (models + data)

---------------------------------------------------------------
2. One-Time Environment Setup
---------------------------------------------------------------
(Recommended) Create a conda environment:

    conda create -n MWDCsAI python=3.11 -y
    conda activate MWDCsAI

Install the Python dependencies (adjust for your package manager):

    pip install streamlit pandas numpy scikit-learn pymatgen mp-api joblib reportlab requests openai torch faiss-cpu mace-torch ase matminer PyPDF2 pdfplumber beautifulsoup4 langchain-community langchain-huggingface sentence-transformers stmol py3Dmol

Some packages are large (torch, mace-torch). If you prefer conda:

    conda install -c conda-forge streamlit pandas numpy scikit-learn pymatgen faiss-cpu
    conda install -c pytorch pytorch
    pip install mace-torch mp-api

Install R 4.3+ from https://cran.r-project.org/ (Windows installer) and note the path to Rscript.exe, e.g. C:\Program Files\R\R-4.3.3\bin\Rscript.exe

(Optional) If you use the MatterGen-based external model, install MatterGen and set MATTERGEN_DIR in config.py to its location.

---------------------------------------------------------------
3. Configuration (settings.ini)
---------------------------------------------------------------
Open settings.ini in the project root and fill in:

    R_EXEC_PATH  = full path to Rscript.exe
    MP_API_KEY   = Materials Project API key (https://next-gen.materialsproject.org/api)
    DS_API_KEY   = DeepSeek API key
    HF_TOKEN     = Hugging Face token

Placeholder values starting with YOUR_ must be replaced with your own credentials before first use.

---------------------------------------------------------------
4. Launch
---------------------------------------------------------------
    streamlit run app_unified.py

The browser opens automatically. Tabs:
    ARMPP     - forward prediction of dielectric properties
    HRKE      - hybrid knowledge retrieval (literature + property database)
    CADO      - natural-language dialogue / multi-agent orchestration
    MPC       - model-to-plugin conversion (Folder / Upload / Template / Installed)
    IDO       - inverse design optimizer (requires mace, faiss, pymatgen)
    Settings  - configuration and help
    About     - authors and contact

---------------------------------------------------------------
5. Data & Models
---------------------------------------------------------------
- The bundled database is a 10% random sample (seed 42) of the original dataset for redistribution; CaWO4 is included as a complete case study with all properties, literature, vectors, and CIF files.
- If the large FAISS index is missing, the knowledge engine automatically falls back to the CSV-based retrieval channel.
- external_models/ contains example model-to-plugin conversion cases (e.g. high-entropy ceramic thermal conductivity, oxide melting temperature).

---------------------------------------------------------------
6. Troubleshooting
---------------------------------------------------------------
- "Rscript not found" -> check R_EXEC_PATH in settings.ini and confirm R 4.3+ is installed.
- 401 / invalid key errors -> verify MP_API_KEY, DS_API_KEY, and HF_TOKEN.
- IDO fails to load -> install mace, faiss, and pymatgen in the active environment.
- Stale results or unexpected states -> click the "Clear Cache" button at the bottom of the page.

---------------------------------------------------------------
Contact: qinjccas@gmail.com
================================================================
