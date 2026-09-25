import os
import subprocess
import tempfile
import pandas as pd
import importlib

# Force-reload config to get the latest settings
import config
importlib.reload(config)

from plugins.base import BasePredictor, BaseFeatureCalculator, ModelMetadata, ModelType

metadata = ModelMetadata(
    name="ABO4 Quality Factor Predictor",
    version="1.0.0",
    author="Built-in",
    description="Model for predicting ABO4 quality factor (Qxf)",
    supported_structures=["ABO4"],
    model_type=ModelType.PYTHON,
    dependencies=["sklearn"]
)

class ABO4QFFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    
    def calculate(self, structure, base_features: dict) -> dict:
        from features import FeatureCalculator
        from data_loader import DataLoader
        
        loader = DataLoader()
        calculator = FeatureCalculator(loader)
        
        z_factor = structure.composition.num_atoms / 6.0
        vm = structure.volume / z_factor
        p = base_features.get("P_MLR_pv", 0) * vm
        
        valences = calculator.get_valences_ordered(structure)
        cations_en = [calculator.get_zhang_en_ordered(s.specie.symbol, valences[i]) 
                      for i, s in enumerate(structure) if s.specie.symbol != "O"]
        en = sum(cations_en) / len(cations_en) if cations_en else 0
        
        return {"vm": vm, "p": p, "en": en}
    
    def get_required_base_features(self) -> list:
        return ["P_MLR_pv"]

class ABO4QFPredictor(BasePredictor):
    metadata = metadata
    _script_path = os.path.join(config.BASE_MODEL_DIR, "predict_ABO4_qf.R")
    _model_path = os.path.join(config.BASE_MODEL_DIR, "ABO4_Qf.rds")
    
    def predict(self, features: dict) -> dict:
        if not os.path.exists(self._script_path):
            print(f"[!] R Script not found")
            return {"qxf_mean": None, "qxf_dev": None}
        
        if not os.path.exists(self._model_path):
            print(f"[!] R Model not found")
            return {"qxf_mean": None, "qxf_dev": None}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', dir='.') as tmp:
            pd.DataFrame([features]).to_csv(tmp.name, index=False)
            input_path = os.path.abspath(tmp.name)
        
        output_path = input_path.replace(".csv", "_out.csv")
        
        try:
            r_exec = os.path.abspath(config.R_SCRIPT_EXEC)
            script_path = os.path.abspath(self._script_path)
            cmd = f'"{r_exec}" "{script_path}" "{input_path}" "{output_path}" "{self._model_path}"'
            print(f"   [*] Running QF with R: {r_exec}")
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60, cwd=config.BASE_DIR)
            
            if result.returncode != 0:
                msg = (result.stderr + result.stdout).decode('utf-8', errors='ignore')
                print(f"[!] R Error: {msg[:300]}")
                return {"qxf_mean": None, "qxf_dev": None}
            
            if os.path.exists(output_path):
                res_df = pd.read_csv(output_path)
                mean_val = float(res_df.iloc[0]['mean'])
                dev_val = float(res_df.iloc[0]['sd'])
                print(f"   [+] Qxf: {mean_val:.2f} ± {dev_val:.2f}")
                return {"qxf_mean": mean_val, "qxf_dev": dev_val}
        except Exception as e:
            print(f"[!] Qxf Error: {e}")
        finally:
            for f in [input_path, output_path]:
                if os.path.exists(f): 
                    try: os.remove(f)
                    except: pass
        
        return {"qxf_mean": None, "qxf_dev": None}
    
    def get_required_features(self) -> list:
        return ["vm", "p", "en"]

from plugins.registry import register_plugin, Plugin

register_plugin(Plugin(
    id="abo4_qf_predictor",
    structure_type="ABO4",
    feature_calculator=ABO4QFFeatureCalculator(),
    predictor=ABO4QFPredictor()
))

print("[*] ABO4 Qf predictor loaded")
