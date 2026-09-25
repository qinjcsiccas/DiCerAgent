import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import importlib

# Force-reload config to get the latest settings
import config
importlib.reload(config)

from plugins.base import BasePredictor, BaseFeatureCalculator, ModelMetadata, ModelType

metadata = ModelMetadata(
    name="ABO4 TCF Predictor",
    version="1.0.0",
    author="Built-in",
    description="R-based model for predicting ABO4 temperature coefficient of frequency (TCF)",
    supported_structures=["ABO4"],
    model_type=ModelType.R_SCRIPT,
    dependencies=["R"]
)

class ABO4TCFFeatureCalculator(BaseFeatureCalculator):
    """ABO4 TCF feature calculator"""
    
    metadata = metadata
    
    def calculate(self, structure, base_features: dict) -> dict:
        from features import FeatureCalculator
        from data_loader import DataLoader
        
        loader = DataLoader()
        calculator = FeatureCalculator(loader)
        
        z_factor = structure.composition.num_atoms / 6.0
        vm = structure.volume / z_factor
        p = base_features.get("P_MLR_pv", 0) * vm
        
        try:
            sp = structure.get_space_group_info()[1]
        except:
            sp = 0
        
        valences = calculator.get_valences_ordered(structure)
        total_bvs = 0.0
        total_cvd = 0.0
        cation_count = 0
        
        cnn = calculator.cnn
        
        for i, site in enumerate(structure):
            if site.specie.symbol == "O":
                continue
            el = site.specie.symbol
            val = int(round(valences[i]))
            cation_count += 1
            
            r0 = loader.bond_val_dict.get((el, val), 0)
            if r0 == 0:
                candidates = [v for (e, v), r in loader.bond_val_dict.items() if e == el]
                if candidates:
                    r0 = loader.bond_val_dict[(el, candidates[0])]
            
            if r0 > 0:
                neighbors = structure.get_neighbors(site, r=6.0)
                for entry in neighbors:
                    if hasattr(entry, "nn_distance"):
                        dist = entry.nn_distance
                        neighbor = entry
                    else:
                        neighbor = entry[0]
                        dist = entry[1]
                    if neighbor.specie.symbol == "O":
                        total_bvs += np.exp((r0 - dist) / 0.37)
            
            nn_info = cnn.get_nn_info(structure, i)
            bonding_dists = [site.distance(structure[n['site_index']], jimage=n['image']) 
                             for n in nn_info if structure[n['site_index']].specie.symbol == 'O']
            bl_params = loader.bond_len_dict.get((el, val))
            bc_params = loader.bond_cov_dict.get(el)
            
            if bl_params and bc_params and bonding_dists:
                R_avg = np.mean(bonding_dists)
                if R_avg > 0:
                    S = (R_avg / bl_params['R1']) ** (-bl_params['N'])
                    fc = bc_params['a'] * (S ** bc_params['M'])
                    if S > 0:
                        total_cvd += (fc / S)
        
        cation_count = cation_count or 1
        return {
            "sp": sp,
            "pm": p,
            "cvd": total_cvd / cation_count,
            "bvs": total_bvs / cation_count
        }
    
    def get_required_base_features(self) -> list:
        return ["P_MLR_pv"]

class ABO4TCFPredictor(BasePredictor):
    """ABO4 TCF predictor"""
    
    metadata = metadata
    _script_path = os.path.join(config.BASE_MODEL_DIR, "predict_ABO4_tcf.R")
    _model_path = os.path.join(config.BASE_MODEL_DIR, "ABO4_tcf.rds")
    
    def predict(self, features: dict) -> dict:
        if not os.path.exists(self._script_path):
            print(f"[!] R Script not found: {self._script_path}")
            return {"tcf_mean": None, "tcf_dev": None}
        
        if not os.path.exists(self._model_path):
            print(f"[!] R Model not found: {self._model_path}")
            return {"tcf_mean": None, "tcf_dev": None}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', dir='.') as tmp:
            pd.DataFrame([features]).to_csv(tmp.name, index=False)
            input_path = os.path.abspath(tmp.name)
        
        output_path = input_path.replace(".csv", "_out.csv")
        
        try:
            r_exec = os.path.abspath(config.R_SCRIPT_EXEC)
            script_path = os.path.abspath(self._script_path)
            cmd = f'"{r_exec}" "{script_path}" "{input_path}" "{output_path}" "{self._model_path}"'
            print(f"   [*] Running TCF with R: {r_exec}")
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                timeout=60,
                cwd=config.BASE_DIR
            )
            
            if result.returncode != 0:
                msg = (result.stderr + result.stdout).decode('utf-8', errors='ignore')
                print(f"[!] R Error: {msg[:300]}")
                return {"tcf_mean": None, "tcf_dev": None}
            
            if os.path.exists(output_path):
                res_df = pd.read_csv(output_path)
                mean_val = float(res_df.iloc[0]['mean'])
                sd_val = float(res_df.iloc[0]['sd'])
                print(f"   [+] TCF prediction: {mean_val} ± {sd_val}")
                return {"tcf_mean": mean_val, "tcf_dev": sd_val}
        except Exception as e:
            print(f"[!] TCF Prediction Error: {e}")
        finally:
            for f in [input_path, output_path]:
                if os.path.exists(f): 
                    try: os.remove(f)
                    except: pass
        
        return {"tcf_mean": None, "tcf_dev": None}
    
    def get_required_features(self) -> list:
        return ["sp", "pm", "cvd", "bvs"]

from plugins.registry import register_plugin, Plugin

register_plugin(Plugin(
    id="abo4_tcf_predictor",
    structure_type="ABO4",
    feature_calculator=ABO4TCFFeatureCalculator(),
    predictor=ABO4TCFPredictor()
))
