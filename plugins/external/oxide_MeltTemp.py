import os
import numpy as np
import joblib
import configparser
import pandas as pd
from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin

metadata = ModelMetadata(
    name="Tm_TwoStage_ANN",
    version="1.0.0",
    author="Jincheng Qin et al.",
    description="Two-stage ANN model for melting point prediction of inorganic oxides",
    supported_structures=["general"],
    model_type=ModelType.PYTHON
)

ELEMENT_COLS = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 
                'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Cu', 
                'Zn', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Ag', 
                'Cd', 'Te', 'I', 'Cs', 'Ba', 'W', 'Re', 'Tl', 'Pb']

STAGE2_REQUIRED_FEATS = ['d', 'na', 'fepa']

MP_CACHE_FILE = r'D:\QJC\MWDCsAI_v3\batch_predict\all_properties_cache.csv'

USE_LOCAL_FEPA_ONLY = False

# 全局缓存开关（2026-08-25 治本改造）：
# False = 禁用本地缓存读写，fepa / density 一律真实联网从 Materials Project API 获取。
# 测试/验证阶段必须为 False，以体现插件真实联网能力；True 仅用于离线环境加速调试。
USE_LOCAL_CACHE = False

class TmFeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    
    def __init__(self):
        self._mp_api_key = None
        self._fepa_cache = {}
        self._local_cache_df = None
    
    def _load_local_cache(self):
        """加载本地缓存文件（USE_LOCAL_CACHE=False 时禁用，确保真实联网）"""
        if not USE_LOCAL_CACHE:
            return pd.DataFrame()
        if self._local_cache_df is None:
            if os.path.exists(MP_CACHE_FILE):
                try:
                    self._local_cache_df = pd.read_csv(MP_CACHE_FILE)
                except:
                    self._local_cache_df = pd.DataFrame()
            else:
                self._local_cache_df = pd.DataFrame()
        return self._local_cache_df
    
    def _save_to_local_cache(self, mp_id, data):
        """保存到本地缓存（USE_LOCAL_CACHE=False 时禁用）"""
        if not USE_LOCAL_CACHE:
            return
        if self._local_cache_df is None:
            self._load_local_cache()
        
        new_row = pd.DataFrame([{'mp_id': mp_id, **data}])
        self._local_cache_df = pd.concat([self._local_cache_df, new_row], ignore_index=True)
        self._local_cache_df.to_csv(MP_CACHE_FILE, index=False)
    
    def _get_mp_api_key(self):
        if self._mp_api_key is None:
            import configparser
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.ini')
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config.read_file(f)
                self._mp_api_key = config.get('DEFAULT', 'MP_API_KEY', fallback='')
            except:
                self._mp_api_key = ''
        return self._mp_api_key
    
    def _fetch_fepa_by_mp_id(self, mp_id):
        """通过 MP ID 获取 fepa，优先本地缓存，再调API"""
        if mp_id in self._fepa_cache:
            return self._fepa_cache[mp_id]
        
        df = self._load_local_cache()
        if not df.empty and 'mp_id' in df.columns:
            match = df[df['mp_id'] == mp_id]
            if not match.empty:
                fepa = match.iloc[0].get('formation_energy_per_atom', None)
                if pd.notna(fepa):
                    self._fepa_cache[mp_id] = float(fepa)
                    return self._fepa_cache[mp_id]
        
        if USE_LOCAL_FEPA_ONLY:
            self._fepa_cache[mp_id] = None
            return None
        
        try:
            from mp_api.client import MPRester
            api_key = self._get_mp_api_key()
            if not api_key:
                self._fepa_cache[mp_id] = None
                return None
            
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=[mp_id],
                    fields=["formation_energy_per_atom"]
                )
                if docs:
                    fepa = getattr(docs[0], 'formation_energy_per_atom', None)
                    self._fepa_cache[mp_id] = fepa
                    self._save_to_local_cache(mp_id, {'formation_energy_per_atom': fepa})
                    return fepa
        except:
            pass
        self._fepa_cache[mp_id] = None
        return None
    
    def _fetch_fepa_by_formula(self, composition):
        """通过化学式获取 fepa（兜底方案）"""
        formula = composition.reduced_formula
        if formula in self._fepa_cache:
            return self._fepa_cache[formula]
        
        df = self._load_local_cache()
        if not df.empty and 'formula' in df.columns:
            match = df[df['formula'] == formula]
            if not match.empty:
                fepa = match.iloc[0].get('formation_energy_per_atom', None)
                if pd.notna(fepa):
                    self._fepa_cache[formula] = float(fepa)
                    return self._fepa_cache[formula]
        
        if USE_LOCAL_FEPA_ONLY:
            self._fepa_cache[formula] = None
            return None
        
        try:
            from mp_api.client import MPRester
            api_key = self._get_mp_api_key()
            if not api_key:
                self._fepa_cache[formula] = None
                return None
            
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["formation_energy_per_atom", "energy_above_hull"]
                )
                if docs:
                    doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
                    fepa = getattr(doc, 'formation_energy_per_atom', None)
                    self._fepa_cache[formula] = fepa
                    return fepa
        except:
            pass
        self._fepa_cache[formula] = None
        return None
    
    def _fetch_density_by_formula(self, composition):
        """通过化学式从 Materials Project API 获取密度（真实联网，2026-08-25 治本改造）。

        训练数据 d（密度）特征与 MP 同源；批量验证阶段结构为默认骨架晶格，
        structure.density 不可信，必须由插件真实联网获取 density 覆盖。
        """
        formula = composition.reduced_formula
        try:
            from mp_api.client import MPRester
            api_key = self._get_mp_api_key()
            if not api_key:
                return None
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["density", "formation_energy_per_atom", "energy_above_hull"]
                )
                if not docs:
                    return None
                # 取最稳定候选（energy_above_hull 最小）
                doc = min(docs, key=lambda x: (x.energy_above_hull if getattr(x, 'energy_above_hull', None) is not None else float('inf')))
                dens = getattr(doc, 'density', None)
                return float(dens) if dens is not None else None
        except:
            return None
    
    def calculate(self, structure, base_features, mp_id=None, fepa=None, density=None, api_fetch=False):
        from pymatgen.core import Composition as _Comp

        # 全部原子数类特征一律从 reduced formula 推导（与论文训练列一致），
        # 不依赖输入结构晶胞大小：同一成分可能以 Ca2O2 / Ca4O4 等任意晶胞传入，
        # 若用 structure.composition 的绝对原子数，特征会随晶胞放大而静默改变。
        # 修复记录：2026-08-24，配合转换器 HARD CONSTRAINTS k) 条款。
        red_comp = _Comp(structure.composition.reduced_formula)
        el_dict = red_comp.get_el_amt_dict()

        unsupported = [e for e in el_dict.keys() if e not in ELEMENT_COLS]
        if unsupported:
            return {"_tm_available": False, "_tm_reason": f"不支持元素: {','.join(unsupported)}"}

        features = {}
        for elem in ELEMENT_COLS:
            features[elem] = el_dict.get(elem, 0.0)

        features['na'] = red_comp.num_atoms
        # d 特征来源优先级（2026-08-25 治本改造）：
        #   1) 显式传入 density（真实数据，最优先）
        #   2) api_fetch=True 时从 MP API 真实联网获取（骨架结构场景，体现真实联网能力）
        #   3) structure.density（调用方传入真实结构）
        if density is not None:
            features['d'] = float(density)
        elif api_fetch:
            api_density = self._fetch_density_by_formula(red_comp)
            features['d'] = api_density if api_density is not None else structure.density
        else:
            features['d'] = structure.density
        
        try:
            from pymatgen.analysis.local_env import CrystalNN
            cnn = CrystalNN()
            bonds = cnn.get_bonded_structure(structure)
            bond_lengths = []
            for site in bonds:
                neighbors = bonds.get_connected_sites(site.index)
                for neighbor in neighbors:
                    bond_lengths.append(neighbor.dist)
            if bond_lengths:
                features['blsd'] = float(np.std(bond_lengths))
                features['blnpv'] = len(bond_lengths) / structure.volume
            else:
                features['blsd'] = 0.0
                features['blnpv'] = 0.0
        except:
            features['blsd'] = 0.0
            features['blnpv'] = 0.0
        
        features['epa'] = 0.0
        
        if fepa is not None:
            features['fepa'] = fepa
        elif mp_id:
            features['fepa'] = self._fetch_fepa_by_mp_id(mp_id)
        else:
            features['fepa'] = self._fetch_fepa_by_formula(red_comp)
        
        return features
    
    def get_required_base_features(self):
        return []
    
    def get_feature_cols(self):
        return ELEMENT_COLS + ['na', 'blsd', 'blnpv', 'd', 'epa', 'fepa']

class TmPredictor(BasePredictor):
    metadata = metadata
    
    def __init__(self):
        plugin_file = os.path.abspath(__file__)
        plugin_name = os.path.splitext(os.path.basename(plugin_file))[0]
        parent_dir = os.path.dirname(plugin_file)
        self.model_dir = os.path.join(parent_dir, plugin_name)
        
        self.prop_key = "Tm"
        self.prop_symbol = "Tm"
        self.prop_unit = "K"
        self.models_loaded = False
        self.models_stage1 = []
        self.models_stage2 = []
        self.stage1_cols = ELEMENT_COLS
        self.stage2_cols = ['stage1_pred'] + ['na', 'blsd', 'blnpv', 'd', 'epa', 'fepa']
        self._load_models()
    
    def _load_models(self):
        try:
            model_path = os.path.join(self.model_dir, "two_stage_model.pkl")
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                self.models_stage1 = data.get("models_stage1", [])
                self.models_stage2 = data.get("models_stage2", [])
                self.stage1_cols = data.get("stage1_cols", self.stage1_cols)
                self.stage2_cols = data.get("stage2_cols", self.stage2_cols)
                self.models_loaded = len(self.models_stage1) > 0 and len(self.models_stage2) > 0
        except Exception as e:
            print(f"[!] Load error: {e}")
            self.models_loaded = False
    
    def _run_stage1(self, features):
        stage1_feats = np.array([[features.get(f, 0) for f in self.stage1_cols]])
        preds = []
        for i in range(len(self.models_stage1)):
            pred = self.models_stage1[i].predict(stage1_feats)[0]
            preds.append(pred)
        return float(np.mean(preds)), float(np.std(preds))
    
    def _run_stage2(self, features):
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
        return float(np.mean(preds)), float(np.std(preds))
    
    def _check_stage2_available(self, features):
        missing = []
        for feat in STAGE2_REQUIRED_FEATS:
            val = features.get(feat)
            if val is None or val == 0.0:
                missing.append(feat)
        return missing
    
    def predict(self, features):
        if not self.models_loaded:
            return {f"{self.prop_key}_mean": None, f"{self.prop_key}_dev": None, f"{self.prop_key}_note": "模型未加载"}
        
        if not isinstance(features, dict):
            return {f"{self.prop_key}_mean": None, f"{self.prop_key}_dev": None, f"{self.prop_key}_note": "特征无效"}
        
        tm_available = features.get("_tm_available")
        if tm_available is False:
            reason = features.get("_tm_reason", "未知原因")
            return {f"{self.prop_key}_mean": None, f"{self.prop_key}_dev": None, f"{self.prop_key}_note": reason}
        
        try:
            stage1_mean, stage1_dev = self._run_stage1(features)
            
            missing = self._check_stage2_available(features)
            if missing:
                note = f"Stage1 (缺少: {','.join(missing)})"
                return {f"{self.prop_key}_mean": stage1_mean, f"{self.prop_key}_dev": stage1_dev, f"{self.prop_key}_note": note}
            
            stage2_mean, stage2_dev = self._run_stage2(features)
            return {f"{self.prop_key}_mean": stage2_mean, f"{self.prop_key}_dev": stage2_dev, f"{self.prop_key}_note": "Stage2"}
        
        except Exception as e:
            return {f"{self.prop_key}_mean": None, f"{self.prop_key}_dev": None, f"{self.prop_key}_note": f"错误: {str(e)[:50]}"}
    
    def get_required_features(self):
        return self.stage1_cols + ['na', 'blsd', 'blnpv', 'd', 'epa', 'fepa']

register_plugin(Plugin(
    id="tm_twostage_ann",
    structure_type="general",
    feature_calculator=TmFeatureCalculator(),
    predictor=TmPredictor()
))
