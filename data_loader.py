import os
import pandas as pd
import config

class DataLoader:
    def __init__(self):
        # print(f"📂 Loading Data Tables...")
        self.mlr_df = self._load_csv_clean(config.PATH_MLR_TABLE)
        self.ml_df  = self._load_csv_clean(config.PATH_ML_TABLE)
        
        self.zhang_en_dict = self._load_zhang_en(config.PATH_ZHANG_EN)
        self.bond_len_dict = self._load_bond_length(config.PATH_BOND_LENGTH)
        self.bond_val_dict = self._load_bond_valence(config.PATH_BOND_VALENCE)
        self.bond_cov_dict = self._load_bond_covalency(config.PATH_BOND_COVALENCY)

    def _load_csv_clean(self, path):
        if not os.path.exists(path): return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def _load_zhang_en(self, path):
        if not os.path.exists(path): return {}
        en_map = {}
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                try: en_map[(str(row['element']).strip(), int(float(row['state'])))] = float(row['en'])
                except: continue
            return en_map
        except: return {}

    def _load_bond_length(self, path):
        if not os.path.exists(path): return {}
        res = {}
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns] 
            for _, row in df.iterrows():
                try: res[(str(row['element']).strip(), int(float(row['valence'])))] = {'R1': float(row['r1']), 'N': float(row['n'])}
                except: continue
            return res
        except: return {}

    def _load_bond_valence(self, path):
        if not os.path.exists(path): return {}
        res = {}
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns] 
            for _, row in df.iterrows():
                try: res[(str(row['element']).strip(), int(float(row['valence'])))] = float(row['o'])
                except: continue
            return res
        except: return {}

    def _load_bond_covalency(self, path):
        if not os.path.exists(path): return {}
        res = {}
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                try: res[str(row['element']).strip()] = {'a': float(row['a']), 'M': float(row['m'])}
                except: continue
            return res
        except: return {}