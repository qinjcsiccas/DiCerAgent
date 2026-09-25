import os
import re
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict

from plugins.base import BaseFeatureCalculator, BasePredictor, ModelMetadata, ModelType, Plugin
from plugins.registry import register_plugin

# ============ CONSTANT TABLES (auto-deployed) ============
_CONST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HEC_ThermalConductivity_constants')
_MAGPIE_CSV = os.path.join(_CONST_DIR, 'magpie_element_properties.csv')

# Load magpie table at module level
_MAGPIE_TABLE = pd.read_csv(_MAGPIE_CSV)
_MAGPIE_TABLE['Symbol'] = _MAGPIE_TABLE['Symbol'].str.strip()
_MAGPIE_TABLE = _MAGPIE_TABLE.set_index('Symbol')

# Fill missing values with column medians (numeric only) - training convention
_numeric_cols = _MAGPIE_TABLE.select_dtypes(include=[np.number]).columns
for col in _numeric_cols:
    _MAGPIE_TABLE[col] = pd.to_numeric(_MAGPIE_TABLE[col], errors='coerce')
    _MAGPIE_TABLE[col] = _MAGPIE_TABLE[col].fillna(_MAGPIE_TABLE[col].median())

# ============ METADATA ============
metadata = ModelMetadata(
    name="HEC_ThermalConductivity",
    version="1.0.0",
    author="DiCerAgent",
    description="High-entropy ceramics thermal conductivity prediction (Cornell 2023)",
    supported_structures=["ABO3"],
    model_type=ModelType.PYTHON
)

# ============ FEATURE DEFINITIONS ============
FINAL_FEATURES = [
    'temperature', 'NComp',
    'mean_Number', 'maxdiff_Number', 'dev_Number', 'min_Number', 'most_Number',
    'mean_MendeleevNumber', 'maxdiff_MendeleevNumber', 'dev_MendeleevNumber',
    'max_MendeleevNumber', 'most_MendeleevNumber',
    'maxdiff_MeltingT', 'dev_MeltingT', 'max_MeltingT',
    'maxdiff_Column', 'dev_Column',
    'maxdiff_Row', 'dev_Row',
    'mean_Electronegativity',
    'maxdiff_NsValence', 'dev_NpValence',
    'maxdiff_NdValence', 'dev_NdValence', 'max_NdValence',
    'mean_NfValence', 'most_NfValence',
    'mean_NValance', 'maxdiff_NValance', 'min_NValance', 'most_NValance',
    'mean_NpUnfilled', 'maxdiff_NpUnfilled', 'most_NpUnfilled',
    'mean_NdUnfilled', 'maxdiff_NdUnfilled', 'dev_NdUnfilled',
    'mean_NfUnfilled',
    'mean_NUnfilled', 'maxdiff_NUnfilled', 'dev_NUnfilled', 'min_NUnfilled',
    'mean_GSvolume_pa', 'maxdiff_GSvolume_pa', 'dev_GSvolume_pa',
    'min_GSvolume_pa', 'most_GSvolume_pa',
    'most_GSbandgap',
    'max_SpaceGroupNumber',
    'CanFormIonic',
]

# Magpie property columns used for stats
_MAGPIE_PROPS = [
    'Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row',
    'CovalentRadius', 'Electronegativity', 'NsValence', 'NpValence', 'NdValence',
    'NfValence', 'NValance', 'NsUnfilled', 'NpUnfilled', 'NdUnfilled',
    'NfUnfilled', 'NUnfilled', 'GSvolume_pa', 'GSbandgap', 'GSmagmom',
    'SpaceGroupNumber'
]

# ============ FORMULA PARSING (replicate original) ============
def _expand_brackets(formula):
    """Expand simple bracket blocks: (A0.2B0.2)2C1O3 -> A0.4B0.4C1O3."""
    m = re.search(r'\(([^()]*)\)([\d.]+)?', formula)
    if not m:
        return formula
    inner, mult = m.group(1), m.group(2)
    mult = float(mult) if mult else 1.0
    parts = re.findall(r'([A-Z][a-z]?)([\d.]+)?', inner)
    expanded = ''
    for ele, num in parts:
        n = float(num) if num else 1.0
        n *= mult
        expanded += f"{ele}{n:g}"
    return formula[:m.start()] + expanded + formula[m.end():]


def decompose_formula(formula):
    """Return (element_list, mole_coefficient_list). Supports bracket formulas and spaces."""
    formula = str(formula).replace(' ', '')
    if '(' in formula:
        formula = _expand_brackets(formula)
    namelist, numlist = [], []
    ccomps = formula
    while len(ccomps) != 0:
        stemp = ccomps[1:]
        if len(stemp) == 0:
            namelist.append(ccomps)
            numlist.append(1.0)
            break
        it = 0
        matched = False
        for st in stemp:
            it += 1
            if st.isupper():
                im = 0
                for mt in stemp[:it]:
                    im += 1
                    if mt.isdigit():
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:it]))
                        ccomps = ccomps[it:]
                        matched = True
                        break
                    elif im == len(stemp[:it]):
                        namelist.append(ccomps[0:im])
                        numlist.append(1.0)
                        ccomps = ccomps[it:]
                        matched = True
                        break
                break
            elif it == len(stemp):
                im = 0
                for mt in stemp:
                    im += 1
                    if mt.isdigit():
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:]))
                        ccomps = ccomps[it + 1:]
                        matched = True
                        break
                    elif im == len(stemp):
                        namelist.append(ccomps)
                        numlist.append(1.0)
                        ccomps = ccomps[it + 1:]
                        matched = True
                        break
                break
        if not matched:
            break
    return namelist, numlist


def magpie_stats(values, weights):
    """magpie six statistics: mean/maxdiff/dev/max/min/most."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    w = w / w.sum()
    mean = np.sum(w * v)
    maxdiff = np.max(v) - np.min(v)
    dev = np.sum(w * np.abs(v - mean))
    vmax = np.max(v)
    vmin = np.min(v)
    wmax = np.max(w)
    vmost = np.mean(v[w == wmax])
    return mean, maxdiff, dev, vmax, vmin, vmost


def _get_element_property(element, prop):
    """Get property value for element from magpie table. Returns NaN if missing."""
    if element not in _MAGPIE_TABLE.index:
        return np.nan
    val = _MAGPIE_TABLE.loc[element, prop]
    return val


def _compute_features_from_formula(formula, temperature):
    """Compute the 50 final features from a chemical formula and temperature."""
    elements, amounts = decompose_formula(formula)
    if not elements:
        raise ValueError(f"Cannot parse formula: {formula}")
    
    # Convert to mole fractions
    amounts = np.array(amounts, dtype=float)
    total = amounts.sum()
    weights = amounts / total
    
    # Build feature dict
    features = OrderedDict()
    features['temperature'] = float(temperature)
    features['NComp'] = len(set(elements))
    
    # Compute magpie stats for each property
    for prop in _MAGPIE_PROPS:
        values = []
        valid_weights = []
        for ele, w in zip(elements, weights):
            val = _get_element_property(ele, prop)
            if pd.isna(val):
                # Skip missing values (use pre-filled median from table)
                continue
            values.append(val)
            valid_weights.append(w)
        
        if len(values) == 0:
            # All elements missing this property - raise explicit error
            raise ValueError(f"Property '{prop}' has no values for any element in {formula}")
        
        mean, maxdiff, dev, vmax, vmin, vmost = magpie_stats(values, valid_weights)
        
        # Only add features that are in FINAL_FEATURES
        prefix = prop
        for stat_name, stat_val in [('mean', mean), ('maxdiff', maxdiff), ('dev', dev), 
                                     ('max', vmax), ('min', vmin), ('most', vmost)]:
            feat_name = f"{stat_name}_{prop}"
            if feat_name in FINAL_FEATURES:
                features[feat_name] = stat_val
    
    # CanFormIonic: replicate magpie OxidationStateGuesser (charge-balanced combo exists)
    can_form_ionic = 0
    if len(elements) == 1:
        can_form_ionic = 0
    else:
        states = []
        ok = True
        for ele in elements:
            ox_states = _MAGPIE_TABLE.loc[ele, 'OxidationStates'] if ele in _MAGPIE_TABLE.index else np.nan
            if pd.isna(ox_states) or not str(ox_states).strip():
                ok = False
                break
            states.append([float(x) for x in str(ox_states).replace(',', ' ').split()])
        if ok:
            import itertools
            found = False
            for combo in itertools.product(*states):
                if abs(np.dot(combo, weights)) < 1E-6:
                    found = True
                    break
            can_form_ionic = 1 if found else 0
    features['CanFormIonic'] = can_form_ionic
    
    # Ensure all FINAL_FEATURES are present
    for feat in FINAL_FEATURES:
        if feat not in features:
            raise ValueError(f"Feature '{feat}' could not be computed for {formula}")
    
    return features


# ============ FEATURE CALCULATOR ============
class FeatureCalculator(BaseFeatureCalculator):
    metadata = metadata
    
    def get_required_base_features(self):
        return []
    
    def get_required_features(self):
        return FINAL_FEATURES
    
    def calculate(self, structure, base_features):
        """Calculate features from a structure object."""
        # Get formula from structure
        formula = structure.composition.reduced_formula
        
        # Try to get original formula with brackets if available
        orig = getattr(structure, '_orig_formula', None)
        if orig is not None and '(' in orig:
            formula = orig
        
        # Get temperature from base_features if provided, else from structure.temperature, else default to 300K
        temperature = 300.0
        if base_features and 'temperature' in base_features:
            temperature = float(base_features['temperature'])
        elif hasattr(structure, 'temperature'):
            temperature = float(structure.temperature)
        
        # Compute features
        features = _compute_features_from_formula(formula, temperature)
        
        # Verify all required features are present
        for feat in FINAL_FEATURES:
            if feat not in features:
                raise ValueError(f"Missing feature '{feat}' in calculation")
        
        return features


# ============ PREDICTOR ============
class Predictor(BasePredictor):
    metadata = metadata
    
    def __init__(self):
        plugin_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_name, 'model.pkl')
        
        self.models = []
        self.feature_cols = FINAL_FEATURES
        self.model_path = model_path
        self.load_error = None
        
        try:
            if not os.path.exists(model_path):
                self.load_error = f"Model file not found: {model_path}"
                return
            
            model_data = joblib.load(model_path)
            
            # Apply sklearn compatibility patch
            self._patch_sklearn_compat(model_data)
            
            # Adapt to actual layout
            if isinstance(model_data, dict):
                if 'models' in model_data:
                    self.models = model_data['models']
                    if 'feature_cols' in model_data:
                        self.feature_cols = model_data['feature_cols']
                else:
                    # Maybe it's a single estimator wrapped in dict
                    self.models = [model_data]
            elif isinstance(model_data, (list, tuple)):
                self.models = list(model_data)
            else:
                # Single estimator
                self.models = [model_data]
            
            if len(self.models) == 0:
                self.load_error = "No models found in model file"
                
        except Exception as e:
            self.load_error = f"Failed to load model: {str(e)}"
    
    def _patch_sklearn_compat(self, model_data):
        """Patch sklearn compatibility for older versions."""
        if isinstance(model_data, dict):
            for v in model_data.values():
                self._patch_sklearn_compat(v)
        elif isinstance(model_data, (list, tuple)):
            for v in model_data:
                self._patch_sklearn_compat(v)
        elif hasattr(model_data, 'estimators_'):
            if not hasattr(model_data, 'monotonic_cst'):
                try:
                    model_data.monotonic_cst = None
                except Exception:
                    pass
            for est in model_data.estimators_:
                self._patch_sklearn_compat(est)
        elif type(model_data).__name__ in ('DecisionTreeRegressor', 'DecisionTreeClassifier',
                                           'ExtraTreeRegressor', 'ExtraTreeClassifier',
                                           'RandomForestRegressor', 'RandomForestClassifier',
                                           'GradientBoostingRegressor', 'GradientBoostingClassifier'):
            if not hasattr(model_data, 'monotonic_cst'):
                try:
                    model_data.monotonic_cst = None
                except Exception:
                    pass
    
    def get_required_features(self):
        return self.feature_cols
    
    def predict(self, features):
        """Predict thermal conductivity from features dict."""
        if self.load_error:
            return {"kappa_mean": None, "kappa_dev": None, "note": self.load_error}
        
        if len(self.models) == 0:
            return {"kappa_mean": None, "kappa_dev": None, "note": "No models loaded"}
        
        try:
            # Build feature vector in correct order
            X = np.array([features.get(col, 0.0) for col in self.feature_cols], dtype=float).reshape(1, -1)
            
            # Check feature count matches model expectation
            expected_n = getattr(self.models[0], 'n_features_in_', len(self.feature_cols))
            if X.shape[1] != expected_n:
                return {"kappa_mean": None, "kappa_dev": None, 
                        "note": f"Feature count mismatch: got {X.shape[1]}, expected {expected_n}"}
            
            # Predict with all models
            predictions = []
            for est in self.models:
                try:
                    pred = est.predict(X)[0]
                    predictions.append(float(pred))
                except Exception as e:
                    return {"kappa_mean": None, "kappa_dev": None, 
                            "note": f"Prediction error: {str(e)}"}
            
            if len(predictions) == 0:
                return {"kappa_mean": None, "kappa_dev": None, "note": "No predictions generated"}
            
            mean_pred = float(np.mean(predictions))
            dev_pred = float(np.std(predictions))
            
            return {"kappa_mean": mean_pred, "kappa_dev": dev_pred}
            
        except Exception as e:
            return {"kappa_mean": None, "kappa_dev": None, "note": f"Prediction failed: {str(e)}"}


# ============ REGISTER PLUGIN ============
register_plugin(Plugin(
    id="HEC_ThermalConductivity",
    structure_type="ABO3",
    feature_calculator=FeatureCalculator(),
    predictor=Predictor(),
    metadata=metadata
))