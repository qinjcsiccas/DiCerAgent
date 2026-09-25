from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class ModelType(Enum):
    PYTHON = "python"
    R_SCRIPT = "r_script"
    COMPOUND = "compound"

@dataclass
class PropertyInfo:
    """Property info: name, symbol, unit"""
    key: str  # e.g. "er", "qxf", "tcf", "melt_temp"
    name: str  # full name e.g. "Dielectric Constant"
    symbol: str  # display symbol e.g. "εr", "Qxf", "τf"
    unit: str  # unit e.g. "", "GHz", "ppm/°C", "K"
    
    @property
    def display_symbol(self) -> str:
        return f"{self.symbol}"

PROPERTY_REGISTRY = {
    "er": PropertyInfo(key="er", name="Dielectric Constant", symbol="εr", unit=""),
    "qxf": PropertyInfo(key="qxf", name="Quality Factor", symbol="Qxf", unit="GHz"),
    "qf": PropertyInfo(key="qf", name="Quality Factor", symbol="Qxf", unit="GHz"),  # alias
    "tcf": PropertyInfo(key="tcf", name="Temperature Coefficient of Frequency", symbol="τf", unit="ppm/°C"),
    "tec": PropertyInfo(key="tec", name="Thermal Expansion Coefficient", symbol="α", unit="10^-6 K^-1"),
    "tm": PropertyInfo(key="tm", name="Melting Temperature", symbol="Tm", unit="K"),
    "melt_temp": PropertyInfo(key="melt_temp", name="Melting Temperature", symbol="Tm", unit="K"),
}

def register_property_info(key, symbol, unit, name=""):
    """Dynamically register property info (for external plugins)"""
    PROPERTY_REGISTRY[key] = PropertyInfo(key=key, name=name or key, symbol=symbol, unit=unit)

def get_property_info(key: str) -> PropertyInfo:
    """Get property info, automatically handling aliases and variant suffixes"""
    import re
    base_key = key.replace("_mean", "").replace("_dev", "").lower().strip()
    # Strip variant suffixes such as _abo3 / _abo4
    base_key = re.sub(r"_(abo\d+|other)$", "", base_key)
    return PROPERTY_REGISTRY.get(base_key, PropertyInfo(key=base_key, name=base_key, symbol=base_key.upper(), unit=""))

@dataclass
class ModelMetadata:
    name: str
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    supported_structures: List[str] = field(default_factory=list)
    model_type: ModelType = ModelType.PYTHON
    dependencies: List[str] = field(default_factory=list)
    scope: str = ""

class BaseFeatureCalculator(ABC):
    """Base feature calculator"""
    
    metadata: ModelMetadata
    
    @abstractmethod
    def calculate(self, structure, base_features: Dict[str, float]) -> Dict[str, float]:
        """Compute features and return a feature dict"""
        pass
    
    @abstractmethod
    def get_required_base_features(self) -> List[str]:
        """Required base features (e.g. p_mlr_pv, structure, etc.)"""
        pass

class BasePredictor(ABC):
    """Base predictor"""
    
    metadata: ModelMetadata
    
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Run prediction
        Return format: {"property_mean": float, "property_dev": float, ...}
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Input features required by the model"""
        pass

@dataclass
class Plugin:
    """Complete plugin unit"""
    id: str
    structure_type: str  # e.g. "ABO4", "ABO3", "general"
    feature_calculator: Optional[BaseFeatureCalculator] = None
    predictor: Optional[BasePredictor] = None
    metadata: Optional[ModelMetadata] = None
    enabled: bool = True
    scope: str = ""
    
    def __post_init__(self):
        if self.metadata is None and (self.feature_calculator or self.predictor):
            calc = self.feature_calculator
            pred = self.predictor
            self.metadata = calc.metadata if calc else (pred.metadata if pred else ModelMetadata(name=self.id))
