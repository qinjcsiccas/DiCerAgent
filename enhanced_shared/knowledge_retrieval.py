"""
Knowledge Retrieval Index Loader
------------------------
Responsible for building, caching, and retrieving the knowledge retrieval FAISS index.
Supports three target properties: er / qxf / tau_f
"""

import os, sys, json, pickle, time, glob
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.config import (
    KR_DATA_DIR, KR_INDEX_PATH, ELEMENTS, PROPERTY_CONFIG,
    K_NEIGHBORS, KR_SEARCH_MULTIPLIER,
)


def formula_to_vector(formula_str):
    """Chemical formula -> element fraction vector"""
    from pymatgen.core import Composition
    vec = np.zeros(len(ELEMENTS), dtype=np.float32)
    try:
        comp = Composition(formula_str)
        total = comp.num_atoms
        if total == 0:
            return vec
        for el, amt in comp.get_el_amt_dict().items():
            if el in ELEMENTS:
                vec[ELEMENTS.index(el)] = amt / total
    except Exception:
        pass
    return vec


def build_kr_index(force_rebuild=False):
    """
    Parse all knowledge retrieval JSON files and build the FAISS index and metadata.
    Cached to kr_index.pkl.
    
    Returns:
        vectors: np.ndarray (N, 118) element fraction vectors
        property_values: dict[str, np.ndarray] value arrays for each property
        meta: list[dict] metadata list
    """
    if not force_rebuild and os.path.exists(KR_INDEX_PATH):
        print("Loading cached knowledge retrieval index...", flush=True)
        with open(KR_INDEX_PATH, "rb") as f:
            vectors, prop_values, meta = pickle.load(f)
        return vectors, prop_values, meta

    print("Parsing knowledge retrieval JSON files to build index...", flush=True)
    files = sorted(glob.glob(os.path.join(KR_DATA_DIR, "*.json")))

    vectors = []
    meta = []
    er_vals, qxf_vals, tauf_vals = [], [], []

    t0 = time.time()
    for fi, fp in enumerate(files):
        if fi % 2000 == 0 and fi > 0:
            print(f"  {fi}/{len(files)} files, {len(vectors)} valid entries", flush=True)

        try:
            if os.path.getsize(fp) < 100:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        md = data.get("material_data")
        if not isinstance(md, list):
            continue

        for entry in md:
            if not isinstance(entry, dict):
                continue

            dp = entry.get("dielectric_properties")
            if not isinstance(dp, dict):
                dp = {}
            er = dp.get("permittivity_er")
            if er is None:
                continue
            try:
                er = float(er)
            except (ValueError, TypeError):
                continue
            if er < 1 or er > 500:
                continue

            mi = entry.get("material_identity")
            if not isinstance(mi, dict):
                mi = {}
            formula = mi.get("normalized_formula") or mi.get("formula_string", "")
            if not formula:
                continue

            vec = formula_to_vector(formula)
            if np.sum(vec) == 0:
                continue

            # Extract processing/phase/microstructure fields (defensive against non-dict types)
            _ps = entry.get("processing_history") or {}
            ps = _ps if isinstance(_ps, dict) else {}
            sint = ps.get("sintering") or {}
            sint = sint if isinstance(sint, dict) else {}
            calc = ps.get("calcination") or {}
            calc = calc if isinstance(calc, dict) else {}
            _pas = entry.get("phase_and_structure") or {}
            pas = _pas if isinstance(_pas, dict) else {}
            lp = pas.get("lattice_parameters") or {}
            lp = lp if isinstance(lp, dict) else {}
            pc = pas.get("phase_constitution") or {}
            pc = pc if isinstance(pc, dict) else {}
            _micro = entry.get("microstructure") or {}
            micro = _micro if isinstance(_micro, dict) else {}
            _cm = entry.get("composition_modification") or {}
            cm = _cm if isinstance(_cm, dict) else {}
            _ctx = entry.get("contextual_metadata") or {}
            ctx = _ctx if isinstance(_ctx, dict) else {}

            qxf = dp.get("quality_factor_Qxf_GHz")
            tauf = dp.get("tau_f_ppm_C")

            entry_meta = {
                "formula": str(formula),
                "er": er,
                "qxf": float(qxf) if qxf is not None else None,
                "tau_f": float(tauf) if tauf is not None else None,
                "sintering_temp_C": sint.get("temperature_C"),
                "sintering_time_h": sint.get("time_h"),
                "sintering_atmosphere": sint.get("atmosphere"),
                "calcination_temp_C": calc.get("temperature_C"),
                "calcination_time_h": calc.get("time_h"),
                "synthesis_route": ps.get("synthesis_route"),
                "raw_materials": ps.get("raw_materials"),
                "forming_method": (lambda f: f.get("method") if isinstance(f, dict) else None)(ps.get("forming") or {}),
                "crystal_system": lp.get("crystal_system"),
                "space_group": lp.get("space_group"),
                "cell_volume_A3": lp.get("cell_volume_A3"),
                "theoretical_density": lp.get("theoretical_density_g_cm3"),
                "is_single_phase": pc.get("is_single_phase"),
                "secondary_phases": pc.get("secondary_phases"),
                "grain_size_um": micro.get("average_grain_size_um"),
                "porosity_pct": micro.get("porosity_percent"),
                "grain_morphology": micro.get("grain_morphology"),
                "dopants": cm.get("dopants"),
                "sintering_additives": cm.get("sintering_additives"),
                "bulk_density": (lambda o: o.get("bulk_density_g_cm3") if isinstance(o, dict) else None)(entry.get("other_properties") or {}),
                "is_optimal": ctx.get("is_optimal"),
                "measurement_frequency_GHz": dp.get("measurement_frequency_GHz"),
            }

            vectors.append(vec)
            meta.append(entry_meta)
            er_vals.append(er)
            qxf_vals.append(entry_meta["qxf"])
            tauf_vals.append(entry_meta["tau_f"])

    vectors = np.array(vectors, dtype=np.float32)
    prop_values = {
        "er": np.array(er_vals, dtype=np.float32),
        "qxf": np.array([v if v is not None else np.nan for v in qxf_vals], dtype=np.float32),
        "tau_f": np.array([v if v is not None else np.nan for v in tauf_vals], dtype=np.float32),
    }

    with open(KR_INDEX_PATH, "wb") as f:
        pickle.dump((vectors, prop_values, meta), f)

    elapsed = (time.time() - t0) / 60
    print(f"  Done: {len(vectors)} entries in {elapsed:.1f} min, cached to {KR_INDEX_PATH}", flush=True)
    return vectors, prop_values, meta


def load_kr_index():
    """Load the knowledge retrieval index (auto-builds cache)"""
    return build_kr_index()


class KnowledgeRetriever:
    """
    Knowledge retrieval neighbor retriever with anti-leakage exclusion.

    Two usage modes:
      - Test/ablation: pass exclude_set = test-set formula set to prevent data leakage.
      - Agent real application: no exclude_set is passed; exact-match experimental data
        is read directly. This is an advantage of real application, not data leakage.
    """

    def __init__(self, vectors, meta, property_values, target_property="er"):
        from sklearn.neighbors import NearestNeighbors

        field = PROPERTY_CONFIG.get(target_property, {}).get("kr_field", target_property)
        if field == "er":
            self.prop_values = property_values["er"]
        elif field == "qxf":
            self.prop_values = property_values["qxf"]
        elif field == "tau_f":
            self.prop_values = property_values["tau_f"]
        else:
            self.prop_values = property_values.get(field, property_values["er"])

        self.vectors = vectors
        self.meta = meta
        self.target_property = target_property
        self.k = K_NEIGHBORS

        self.nn = NearestNeighbors(
            n_neighbors=min(self.k * KR_SEARCH_MULTIPLIER, len(vectors)),
            metric="cosine"
        )
        self.nn.fit(vectors.astype(np.float32))

    def retrieve(self, formula, exclude_set=None, k=None):
        """Retrieve top-k neighbors, excluding formulas in exclude_set"""
        if k is None:
            k = self.k

        vec = formula_to_vector(formula).reshape(1, -1)
        if np.sum(vec) == 0:
            return []

        dists, idxs = self.nn.kneighbors(vec)
        results = []
        for d, idx in zip(dists[0], idxs[0]):
            m = self.meta[idx]
            if exclude_set and m["formula"] in exclude_set:
                continue
            neighbor = {
                "formula": m["formula"],
                "value": float(self.prop_values[idx]),
                "distance": float(d),
                "similarity": float(1.0 - d / 2.0),
                # Processing conditions
                "synthesis_route": m.get("synthesis_route"),
                "raw_materials": m.get("raw_materials"),
                "forming_method": m.get("forming_method"),
                "sintering_temp_C": m.get("sintering_temp_C"),
                "sintering_time_h": m.get("sintering_time_h"),
                "sintering_atmosphere": m.get("sintering_atmosphere"),
                "sintering_additives": m.get("sintering_additives"),
                "calcination_temp_C": m.get("calcination_temp_C"),
                "calcination_time_h": m.get("calcination_time_h"),
                # Structure and phases
                "crystal_system": m.get("crystal_system"),
                "space_group": m.get("space_group"),
                "cell_volume_A3": m.get("cell_volume_A3"),
                "is_single_phase": m.get("is_single_phase"),
                "secondary_phases": m.get("secondary_phases"),
                # Microstructure
                "grain_size_um": m.get("grain_size_um"),
                "grain_morphology": m.get("grain_morphology"),
                "porosity_pct": m.get("porosity_pct"),
                # Composition modification
                "dopants": m.get("dopants"),
                # Density
                "theoretical_density": m.get("theoretical_density"),
                "bulk_density": m.get("bulk_density"),
                # Test conditions
                "freq_GHz": m.get("measurement_frequency_GHz"),
            }
            neighbor = {k: v for k, v in neighbor.items() if v is not None}
            results.append(neighbor)
            if len(results) >= k:
                break

        return results
