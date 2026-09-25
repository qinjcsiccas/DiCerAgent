# -*- coding: utf-8 -*-
"""
Inverse Design Module — Tiered Search Pipeline v2
  Step 1: Multi-objective scoring of known-property materials
  Step 2: Analog recommendation of unknown-property materials
  Step 3: Extract chemical prior → MatterGen constrained generation
  Step 4: Forward prediction validation + LLM explanation & strategy
Embedded as a tab within app_unified.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import faiss
import json
import os
import re
import glob
import shutil
import requests
from pathlib import Path
import tempfile
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from mace.calculators import MACECalculator
import config

# ===================================================================
#  Cached resource loaders
# ===================================================================


@st.cache_resource
def _load_all():
    index = faiss.read_index(config.PATH_FAISS_INDEX)
    material_ids = np.load(config.PATH_MATERIAL_IDS, allow_pickle=True).tolist()
    metadata = pd.read_csv(config.PATH_MATERIAL_PROPERTIES, encoding="utf-8-sig")
    metadata = metadata.set_index("material_id").reindex(material_ids).reset_index()

    # Pre-compute normalized formula and element set for formula search
    def _reduce(f):
        try:      return Composition(f).reduced_formula
        except:   return str(f)
    def _elems(f):
        try:      return "|".join(sorted(Composition(f).get_el_amt_dict().keys()))
        except:   return ""

    metadata["_reduced_formula"] = metadata["formula"].apply(_reduce)
    metadata["_elem_str"] = metadata["formula"].apply(_elems)

    data_dict = np.load(config.PATH_FINAL_CRYSTAL_VECTORS, allow_pickle=True).item()
    for mid in material_ids:
        if mid not in data_dict:
            data_dict[mid] = np.zeros(374)

    return index, material_ids, metadata, data_dict


@st.cache_resource
def _load_mace():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(
        model_paths=config.PATH_MACE_MODEL, device=device, default_dtype="float32"
    )
    adaptor = AseAtomsAdaptor()
    return calc, adaptor


# ===================================================================
#  CIF -> Feature vector
# ===================================================================


def _get_vector_from_cif(cif_path, mace_calc, adaptor, feat_dim):
    struct = Structure.from_file(cif_path)
    atoms = adaptor.get_atoms(struct)
    descriptors = mace_calc.get_descriptors(atoms)

    if isinstance(descriptors, list):
        node_feats_tensor = descriptors[0]
    elif isinstance(descriptors, torch.Tensor):
        node_feats_tensor = descriptors
    elif isinstance(descriptors, np.ndarray):
        node_feats_np = descriptors
        n_atoms = atoms.get_global_number_of_atoms()
        if node_feats_np.ndim == 1:
            node_feats_np = node_feats_np.reshape(n_atoms, -1)
        elif node_feats_np.ndim > 2:
            node_feats_np = node_feats_np.reshape(n_atoms, -1)
        if n_atoms > 1000:
            indices = np.random.choice(n_atoms, min(n_atoms, 1000), replace=False)
            node_feats_np = node_feats_np[indices]
        v_mace = np.mean(node_feats_np, axis=0)
        comp_dim = feat_dim - len(v_mace)
        comp = np.zeros(comp_dim)
        vec = np.concatenate([v_mace, comp]).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
    else:
        node_feats_tensor = descriptors

    node_feats_np = node_feats_tensor.detach().cpu().numpy()
    n_atoms = atoms.get_global_number_of_atoms()
    if node_feats_np.ndim == 1:
        node_feats_np = node_feats_np.reshape(n_atoms, -1)
    elif node_feats_np.ndim > 2:
        node_feats_np = node_feats_np.reshape(n_atoms, -1)
    if n_atoms > 1000:
        indices = np.random.choice(n_atoms, min(n_atoms, 1000), replace=False)
        node_feats_np = node_feats_np[indices]
    v_mace = np.mean(node_feats_np, axis=0)
    comp_dim = feat_dim - len(v_mace)
    comp = np.zeros(comp_dim)
    vec = np.concatenate([v_mace, comp]).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def _get_vector_from_structure(structure, mace_calc, adaptor, feat_dim):
    atoms = adaptor.get_atoms(structure)
    descriptors = mace_calc.get_descriptors(atoms)

    if isinstance(descriptors, list):
        node_feats_tensor = descriptors[0]
    elif isinstance(descriptors, torch.Tensor):
        node_feats_tensor = descriptors
    elif isinstance(descriptors, np.ndarray):
        node_feats_np = descriptors
        n_atoms = atoms.get_global_number_of_atoms()
        if node_feats_np.ndim == 1:
            node_feats_np = node_feats_np.reshape(n_atoms, -1)
        elif node_feats_np.ndim > 2:
            node_feats_np = node_feats_np.reshape(n_atoms, -1)
        if n_atoms > 1000:
            indices = np.random.choice(n_atoms, min(n_atoms, 1000), replace=False)
            node_feats_np = node_feats_np[indices]
        v_mace = np.mean(node_feats_np, axis=0)
        comp_dim = feat_dim - len(v_mace)
        comp = np.zeros(comp_dim)
        vec = np.concatenate([v_mace, comp]).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
    else:
        node_feats_tensor = descriptors

    node_feats_np = node_feats_tensor.detach().cpu().numpy()
    n_atoms = atoms.get_global_number_of_atoms()
    if node_feats_np.ndim == 1:
        node_feats_np = node_feats_np.reshape(n_atoms, -1)
    elif node_feats_np.ndim > 2:
        node_feats_np = node_feats_np.reshape(n_atoms, -1)
    if n_atoms > 1000:
        indices = np.random.choice(n_atoms, min(n_atoms, 1000), replace=False)
        node_feats_np = node_feats_np[indices]
    v_mace = np.mean(node_feats_np, axis=0)
    comp_dim = feat_dim - len(v_mace)
    comp = np.zeros(comp_dim)
    vec = np.concatenate([v_mace, comp]).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


# ===================================================================
#  MatterGen generation
# ===================================================================


@st.cache_resource
def _load_mattergen_generator(checkpoint_key="base"):
    """Load the MatterGen generator.

    checkpoint_key:
      - "base": default composition-trained checkpoint (unchanged behavior);
      - "sg":   space_group-conditioned checkpoint (for structure-guided runs).
    """
    import sys, os, warnings

    warnings.filterwarnings("ignore")
    mg_dir = config.MATTERGEN_DIR
    if mg_dir not in sys.path:
        sys.path.insert(0, mg_dir)
    old_cwd = os.getcwd()
    os.chdir(mg_dir)
    try:
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator

        if checkpoint_key == "sg":
            ckpt_path = config.PATH_MATTERGEN_CHECKPOINT_SG
        else:
            ckpt_path = config.PATH_MATTERGEN_CHECKPOINT
        ckpt = MatterGenCheckpointInfo(
            model_path=ckpt_path,
            load_epoch="last",
            config_overrides=[
                '++lightning_module.diffusion_module.model.element_mask_func={_target_:"mattergen.denoiser.mask_disallowed_elements",_partial_:True}'
            ],
        )
        gen = CrystalGenerator(checkpoint_info=ckpt, record_trajectories=False)
        gen.prepare()
        return gen
    finally:
        os.chdir(old_cwd)


# ===================================================================
#  Anion policy (UI-configurable element pre-restriction)
# ===================================================================

# Policy keys shared by the Streamlit UI and the programmatic entry points.
ANION_POLICY_DEFAULT = "default"
ANION_POLICY_FULL = "full"
ANION_POLICY_CUSTOM = "custom"

# Default ceramic anion base — identical to the historical hard-coded set.
_ANION_DEFAULT = ("O", "F", "N", "S", "Cl")

# "All anions" option: exactly the anion set the downstream anion bias
# (_inject_anion_boost) and the ceramic filter (_filter_chemically_plausible)
# already recognise, so the whitelist never contradicts the post-filter.
_ANION_FULL = ("O", "F", "N", "S", "Cl", "P", "Se", "Br", "Te", "I", "As")


def _resolve_anion_base(policy, custom=None):
    """Resolve the anion base (allowed-element seed) from a UI/programmatic policy.

    policy:
      - "default" : O/F/N/S/Cl (historical behaviour, unchanged);
      - "full"    : all anions the pipeline supports (see _ANION_FULL);
      - "custom"  : the caller-supplied subset of _ANION_FULL.
    Falls back to the default set when the policy is unknown or the custom
    selection is empty, so the effective element mask is never empty.
    """
    if policy == ANION_POLICY_FULL:
        return list(_ANION_FULL)
    if policy == ANION_POLICY_CUSTOM and custom:
        picked = [el for el in custom if el in _ANION_FULL]
        if picked:
            return picked
    return list(_ANION_DEFAULT)


def _inject_anion_boost(gen):
    """Wrap the denoiser's element_mask_func to boost anion logits
    when the model predicts an all-metallic composition at any step.

    This runs INSIDE the denoiser forward pass at every diffusion timestep,
    steering the model away from intermetallics without wasting compute on
    full denoising runs that would ultimately be discarded.
    """
    import torch

    denoiser = gen.model.diffusion_module.model  # GemNetTDenoiser
    original_mask = denoiser.element_mask_func  # typically mask_disallowed_elements

    # Anion atomic numbers (1-based, zero-based logit index = atomic number)
    ANION_Z = {8, 9, 7, 16, 17, 15, 34, 35, 52, 53, 33}  # O F N S Cl P Se Br Te I As
    BOOST = 5.0  # logit boost for anion classes

    # Pre-allocate anion set tensor (created lazily on first call, moved to GPU)
    _anion_tensor = None

    def _ceramic_biased_mask(logits, x, batch_idx):
        nonlocal _anion_tensor

        # Step 1: apply original element restriction
        logits = original_mask(logits=logits, x=x, batch_idx=batch_idx)

        # Step 2: check each sample's predicted x_0; boost anions if all-metallic
        if _anion_tensor is None or _anion_tensor.device != logits.device:
            _anion_tensor = torch.tensor(
                sorted(ANION_Z), dtype=torch.long, device=logits.device
            )

        pred_z = torch.argmax(logits, dim=-1)  # [total_atoms]

        if batch_idx is None:
            # single sample
            has_anion = torch.isin(pred_z, _anion_tensor).any()
            if not has_anion:
                logits[:, _anion_tensor] += BOOST
        else:
            for b in torch.unique(batch_idx):
                sample_mask = batch_idx == b
                has_anion = torch.isin(pred_z[sample_mask], _anion_tensor).any()
                if not has_anion:
                    # Boolean indexing returns a copy, so use integer indexing
                    rows = sample_mask.nonzero(as_tuple=True)[0]
                    logits[rows.unsqueeze(1), _anion_tensor.unsqueeze(0)] += BOOST

        return logits

    denoiser.element_mask_func = _ceramic_biased_mask


def _sg_to_number(sg):
    """Resolve a space-group identifier (int SG number or intl. symbol) to its 1-230 number.

    Returns None if unresolvable.
    """
    import numpy as _np

    if sg is None:
        return None
    if isinstance(sg, _np.integer):
        sg = int(sg)
    if isinstance(sg, int):
        return sg if 1 <= sg <= 230 else None
    # International symbol → number (pymatgen)
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        n = SpaceGroup(str(sg).strip().replace(" ", "")).int_number
        return n if 1 <= n <= 230 else None
    except Exception:
        return None


# SG-number ranges per crystal system (standard IUCr ordering)
_SG_CRYSTAL_SYSTEM = {
    "triclinic": (1, 2),
    "monoclinic": (3, 15),
    "orthorhombic": (16, 74),
    "tetragonal": (75, 142),
    "trigonal": (143, 167),
    "hexagonal": (168, 194),
    "cubic": (195, 230),
}


def _sg_relax(sg, mode="same"):
    """Build the space-group constraint set from a seed space group.

    mode:
      - "same":    [sg_number] — strict, condition on the exact seed SG;
      - "system":  all SG numbers in the seed's crystal system (recommended relax);
      - "none":    [] — no structural constraint.
    Returns:
      - None          → no constraint (caller keeps default behavior);
      - [int, ...]    → one or more valid 1-230 SG numbers.
    """
    if not mode or mode == "none":
        return None
    n = _sg_to_number(sg)
    if n is None:
        return None
    if mode == "same":
        return [n]
    # crystal-system relaxation via IUCr SG-number ranges (no pymatgen needed)
    for lo, hi in _SG_CRYSTAL_SYSTEM.values():
        if lo <= n <= hi:
            return list(range(lo, hi + 1))
    return [n]


# ===================================================================
#  Generation planning (auto batch size) — single source of truth
#  Batch size is no longer a user-facing parameter: it is derived here
#  from the requested maximum number of structures, and BOTH the UI
#  estimate and _run_mattergen consume the very same plan, so that what
#  is displayed is exactly what gets generated.
# ===================================================================

# (upper bound of num_structures, batch size)
_AUTO_BATCH_TIERS = ((4, 4), (16, 8), (64, 16), (10**9, 32))


def _auto_batch_size(num_structures):
    """Pick a GPU-friendly batch size from the requested structure count.

    Tiers: <=4 -> 4, <=16 -> 8, <=64 -> 16, >64 -> 32.
    Larger requests use larger batches (better throughput per unit time);
    small requests stay small to avoid wasting memory on tiny jobs.
    """
    n = max(1, int(num_structures or 1))
    for upper, bs in _AUTO_BATCH_TIERS:
        if n <= upper:
            return bs
    return _AUTO_BATCH_TIERS[-1][1]


def _plan_generation(num_structures, batch_size=None):
    """Return the one and only generation plan for a request.

    batch_size=None means "auto" (derived from num_structures); an explicit
    value is honoured as-is (legacy callers such as the chat path).

    Keys:
      num_structures     : requested cap (final kept count never exceeds it)
      batch_size         : batch size actually used
      num_batches        : diffusion batches actually run
      gen_target         : 1.5x safety target fed to the diffusion sampler
      expected_generated : batch_size * num_batches
      expected_kept      : min(num_structures, expected_generated)
      est_minutes_low / est_minutes_high : rough wall-clock estimate
    """
    n = max(1, int(num_structures or 1))
    bs = max(1, int(batch_size)) if batch_size else _auto_batch_size(n)
    gen_target = max(1, int(round(n * 1.5)))
    num_batches = max(1, -(-gen_target // bs))  # ceil division
    generated = bs * num_batches
    return {
        "num_structures": n,
        "batch_size": bs,
        "num_batches": num_batches,
        "gen_target": gen_target,
        "expected_generated": generated,
        "expected_kept": min(n, generated),
        # per-batch cost calibrated on measured runs (~3.5-8 min/batch)
        "est_minutes_low": round(num_batches * 3.5),
        "est_minutes_high": round(num_batches * 8),
    }


def _run_mattergen(
    num_structures=8,
    batch_size=None,
    output_subdir="gen",
    allowed_elements=None,
    anion_policy=ANION_POLICY_DEFAULT,
    anion_whitelist=None,
    space_group_cond=None,
    diffusion_guidance_factor=0.0,
):
    import os, sys

    mg_dir = config.MATTERGEN_DIR
    sys.path.insert(0, mg_dir)
    # space-group conditioning requires the SG checkpoint; otherwise use default
    use_sg_ckpt = space_group_cond not in (None, []) and space_group_cond is not False
    gen = _load_mattergen_generator(checkpoint_key="sg" if use_sg_ckpt else "base")
    out_dir = os.path.join(config.PATH_MATTERGEN_OUTPUT, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Auto batch planning (same helper the UI uses) ──
    _batch_is_auto = batch_size in (None, 0, "")
    plan = _plan_generation(num_structures, batch_size=batch_size)
    batch_size = plan["batch_size"]  # lock the batch size for all SG routes
    print(
        f"[_run_mattergen] requested={plan['num_structures']}, "
        f"batch_size={batch_size} ({'auto' if _batch_is_auto else 'explicit'}), "
        f"num_batches={plan['num_batches']}, "
        f"expected_generated={plan['expected_generated']}, "
        f"expected_kept={plan['expected_kept']}"
    )

    # ── Inject anion-biased element_mask_func ──
    # This runs inside the denoiser at EVERY forward pass.
    # When the model's x_0 prediction is all-metallic, it boosts
    # anion logits to steer the diffusion away from intermetallics.
    _inject_anion_boost(gen)

    # ── Pre-generation element restriction ──
    old_selected = None
    if allowed_elements and len(allowed_elements) > 0:
        _atomic_numbers = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            "Sc": 21,
            "Ti": 22,
            "V": 23,
            "Cr": 24,
            "Mn": 25,
            "Fe": 26,
            "Co": 27,
            "Ni": 28,
            "Cu": 29,
            "Zn": 30,
            "Ga": 31,
            "Ge": 32,
            "As": 33,
            "Se": 34,
            "Br": 35,
            "Kr": 36,
            "Rb": 37,
            "Sr": 38,
            "Y": 39,
            "Zr": 40,
            "Nb": 41,
            "Mo": 42,
            "Tc": 43,
            "Ru": 44,
            "Rh": 45,
            "Pd": 46,
            "Ag": 47,
            "Cd": 48,
            "In": 49,
            "Sn": 50,
            "Sb": 51,
            "Te": 52,
            "I": 53,
            "Xe": 54,
            "Cs": 55,
            "Ba": 56,
            "La": 57,
            "Ce": 58,
            "Pr": 59,
            "Nd": 60,
            "Pm": 61,
            "Sm": 62,
            "Eu": 63,
            "Gd": 64,
            "Tb": 65,
            "Dy": 66,
            "Ho": 67,
            "Er": 68,
            "Tm": 69,
            "Yb": 70,
            "Lu": 71,
            "Hf": 72,
            "Ta": 73,
            "W": 74,
            "Re": 75,
            "Os": 76,
            "Ir": 77,
            "Pt": 78,
            "Au": 79,
            "Hg": 80,
            "Tl": 81,
            "Pb": 82,
            "Bi": 83,
            "Po": 84,
            "At": 85,
            "Rn": 86,
        }
        # Base ceramic elements: anions only — cations are strictly restricted
        # by allowed_elements (from composition prior). This prevents cations
        # that weren't in the prior from slipping through the element mask.
        # The anion base is configurable (UI "Anion restriction"): the default
        # keeps the historical O/F/N/S/Cl set unchanged.
        _base_ceramic = set(_resolve_anion_base(anion_policy, anion_whitelist))
        merged = _base_ceramic | set(allowed_elements)
        allowed_z = sorted(
            _atomic_numbers[el] for el in merged if el in _atomic_numbers
        )
        if not allowed_z:
            allowed_z = sorted(
                _atomic_numbers[el] for el in _base_ceramic if el in _atomic_numbers
            )
        try:
            from mattergen import denoiser as mg_denoiser

            old_selected = list(mg_denoiser.SELECTED_ATOMIC_NUMBERS)
            mg_denoiser.SELECTED_ATOMIC_NUMBERS = allowed_z
        except Exception:
            pass

    old_cwd = os.getcwd()
    os.chdir(mg_dir)
    try:
        # ── Space-group conditioning plan ──
        # int  → single exact SG (whole run conditioned on it);
        # list → multi-route generation: split num_structures across each SG.
        if isinstance(space_group_cond, int):
            sg_routes = [space_group_cond]
        elif space_group_cond is None or space_group_cond is False or space_group_cond == []:
            sg_routes = [None]
        else:
            sg_routes = [int(s) for s in space_group_cond]

        route_counts: dict = {}
        if len(sg_routes) == 1:
            route_counts[sg_routes[0]] = num_structures
        else:
            base = num_structures // len(sg_routes)
            rem = num_structures % len(sg_routes)
            for i, sg in enumerate(sg_routes):
                route_counts[sg] = base + (1 if i < rem else 0)

        # anion boost already injected above, shared across all routes
        structures = []
        for sg, n in route_counts.items():
            if n <= 0:
                continue
            if sg is not None:
                # condition on the exact space group (1-230)
                gen.properties_to_condition_on = {"space_group": sg}
                gen.diffusion_guidance_factor = float(diffusion_guidance_factor)
            else:
                # default unconditional generation (existing behavior)
                gen.properties_to_condition_on = None
                gen.diffusion_guidance_factor = 0.0
            # Anion boosting eliminates need for 3x over-generation.
            # Small multiplier (1.5x) as safety for edge cases.
            # num_batches is derived from the shared planner (ceil), so the
            # sampler always covers the 1.5x target for this route.
            route_plan = _plan_generation(n, batch_size=batch_size)
            gen.batch_size = route_plan["batch_size"]
            gen.num_batches = route_plan["num_batches"]
            try:
                structures.extend(gen.generate(output_dir=out_dir))
            except Exception as e:
                print(f"[_run_mattergen] route sg={sg} failed: {e}")
                continue
    finally:
        os.chdir(old_cwd)
        if old_selected is not None:
            try:
                import mattergen.denoiser as mg_denoiser

                mg_denoiser.SELECTED_ATOMIC_NUMBERS = old_selected
            except Exception:
                pass

    # ── Ceramic filter: keep only structures with anions (O/F/N/…) ──
    n_before = len(structures)
    structures = _filter_chemically_plausible(structures)
    n_after = len(structures)
    structures = structures[:num_structures]
    if n_before > n_after:
        print(
            f"[_run_mattergen] {n_before} generated, "
            f"{n_before - n_after} non-ceramic filtered out, "
            f"{min(n_after, num_structures)} kept"
        )

    return structures, out_dir


# ===================================================================
#  Ceramics filter
# ===================================================================

_NON_CERAMIC_ELEMENTS = {
    # Noble gases
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "Rn",
    # Radioactive / actinides (often excluded from practical ceramics)
    "Tc",
    "Po",
    "At",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
}

# ===================================================================
#  Chemical plausibility filter
# ===================================================================

_ANIONS = {"O", "F", "N", "S", "Cl", "Br", "I", "Se", "Te", "P", "As"}
# Elements that form intermetallics when combined without anions
_METALLIC_ONLY = {
    "Al",
    "Ga",
    "In",
    "Sn",
    "Pb",
    "Bi",
    "Sb",
    "Ge",
    "Zn",
    "Cd",
    "Hg",
    "Cu",
    "Ag",
    "Au",
    "Ni",
    "Pd",
    "Pt",
    "Co",
    "Rh",
    "Ir",
    "Fe",
    "Ru",
    "Os",
    "Mn",
    "Re",
    "Cr",
    "Mo",
    "W",
    "V",
    "Nb",
    "Ta",
    "Ti",
    "Zr",
    "Hf",
}


def _filter_chemically_plausible(structures):
    """Keep ceramics: must contain anion (O/F/N/S/...), exclude pure intermetallics."""
    kept = []
    for s in structures:
        comp = s.composition
        elements = set(str(el) for el in comp.elements)
        if len(elements) < 2:
            continue
        has_anion = bool(elements & _ANIONS)
        if not has_anion:
            continue
        non_anions = elements - _ANIONS
        # All-non-anion-elements are metallic → likely intermetallic, skip
        if all(e in _METALLIC_ONLY for e in non_anions) and len(non_anions) >= 2:
            continue
        if any(e in _NON_CERAMIC_ELEMENTS for e in non_anions):
            continue
        kept.append(s)
    return kept


# ===================================================================


def _search_by_vector(
    qvec, top_k, index, material_ids, metadata, db_filter=None, min_sim=0.0
):
    scores, indices = index.search(qvec, top_k * 3)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        mid = material_ids[idx]
        if db_filter == "MP" and not mid.startswith("MP_"):
            continue
        if db_filter == "COD" and not mid.startswith("COD_"):
            continue
        if score < min_sim:
            continue
        row = metadata.iloc[idx]
        results.append(
            {
                "material_id": mid,
                "formula": row["formula"],
                "spacegroup": row["spacegroup"],
                "similarity": float(score),
                "pred_tm": row.get("pred_tm", None),
                "pred_er": row.get("pred_er", None),
                "pred_qf": row.get("pred_qf", None),
                "pred_tcf": row.get("pred_tcf", None),
                "cif_path": row["cif_path"],
            }
        )
        if len(results) >= top_k:
            break
    return results


def _normalize_fractional(formula: str) -> str:
    """Preprocess user input: convert e.g. Ba(Zn1/3Ta2/3)O3 → Ba(Zn0.333Ta0.667)O3."""
    # Pattern: element followed by a fraction like 1/3, 2/3, 1/2, 3/4, etc.
    def _replace(m):
        elem = m.group(1)
        num = float(m.group(2))
        denom = float(m.group(3))
        return f"{elem}{num/denom:.4g}"
    return re.sub(r'([A-Z][a-z]?)(\d+)/(\d+)', _replace, formula)


def _search_by_formula(query, metadata, index=None, material_ids=None, feature_dict=None):
    """Three-tier formula search.

    L1: Normalized exact match (via pymatgen Composition.reduced_formula).
    L2: Element set match (identical elements, any stoichiometry).
    L3: Structural similarity fallback via FAISS (requires index/material_ids/feature_dict).
    """
    # Parse user input — normalize fractional formulas first
    query = _normalize_fractional(query)
    try:
        q_comp = Composition(query)
        q_reduced = q_comp.reduced_formula
        q_elem_str = "|".join(sorted(q_comp.get_el_amt_dict().keys()))
    except Exception:
        # Fallback: substring search on raw formula if pymatgen can't parse
        matches = metadata[metadata["formula"].str.contains(query, case=False, na=False)]
        return (
            matches[["material_id", "formula", "spacegroup", "cif_path",
                     "pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
            .head(100).to_dict("records")
        )

    output_cols = ["material_id", "formula", "spacegroup", "cif_path",
                   "pred_tm", "pred_er", "pred_qf", "pred_tcf"]

    # ---- L1: Normalized exact match ----
    l1 = metadata[metadata["_reduced_formula"] == q_reduced]
    if len(l1) > 0:
        return l1[output_cols].head(100).to_dict("records")

    # ---- L2: Element set match (same elements, any stoichiometry) ----
    l2 = metadata[metadata["_elem_str"] == q_elem_str]
    if len(l2) > 0:
        return l2[output_cols].head(100).to_dict("records")

    # ---- L3: Structural similarity fallback ----
    if index is not None and material_ids is not None and feature_dict is not None:
        q_elems_set = set(q_comp.get_el_amt_dict().keys())
        # Find materials sharing >= 2 elements with query
        def _overlap(s):
            try:     return len(set(s.split("|")) & q_elems_set) >= 2
            except:  return False
        l3_base = metadata[metadata["_elem_str"].apply(_overlap)]
        if len(l3_base) > 0:
            base_vectors = []
            for mid in l3_base["material_id"]:
                if mid in feature_dict:
                    base_vectors.append(feature_dict[mid])
            if base_vectors:
                avg_vec = np.mean(base_vectors, axis=0, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(avg_vec)
                D, I = index.search(avg_vec, 20)
                results = []
                seen_fid = set()
                for rank, idx in enumerate(I[0]):
                    mid = material_ids[idx]
                    if mid in seen_fid:
                        continue
                    seen_fid.add(mid)
                    row = metadata[metadata["material_id"] == mid]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    results.append({
                        "material_id": row["material_id"],
                        "formula": row["formula"],
                        "spacegroup": row["spacegroup"],
                        "cif_path": row["cif_path"],
                        "similarity": float(D[0][rank]),
                        "pred_tm": row.get("pred_tm", None),
                        "pred_er": row.get("pred_er", None),
                        "pred_qf": row.get("pred_qf", None),
                        "pred_tcf": row.get("pred_tcf", None),
                    })
                return results

    return []


# ===================================================================
#  Pipeline Step 1: Multi-Objective Scoring
# ===================================================================


_OP_SET = ("eq", "gte", "lte", "range", "none")


def _normalize_target_specs(targets):
    """Unify target specifications:
    - dict with explicit op (legacy eq/gte/lte/range) -> parsed by its original semantics (backward compatible)
    - dict with only lo/hi bounds -> inferred from the bounds:
        both filled and unequal = range; both filled and equal = eq; only lo = gte; only hi = lte; both empty = none (unconstrained)
    - scalar → eq
    """
    specs = []
    for t in targets:
        if isinstance(t, dict):
            op = str(t.get("op", ""))
            if op in _OP_SET:
                spec = {"op": op}
                if op == "range":
                    lo = t.get("lo", t.get("value", None))
                    hi = t.get("hi", t.get("value", None))
                    spec["lo"] = float(lo) if lo not in (None, "") else None
                    spec["hi"] = float(hi) if hi not in (None, "") else None
                else:
                    spec["value"] = float(t.get("value", 0.0))
                specs.append(spec)
                continue
            lo = t.get("lo", None)
            hi = t.get("hi", None)
            lo_f = float(lo) if lo not in (None, "") else None
            hi_f = float(hi) if hi not in (None, "") else None
            if lo_f is None and hi_f is None:
                specs.append({"op": "none"})
            elif lo_f is not None and hi_f is not None:
                if abs(lo_f - hi_f) < 1e-12:
                    specs.append({"op": "eq", "value": lo_f})
                else:
                    specs.append({"op": "range", "lo": lo_f, "hi": hi_f})
            elif lo_f is not None:
                specs.append({"op": "gte", "value": lo_f})
            else:
                specs.append({"op": "lte", "value": hi_f})
        else:
            specs.append({"op": "eq", "value": float(t)})
    return specs


def _spec_center(spec):
    if spec["op"] == "range":
        lo, hi = spec.get("lo"), spec.get("hi")
        if lo is None and hi is None:
            return 0.0
        if lo is None:
            return hi
        if hi is None:
            return lo
        return (lo + hi) / 2.0
    if spec["op"] == "none":
        return 0.0
    return spec.get("value", 0.0)


def _spec_distance(value, spec, std):
    """Constraint distance (divided by std):
    eq    -> |value - target|
    gte   -> max(0, target - value)
    lte   -> max(0, value - target)
    range -> 0 inside the interval, otherwise distance to the nearest endpoint
    """
    std = std if std and std > 0 else 1.0
    op = spec["op"]
    try:
        if op == "none":
            return 0.0
        if op == "gte":
            return max(0.0, spec["value"] - value) / std
        if op == "lte":
            return max(0.0, value - spec["value"]) / std
        if op == "range":
            lo = spec.get("lo")
            hi = spec.get("hi")
            if lo is None:
                lo = float("-inf")
            if hi is None:
                hi = float("inf")
            if lo <= value <= hi:
                return 0.0
            if value < lo:
                return (lo - value) / std
            return (value - hi) / std
        return abs(value - spec["value"]) / std
    except (TypeError, ValueError):
        return 1.0


def _spec_text(spec, attr=""):
    """Human-readable constraint description (for display/logging)."""
    op = spec.get("op", "eq")
    if op == "none":
        return f"{attr} unconstrained"
    if op == "range":
        lo, hi = spec.get("lo"), spec.get("hi")
        lo_s = "-inf" if lo is None else f"{lo:g}"
        hi_s = "+inf" if hi is None else f"{hi:g}"
        return f"{attr} [{lo_s}, {hi_s}]"
    sym = {"eq": "=", "gte": ">=", "lte": "<="}.get(op, "=")
    return f"{attr} {sym} {spec['value']:g}"


def _step1_score(metadata, targets, weights, prop_stds, top_k=10):
    prop_cols = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
    specs = _normalize_target_specs(targets)
    df_scored = metadata[metadata[prop_cols].notna().any(axis=1)].copy()
    if df_scored.empty:
        return pd.DataFrame(), None

    total_distance = np.zeros(len(df_scored))
    for col, spec, std, w in zip(prop_cols, specs, prop_stds, weights):
        if w > 0:
            mask = df_scored[col].notna()
            # Present part: contribution by constraint distance
            if mask.any():
                vals = df_scored.loc[mask, col].values
                d = np.array(
                    [_spec_distance(v, spec, std) for v in vals], dtype=float
                )
                total_distance[mask.values] += w * d
            # Missing part: penalize by weight (w * 1.0, where 1.0 is treated as a 1-sigma offset)
            missing_mask = ~mask.values
            if missing_mask.any():
                total_distance[missing_mask] += w * 1.0

    df_scored["score"] = 1.0 / (1.0 + total_distance)
    df_scored = df_scored.sort_values("score", ascending=False)
    df_top = df_scored.head(top_k).copy()
    return df_top, df_scored


# ===================================================================
#  Pipeline Step 2: Unknown-Property Analog Recommendation
# ===================================================================


def _step2_recommend_unknown(metadata, feature_dict, material_ids, seed_ids, top_k=10):
    seed_vecs = np.array(
        [feature_dict[mid] for mid in seed_ids if mid in feature_dict], dtype=np.float32
    )
    if len(seed_vecs) == 0:
        return pd.DataFrame()

    avg_vec = np.mean(seed_vecs, axis=0).reshape(1, -1)
    faiss.normalize_L2(avg_vec)

    prop_cols = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
    unknown_mask = metadata[prop_cols].isna().all(axis=1)
    df_unknown = metadata[unknown_mask].copy()
    if df_unknown.empty:
        return pd.DataFrame()

    uids = df_unknown["material_id"].tolist()
    uvecs = np.array(
        [feature_dict.get(mid, np.zeros(374)) for mid in uids], dtype=np.float32
    )
    faiss.normalize_L2(uvecs)
    sims = np.dot(uvecs, avg_vec.T).flatten()
    df_unknown["similarity"] = sims
    df_unknown = df_unknown.sort_values("similarity", ascending=False)
    return df_unknown.head(top_k).copy()


# ===================================================================
#  Pipeline Step 3: Extract Chemical Prior from Candidate Pool
# ===================================================================

# Extended element name mappings for composition analysis
_ALL_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


def _extract_composition_prior(metadata, known_df, unknown_df, n_sample=15):
    formulas = []

    if known_df is not None and not known_df.empty:
        formulas.extend(known_df["formula"].head(n_sample).tolist())
    if unknown_df is not None and not unknown_df.empty:
        formulas.extend(
            unknown_df["formula"].head(max(0, n_sample - len(formulas))).tolist()
        )

    if not formulas:
        return None, None

    element_counts = {}
    total_atoms = 0

    for formula in formulas[:n_sample]:
        if not isinstance(formula, str) or not formula.strip():
            continue
        pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
        matches = re.findall(pattern, formula)
        for elem, count_str in matches:
            if elem not in _ALL_ELEMENTS:
                continue
            count = float(count_str) if count_str else 1.0
            element_counts[elem] = element_counts.get(elem, 0) + count
            total_atoms += count

    if not element_counts:
        return None, None

    top = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    total = sum(cnt for _, cnt in top)
    target_comp = {elem: round(cnt / total, 4) for elem, cnt in top}

    summary = {
        "top_elements": [e[0] for e in top[:8]],
        "n_elements": len(target_comp),
    }

    return [target_comp], summary


# ===================================================================
#  LLM helpers: Explanation & Strategy
# ===================================================================


def _call_deepseek(messages, temperature=0.3, max_tokens=1500):
    if not config.DS_API_KEY:
        return "*LLM not available: API key not configured*"

    headers = {
        "Authorization": f"Bearer {config.DS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DS_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    _base = config.DEEPSEEK_API_URL.rstrip("/")
    _api_url = _base + "/v1/chat/completions" if "/chat/completions" not in _base else _base
    try:
        r = requests.post(
            _api_url, headers=headers, json=payload, timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "*LLM call failed*"


# ===================================================================
#  Local rule-based fallback: Explanation
# ===================================================================

# Rough mapping of common microwave-dielectric ceramic structure types to typical element patterns
# (used for heuristic element-substitution suggestions when no LLM is available)
_DIELECTRIC_STRUCTURE_HINTS = {
    "scheelite": {
        "template": "ABO₄ scheelite",
        "tau_f_neg": "Use larger ionic-radius elements on the A site (e.g. Nd³⁺ for La³⁺, Bi³⁺ for Nd³⁺) to raise τf; co-doping Ti⁴⁺ on the B site can also compensate negative τf",
        "er_low": "Introduce high-polarizability ions on the A site (Bi³⁺, Pb²⁺); partially substitute Ti⁴⁺ for W⁶⁺/Mo⁶⁺ on the B site",
        "qf_low": "Reduce A-site disorder and avoid multi-phase coexistence; raise sintering temperature for densification",
    },
    "perovskite": {
        "template": "ABO₃ perovskite",
        "tau_f_neg": "Increase A-site ionic radius (Ba²⁺ for Sr²⁺, or Ca²⁺ for Ba²⁺ depending on the case); Ta⁵⁺ for Nb⁵⁺ on the B site can fine-tune τf toward positive",
        "er_low": "Introduce high-polarizability ions on the B site (Nb⁵⁺, Ta⁵⁺); substitute Bi³⁺ for La³⁺ on the A site",
        "qf_low": "Order the B site (1:1 ordered complex perovskite); avoid A-site vacancies",
    },
    "spinel": {
        "template": "AB₂O₄ spinel",
        "tau_f_neg": "Use larger ions on the A site (Mg²⁺ for Zn²⁺); Li⁺ on the tetrahedral site can partially help",
        "er_low": "Use high-polarizability ions on the B site (Ti⁴⁺ for part of Al³⁺)",
        "qf_low": "Reduce antisite defects and optimize sintering conditions",
    },
    "wollastonite": {
        "template": "Silicate / wollastonite type",
        "tau_f_neg": "Introduce a TiO₂ secondary phase to compensate negative τf; rare-earth doping (e.g. Nd₂O₃)",
        "er_low": "Introduce TiO₂ or CaTiO₃ to raise the dielectric constant",
        "qf_low": "Optimize glass-phase content and sintering temperature",
    },
    "rutile": {
        "template": "TiO₂ rutile type",
        "tau_f_neg": "Add positive-tf materials (e.g. CaTiO₃, SrTiO₃) to form a composite ceramic for compensation",
        "er_low": "This type is naturally high in εr; if low, check the relative density",
        "qf_low": "Add sintering aids (e.g. ZnO-B₂O₃) to reduce loss",
    },
}

# Generic mapping of elements to substitutable same-class elements (A-site / B-site / R rare earth)
_ELEMENT_SUBSTITUTION_A = {
    "Ca": ["Sr", "Ba", "Mg", "Zn"],
    "Sr": ["Ba", "Ca", "Pb"],
    "Ba": ["Sr", "Ca"],
    "La": ["Nd", "Sm", "Gd", "Bi"],
    "Nd": ["La", "Sm", "Gd", "Bi"],
    "Sm": ["Nd", "La", "Gd"],
    "Gd": ["Nd", "Sm", "La"],
    "Bi": ["La", "Nd", "Pb"],
    "Pb": ["Ba", "Sr", "Bi"],
    "Na": ["K", "Li", "Ag"],
    "K": ["Na", "Rb", "Ag"],
    "Li": ["Na", "K", "Mg"],
    "Mg": ["Zn", "Ca", "Ni", "Co"],
    "Zn": ["Mg", "Co", "Mn"],
    "Mn": ["Zn", "Mg", "Co"],
}

_ELEMENT_SUBSTITUTION_B = {
    "W": ["Mo", "Te"],
    "Mo": ["W", "Te"],
    "Te": ["W", "Mo"],
    "Nb": ["Ta", "Sb", "V"],
    "Ta": ["Nb", "Sb", "V"],
    "Ti": ["Zr", "Sn", "Hf"],
    "Zr": ["Ti", "Sn", "Hf"],
    "Sn": ["Ti", "Zr"],
    "Si": ["Ge", "Sn"],
    "Ge": ["Si", "Sn"],
    "Al": ["Ga", "Sc", "Cr"],
    "Ga": ["Al", "Sc"],
    "Sc": ["Al", "Ga", "Y", "In"],
}

# Rare earth elements
_RARE_EARTH = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}


def _categorize_elements(formula_str):
    """Roughly classify the elements in a formula into A-site / B-site / rare-earth categories. Returns {category: [element_symbols]}."""
    import re
    elems = re.findall(r'[A-Z][a-z]?', formula_str)
    a_site, b_site, rare, others = [], [], [], []
    for e in elems:
        if e in _RARE_EARTH:
            rare.append(e)
        elif e in _ELEMENT_SUBSTITUTION_A:
            a_site.append(e)
        elif e in _ELEMENT_SUBSTITUTION_B:
            b_site.append(e)
        else:
            others.append(e)
    # If the B-site is empty and others is non-empty, some others may count as B-site (e.g. high-valence small ions such as P, V, Sb)
    if not b_site and others:
        b_site, others = others, []
    return {"A": a_site, "B": b_site, "R": rare, "other": others}


def _structure_label(spacegroup_str):
    """Infer a rough structure-family label from a spacegroup string."""
    if not isinstance(spacegroup_str, str):
        return "unknown"
    sg = spacegroup_str.strip().upper()
    if sg.startswith("I41") or sg.startswith("I4₁"):
        return "scheelite"
    if sg.startswith("Pm-3m") or sg.startswith("Pm3"):
        return "perovskite"
    if sg.startswith("Fd-3m") or sg.startswith("Fd3"):
        return "spinel"
    if sg.startswith("P2₁/c") or sg.startswith("P21/c") or sg.startswith("P2/c"):
        return "wollastonite"
    if sg.startswith("P4₂/mnm") or sg.startswith("P42/mnm"):
        return "rutile"
    return "unknown"


def _explain_candidates_local(candidates_df, targets, weights, prop_stds, source_label, n_explain=5):
    """Local rule-based fallback: data-driven candidate summary (no LLM analysis)."""
    if candidates_df is None or candidates_df.empty:
        return ""

    targets = [_spec_center(s) for s in _normalize_target_specs(targets)]
    top = candidates_df.head(n_explain).copy()
    prop_keys = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
    prop_names = {
        "pred_tm": "Tm (°C)",
        "pred_er": "εr",
        "pred_qf": "Q×f (GHz)",
        "pred_tcf": "τf (ppm/°C)",
    }
    target_dict = dict(zip(prop_keys, targets))

    lines = []

    # ── 1. Property deviation table ──
    lines.append("#### Property Deviation Analysis")
    lines.append(f"| Candidate | Score/Sim | {' | '.join(prop_names.values())} |")
    lines.append(f"|-----------|-----------|{'|'.join(['------'] * 4)}|")

    for _, row in top.iterrows():
        formula = row.get("formula", "?")
        score_val = row.get("score") or row.get("similarity") or 0
        cells = []
        for pk in prop_keys:
            val = row.get(pk)
            target = target_dict[pk]
            if pd.notna(val) and val is not None and target != 0:
                dev_pct = (val - target) / abs(target) * 100
                sign = "+" if dev_pct > 0 else ""
                cells.append(f"{val:.2f} ({sign}{dev_pct:.1f}%)")
            elif pd.notna(val) and val is not None:
                cells.append(f"{val:.2f}")
            else:
                cells.append("N/A")
        lines.append(f"| **{formula}** | {score_val:.4f} | {' | '.join(cells)} |")

    lines.append("")

    # ── 2. Structure type distribution ──
    sgs = [r.get("spacegroup", "N/A") for _, r in top.iterrows()]
    labels = [_structure_label(s) for s in sgs]
    from collections import Counter
    label_counts = Counter(labels)
    if "unknown" in label_counts:
        del label_counts["unknown"]
    if label_counts:
        parts = [f"{lbl}({cnt}/{len(top)})" for lbl, cnt in label_counts.most_common(3)]
        lines.append("#### Structure Type Distribution")
        lines.append(f"Top-{len(top)} candidates: {', '.join(parts)}.")
        lines.append("")

    return "\n".join(lines)


# ===================================================================
#  Local rule-based fallback: Strategy
# ===================================================================

def _strategy_suggestion_local(
    targets, weights, prop_stds,
    known_count, unknown_count, generated_count, generated_pass_count, total_known,
    gen_df=None,
):
    """Local rule-based fallback: simplified guidance (no LLM analysis)."""
    targets = _normalize_target_specs(targets)
    lines = []

    prop_names = ["Tm (°C)", "εr", "Q×f (GHz)", "τf (ppm/°C)"]
    prop_keys = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
    pass_pct = (generated_pass_count / generated_count * 100) if generated_count > 0 else 0

    # ── Next steps ──
    lines.append("#### Suggested Next Steps")
    if generated_pass_count > 0:
        lines.append(f"1. **Experimental validation**: Select the top candidate from {generated_pass_count} passing structures for synthesis and characterization")
        lines.append(f"2. **DFT refinement**: Perform first-principles calculations on top candidates to verify thermodynamic stability and electronic structure")
        lines.append(f"3. **Combinatorial optimization**: Explore nearby compositions (A/B-site substitution) in chemical space using the top candidate as seed")
    elif known_count > 0:
        lines.append(f"1. No structures passed generation, but {known_count} candidates are available from the known-property library; use these for experimental validation")
        lines.append(f"2. Adjust MatterGen constraints (relax element mask, increase sampling steps) and re-run Step 3")
    else:
        lines.append(f"1. Review target feasibility: {_spec_text(targets[1], 'εr')}, {_spec_text(targets[2], 'Q×f')}, {_spec_text(targets[3], 'τf')}, {_spec_text(targets[0], 'Tm')}")
        lines.append(f"2. Verify that the feature database and MACE calculator loaded correctly")

    return "\n".join(lines)


def _explain_candidates(
    candidates_df, targets, weights, prop_stds, source_label, n_explain=5,
    gen_df=None,
):
    if candidates_df is None or candidates_df.empty:
        return ""

    targets = _normalize_target_specs(targets)
    top = candidates_df.head(n_explain)
    property_map = {
        "pred_tm": "Melting Point Tm (°C)",
        "pred_er": "Dielectric Constant εr",
        "pred_qf": "Quality Factor Q×f (GHz)",
        "pred_tcf": "Temperature Coefficient of Frequency τf (ppm/°C)",
    }
    target_map = dict(zip(["pred_tm", "pred_er", "pred_qf", "pred_tcf"], targets))
    weight_map = dict(zip(["pred_tm", "pred_er", "pred_qf", "pred_tcf"], weights))
    std_map = dict(zip(["pred_tm", "pred_er", "pred_qf", "pred_tcf"], prop_stds))

    rows = []
    for _, row in top.iterrows():
        parts = [f"**{row['formula']}** (Space Group: {row.get('spacegroup', 'N/A')})"]
        if source_label == "known" and "score" in row:
            parts.append(f"  Score: {row['score']:.4f}")
        elif source_label == "unknown" and "similarity" in row:
            parts.append(f"  Similarity: {row['similarity']:.4f}")
        for col_key in ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]:
            val = row.get(col_key)
            if pd.notna(val) and val is not None:
                parts.append(f"  {property_map[col_key]}: {val:.2f}")
        parts.append("")
        rows.append("\n".join(parts))

    candidate_text = "\n".join(rows)

    # Build generated candidates section if available
    gen_text = ""
    if gen_df is not None and not gen_df.empty:
        gen_top = gen_df.head(min(5, len(gen_df)))
        gen_rows = []
        for _, row in gen_top.iterrows():
            formula = row.get("Formula", row.get("formula", "Unknown"))
            parts = [f"**{formula}** (MatterGen generated)"]
            if "score" in row:
                parts.append(f"  Score: {row['score']:.4f}")
            if "nearest_known" in row and pd.notna(row.get("nearest_known")):
                parts.append(f"  Nearest known: {row['nearest_known']} (sim={row.get('nearest_sim', 'N/A'):.4f})" if isinstance(row.get('nearest_sim'), (int, float)) else f"  Nearest known: {row['nearest_known']}")
            for col_key in ["pred_er", "pred_qf", "pred_tcf", "pred_tm"]:
                val = row.get(col_key)
                if pd.notna(val) and val is not None:
                    parts.append(f"  {property_map.get(col_key, col_key)}: {val:.2f}")
            parts.append("")
            gen_rows.append("\n".join(parts))
        gen_text = "\n".join(gen_rows)

    target_text = "\n".join(
        [
            f"- {_spec_text(targets[1], 'εr')}, weight={weights[1]:.2f}, population σ={prop_stds[1]:.1f}",
            f"- {_spec_text(targets[2], 'Q×f')}, weight={weights[2]:.2f}, population σ={prop_stds[2]:.0f}",
            f"- {_spec_text(targets[3], 'τf')}, weight={weights[3]:.2f}, population σ={prop_stds[3]:.1f}",
            f"- {_spec_text(targets[0], 'Tm')}, weight={weights[0]:.2f}, population σ={prop_stds[0]:.0f}",
        ]
    )

    prompt = f"""You are a microwave dielectric ceramics expert. Below are reverse-design candidates from two sources.

User targets & weights:
{target_text}

--- DATABASE-SCREENED CANDIDATES ({source_label}) ---
{candidate_text}
"""

    if gen_text:
        prompt += f"""
--- MATTERGEN-GENERATED CANDIDATES ---
{gen_text}
"""

    prompt += """
Please provide a scientific interpretation of the reverse-design results:

1. **Database candidates**: Which candidates best match each property target and why — examine their crystal structure types (e.g. scheelite, perovskite, spinel) and identify common structural or compositional patterns among the top-ranked materials.
2. **Generated candidates**: Analyze the novelty of MatterGen-generated structures — what elements or structure types appear that extend beyond the database candidates, and how their predicted property profiles compare.
3. **Cross-source insights**: What does the gap between database and generated candidates reveal about the explored chemical space? Identify composition regions the database may have underrepresented.

Use specific numbers from the data above. Write 3-4 paragraphs in a natural narrative style.
IMPORTANT: Output ONLY the analysis text, no preamble, no markdown headers."""

    messages = [
        {
            "role": "system",
            "content": "You are a materials scientist specializing in microwave dielectric ceramics. Provide concise, data-driven analysis covering both known and generated materials.",
        },
        {"role": "user", "content": prompt},
    ]
    result = _call_deepseek(messages, temperature=0.3, max_tokens=2000)
    if result.startswith("*LLM call failed"):
        return _explain_candidates_local(
            candidates_df, targets, weights, prop_stds, source_label, n_explain
        )
    return result


def _strategy_suggestion(
    targets,
    weights,
    prop_stds,
    known_count,
    unknown_count,
    generated_count,
    generated_pass_count,
    total_known,
    gen_df=None,
):
    total_attempted = generated_count
    targets = _normalize_target_specs(targets)
    msg_parts = [
        "The inverse design pipeline has completed. Here is a strategy analysis:"
    ]
    msg_parts.append(
        f"\n- Step 1 (Known-property scoring): {known_count} candidates from {total_known:,} materials"
    )
    msg_parts.append(
        f"- Step 2 (Unknown-property recommendation): {unknown_count} candidates"
    )
    msg_parts.append(
        f"- Step 3 (MatterGen generation): {generated_count} structures generated"
    )
    msg_parts.append(
        f"- Step 4 (Forward validation): {generated_pass_count} passed target criteria out of {generated_count}"
    )

    if generated_pass_count == 0 and generated_count > 0:
        msg_parts.append(
            f"\n⚠️ No generated structures met the targets. Consider: (1) loosening target ranges, (2) increasing generation count, or (3) checking whether targets are mutually achievable in the candidate chemical space."
        )

    # Include top generated candidates if available
    gen_top_text = ""
    if gen_df is not None and not gen_df.empty:
        gen_top = gen_df.head(min(3, len(gen_df)))
        gen_parts = ["\nTop generated candidates:"]
        property_map = {
            "pred_tm": "Tm",
            "pred_er": "εr",
            "pred_qf": "Q×f",
            "pred_tcf": "τf",
        }
        for _, row in gen_top.iterrows():
            formula = row.get("Formula", row.get("formula", "Unknown"))
            props = []
            for col_key, label in property_map.items():
                val = row.get(col_key)
                if pd.notna(val) and val is not None:
                    props.append(f"{label}={val:.2f}" if col_key != "pred_qf" else f"{label}={val:.0f}")
            nearest = row.get("nearest_known", None)
            sim = row.get("nearest_sim", None)
            nearest_str = ""
            if pd.notna(nearest) and nearest:
                sim_str = f" (sim={sim:.4f})" if isinstance(sim, (int, float)) and pd.notna(sim) else ""
                nearest_str = f", nearest known: {nearest}{sim_str}"
            gen_parts.append(f"  {formula}: {', '.join(props)}{nearest_str}")
        gen_top_text = "\n".join(gen_parts)

    target_summary = "\n".join(
        [
            f"  {_spec_text(targets[1], 'εr')} (weight {weights[1]:.2f}), {_spec_text(targets[2], 'Q×f')} (weight {weights[2]:.2f}), "
            f"{_spec_text(targets[3], 'τf')} (weight {weights[3]:.2f}), {_spec_text(targets[0], 'Tm')} (weight {weights[0]:.2f})"
        ]
    )

    prompt = f"""You are a microwave dielectric ceramics research advisor. A user ran the inverse design pipeline with these targets:
{target_summary}

Results summary:
{chr(10).join(msg_parts)}
{gen_top_text}

Please provide actionable design guidance:

1. **Target adjustment**: Identify which property constraint is hardest to satisfy given the candidate chemical space, and suggest specific target relaxations or weight adjustments to improve the generated pass rate.
2. **Next-step exploration**: Recommend concrete follow-up actions — which top candidates to prioritize for experimental validation, what compositional variants to explore around the best candidates, and whether to re-run generation with adjusted priors.
3. **Synthesis quick reference**: For the top 5 candidates across both sources, provide a one-sentence synthesis route suggestion for each, covering method, key precursors, and typical conditions based on known ceramic chemistry and literature knowledge. Base on composition and structure type.
4. **Novelty validation**: If generated candidates contain elements or compositions absent from the database, suggest experimental strategies to validate their plausibility (e.g., phase stability checks, targeted dopant studies) rather than describing their novelty.

Items 1, 2, and 4: 2-3 concise paragraphs. Item 3: compact list (one sentence per candidate).
Be specific with numbers. Do not repeat observations already covered in the interpretation. Output ONLY the guidance, no markdown headers."""

    messages = [
        {
            "role": "system",
            "content": "You are a materials science research advisor. Provide actionable, concise strategy recommendations.",
        },
        {"role": "user", "content": prompt},
    ]
    result = _call_deepseek(messages, temperature=0.4, max_tokens=2000)
    if result.startswith("*LLM call failed"):
        return _strategy_suggestion_local(
            targets, weights, prop_stds,
            known_count, unknown_count, generated_count, generated_pass_count, total_known,
            gen_df=gen_df,
        )
    return result


# ===================================================================
#  Main render entry point
# ===================================================================


def render():
    """Render inverse design tab content -- called by app_unified.py"""

    index, material_ids, metadata, feature_dict = _load_all()

    try:
        mace_calc, adaptor = _load_mace()
        mace_available = True
    except Exception:
        mace_available = False

    example_vec = feature_dict[material_ids[0]]
    FEAT_DIM = len(example_vec)

    # ---- Tab header ----
    st.markdown(
        """
    <div class="page-header">
        <p class="main-title">Inverse Design Optimizer</p>
        <p class="subtitle">Tiered search pipeline: Score known → Mine unknowns → Generate novel → Validate & Explain</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---- Sub-tabs ----
    sub_pipeline, sub_struct, sub_formula = st.tabs(
        [
            "🧬 Reverse Design Pipeline",
            "🔬 Structure Search",
            "🔤 Formula Search",
        ]
    )

    # ===================================================================
    #  Sub-tab 1: Structure Similarity Search
    # ===================================================================
    with sub_struct:
        st.markdown("### Structure Similarity Search")
        st.markdown(
            "Upload a CIF file to find structurally similar materials "
            "using MACE-learned crystal fingerprints."
        )

        if not mace_available:
            st.warning("MACE calculator is not available — CIF upload disabled.")
        else:
            st.caption(
                f"Loaded {len(material_ids):,} materials · feature dim: {FEAT_DIM}"
            )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            db_filter1 = st.selectbox("Database", ["All", "MP", "COD"], key="db1")
        with col_b:
            sim_thresh1 = st.slider(
                "Similarity Threshold", 0.0, 1.0, 0.0, 0.05, key="sim1"
            )
        with col_c:
            top_k1 = st.slider("Results to show", 1, 50, 10, key="k1")

        uploaded = st.file_uploader("Choose a CIF file", type=["cif"], key="cif_upload")

        if uploaded and mace_available:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            with st.spinner("Extracting MACE fingerprint and searching..."):
                try:
                    qvec = _get_vector_from_cif(tmp_path, mace_calc, adaptor, FEAT_DIM)
                    res = _search_by_vector(
                        qvec,
                        top_k1,
                        index,
                        material_ids,
                        metadata,
                        None if db_filter1 == "All" else db_filter1,
                        sim_thresh1,
                    )
                    if res:
                        st.success(f"Found {len(res)} match(es)")
                        st.dataframe(
                            pd.DataFrame(res)[
                                [
                                    "material_id",
                                    "formula",
                                    "spacegroup",
                                    "similarity",
                                    "pred_tm",
                                    "pred_er",
                                    "pred_qf",
                                    "pred_tcf",
                                ]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.warning("No matching structures found.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # ===================================================================
    #  Sub-tab 2: Formula Search
    # ===================================================================
    with sub_formula:
        st.markdown("### Formula Search")
        st.markdown(
            "Enter a chemical formula to search the database. "
            "Matches by normalized formula first, then by element set, "
            "with structural similarity as fallback when no exact match exists."
        )

        col_input, col_btn = st.columns([4, 1])
        with col_input:
            q_formula = st.text_input(
                "Chemical formula",
                placeholder="e.g. BaTiO3, BaCuSi4O10, SrTiO3",
                key="formula_query",
                label_visibility="collapsed",
            )
        with col_btn:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True, key="formula_search_btn")

        if q_formula and search_btn:
            res = _search_by_formula(q_formula, metadata, index, material_ids, feature_dict)
            if res:
                df_res = pd.DataFrame(res)
                if "similarity" in df_res.columns:
                    st.info(
                        f"No exact or element-matched materials found. "
                        f"Showing **{len(res)}** structurally similar candidates "
                        f"(≥2 shared elements with `{q_formula}`)."
                    )
                else:
                    st.success(f"Found {len(res)} match(es)")
                st.dataframe(df_res, use_container_width=True)
            else:
                st.warning(f"No matching formulas found for `{q_formula}`.")

    # ===================================================================
    #  Sub-tab 0: Reverse Design Pipeline
    # ===================================================================
    with sub_pipeline:
        # ── Target specification ──
        st.markdown("#### 🎯 Target Properties & Weights")
        st.caption(
            "Set lower/upper bounds per property. Both filled = range; "
            "lower only = at least; upper only = at most; equal values = exact; "
            "both empty = unconstrained."
        )
        col1, col2 = st.columns(2)
        with col1:

            def _parse_bound(txt, default):
                s = (txt or "").strip()
                return default if s == "" else float(s)

            _ec1, _ec2 = st.columns(2)
            er_lo = _ec1.number_input("εr lower bound", value=25.0, key="er_lo")
            er_hi_txt = _ec2.text_input(
                "εr upper bound (empty = no upper)", value="35", key="er_hi_txt"
            )
            er_spec = {"lo": float(er_lo), "hi": _parse_bound(er_hi_txt, None)}

            _qc1, _qc2 = st.columns(2)
            qf_lo = _qc1.number_input(
                "Q×f lower bound", value=50000.0, step=1000.0, key="qf_lo"
            )
            qf_hi_txt = _qc2.text_input(
                "Q×f upper bound (empty = no upper)", value="", key="qf_hi_txt"
            )
            qf_spec = {"lo": float(qf_lo), "hi": _parse_bound(qf_hi_txt, None)}

            _tc1, _tc2 = st.columns(2)
            tcf_lo = _tc1.number_input("τf lower bound", value=-10.0, key="tcf_lo")
            tcf_hi_txt = _tc2.text_input(
                "τf upper bound (empty = no upper)", value="10", key="tcf_hi_txt"
            )
            tcf_spec = {"lo": float(tcf_lo), "hi": _parse_bound(tcf_hi_txt, None)}

            _mc1, _mc2 = st.columns(2)
            tm_lo = _mc1.number_input(
                "Tm lower bound", value=800.0, step=100.0, key="tm_lo"
            )
            tm_hi_txt = _mc2.text_input(
                "Tm upper bound (empty = no upper)", value="1400", key="tm_hi_txt"
            )
            tm_spec = {"lo": float(tm_lo), "hi": _parse_bound(tm_hi_txt, None)}
        with col2:
            w_er = st.slider("εr weight", 0.0, 1.0, 0.25, 0.05)
            w_qf = st.slider("Q×f weight", 0.0, 1.0, 0.25, 0.05)
            w_tcf = st.slider("τf weight", 0.0, 1.0, 0.25, 0.05)
            w_tm = st.slider("Tm weight", 0.0, 1.0, 0.25, 0.05)

        # ── Pipeline parameters ──
        st.markdown("#### ⚙️ Pipeline Parameters")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            top_k_known = st.slider("Step 1: Known candidates", 5, 30, 10, key="pk_k")
        with pc2:
            top_k_unknown = st.slider(
                "Step 2: Unknown candidates", 0, 30, 10, key="pu_k"
            )
        with pc3:
            use_existing_cifs = st.checkbox(
                "Load existing CIFs",
                value=False,
                help="Skip generation — use .cif files from a previous MatterGen output folder",
            )
            enable_mattergen = False
            if not use_existing_cifs:
                enable_mattergen = st.checkbox(
                    "Enable MatterGen (Step 3-4)",
                    value=False,
                    help="Diffusion generation + MACE forward validation. ~30-45 min total (varies with the requested structure count).",
                )

        mg_num = 8
        mg_bs = None  # batch size is no longer user-facing: decided automatically in the background
        cif_folder = None
        sg_mode = "OFF (no constraint)"
        mace_sim_mode = "OFF"
        w_sim = 0.5
        anion_policy = ANION_POLICY_DEFAULT
        anion_whitelist = None
        if use_existing_cifs:
            # Find latest generation subfolder
            existing_dirs = sorted(
                glob.glob(os.path.join(config.PATH_MATTERGEN_OUTPUT, "gen_*"))
            )
            default_dir = (
                existing_dirs[-1] if existing_dirs else config.PATH_MATTERGEN_OUTPUT
            )
            cif_folder = st.text_input(
                "CIF folder path",
                value=default_dir,
                help="Path to a MatterGen output subfolder (recursively searches *.cif)",
            )
        elif enable_mattergen:
            mgc1, mgc2 = st.columns(2)
            with mgc1:
                mg_num = st.number_input(
                    "Max structures to keep",
                    min_value=1,
                    max_value=512,
                    value=16,
                    step=1,
                    key="mg_n2",
                    help="Upper bound on the final kept count: a few extra structures (~1.5x safety margin) are generated and then truncated to this value, so the result never exceeds it. The batch size is decided automatically in the background.",
                )
            with mgc2:
                # Same planner as _run_mattergen → what is shown is what runs.
                _ui_plan = _plan_generation(mg_num)
                st.markdown("**Batch size (auto, background)**")
                st.caption(
                    f"{_ui_plan['batch_size']}/batch × {_ui_plan['num_batches']} batch(es) "
                    f"→ ≈{_ui_plan['expected_generated']} generated, truncated to ≤{_ui_plan['expected_kept']}"
                )
                st.caption(
                    f"⏱ Estimated diffusion time: {_ui_plan['est_minutes_low']}–{_ui_plan['est_minutes_high']} min "
                    "(plus forward validation ~5–15 min). A space-group constraint may split the run "
                    "into several routes, adding batches."
                )

            # ── Anion restriction (element pre-restriction) ──
            anc1, anc2 = st.columns(2)
            _anion_ui_options = {
                "Default (O/F/N/S/Cl)": ANION_POLICY_DEFAULT,
                "All anions (11)": ANION_POLICY_FULL,
                "Custom": ANION_POLICY_CUSTOM,
            }
            with anc1:
                _anion_label = st.selectbox(
                    "Anion restriction",
                    list(_anion_ui_options.keys()),
                    index=0,
                    key="mg_anion_policy",
                    help="Anions merged into the allowed element set; cations stay strictly "
                    "restricted to the composition prior. "
                    "Default (O/F/N/S/Cl) keeps the previous behaviour unchanged.",
                )
                anion_policy = _anion_ui_options[_anion_label]
            with anc2:
                if anion_policy == ANION_POLICY_CUSTOM:
                    anion_whitelist = st.multiselect(
                        "Custom anions",
                        list(_ANION_FULL),
                        default=list(_ANION_DEFAULT),
                        key="mg_anion_custom",
                        help="Pick at least one element. An empty selection falls back to "
                        "the default O/F/N/S/Cl set.",
                    )
                else:
                    st.markdown("**Effective anion base**")
                    st.caption(
                        ", ".join(_resolve_anion_base(anion_policy, None))
                        + " — always included in the allowed element set"
                    )

            # ── Structure guidance (space-group condition + MACE seed similarity) ──
            st.markdown("**🔬 Structure guidance (optional)**")
            sgc1, sgc2 = st.columns(2)
            with sgc1:
                sg_mode = st.selectbox(
                    "Space-group condition",
                    ["OFF (no constraint)", "Same SG as seed", "Same crystal system (relaxed)"],
                    index=0,
                    key="mg_sg_mode",
                    help="Constrain MatterGen by the Step-1 seed space group (uses the space_group checkpoint automatically). "
                    "Relaxed = all space-group numbers of the seed's crystal system (recommended). Falls back to unconstrained automatically when no seed or parsing fails.",
                )
            with sgc2:
                mace_sim_mode = st.selectbox(
                    "MACE seed similarity",
                    ["OFF", "Reference (display only)", "Into composite score"],
                    index=0,
                    key="mg_mace_sim",
                    help="Cosine similarity between the generated structure and the Step-1 seed's MACE fingerprint (256-dim); used as a Step-4 ranking reference or weighted composite score.",
                )
            w_sim = 0.5
            if mace_sim_mode == "Into composite score":
                w_sim = st.slider(
                    "Seed similarity weight w_sim (score' = score/(1 + w_sim·(1-sim)))",
                    0.0, 2.0, 0.5, 0.1, key="mg_sim_w",
                )

        # ── Progress tracking ──
        if "pipeline_state" not in st.session_state:
            st.session_state.pipeline_state = {}

        # ── Run button ──
        if st.button("▶ Run Pipeline", type="primary", key="run_pipeline"):
            # Two-bound input -> normalized into a spec (with op) for Step2/4 inline scoring and archive reuse (idempotent)
            targets = _normalize_target_specs([tm_spec, er_spec, qf_spec, tcf_spec])
            weights = [w_tm, w_er, w_qf, w_tcf]
            llm_details = []
            prop_stds = [
                metadata["pred_tm"].std() or 1.0,
                metadata["pred_er"].std() or 1.0,
                metadata["pred_qf"].std() or 1.0,
                metadata["pred_tcf"].std() or 1.0,
            ]

            # ━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━
            with st.status(
                "Step 1/4: Scoring known-property materials...", expanded=True
            ) as status:
                df_known, df_scored_all = _step1_score(
                    metadata, targets, weights, prop_stds, top_k=top_k_known
                )
                if not df_known.empty:
                    # Enhancement 2: LLM backfill of missing properties for Step1 top-N (fill as many as are missing; degrade gracefully without blocking)
                    df_known, _n_bk = _backfill_llm_props(
                        df_known, collect=llm_details
                    )
                    if _n_bk:
                        _apply_scoring(df_known, targets, weights, prop_stds)
                        df_known = df_known.sort_values(
                            "score", ascending=False
                        ).reset_index(drop=True)
                        st.caption(
                            f"LLM backfilled {_n_bk} candidate(s) with missing properties (see llm_backfilled column)"
                        )
                if df_known.empty:
                    st.warning("No materials with property data found.")
                    status.update(label="Step 1: Failed — no data")
                else:
                    n_total = len(df_scored_all)
                    st.success(
                        f"Step 1 complete — {n_total:,} materials evaluated, Top {len(df_known)} shown"
                    )
                    status.update(label="Step 1: Complete")

                    st.markdown("**▼ Step 1: Known Candidates (sorted by score)**")
                    st.dataframe(
                        df_known[
                            [
                                "material_id",
                                "formula",
                                "spacegroup",
                                "pred_tm",
                                "pred_er",
                                "pred_qf",
                                "pred_tcf",
                                "llm_backfilled",
                                "score",
                            ]
                        ].fillna(float("nan")).style.format(
                            {"score": "{:.4f}"}, na_rep="N/A"
                        ),
                        use_container_width=True,
                    )
                    st.session_state.pipeline_state["df_known"] = df_known
                    st.session_state.pipeline_state["targets"] = targets
                    st.session_state.pipeline_state["weights"] = weights
                    st.session_state.pipeline_state["prop_stds"] = prop_stds

            # ━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━
            df_unknown = pd.DataFrame()
            if not df_known.empty and top_k_unknown > 0:
                with st.status(
                    "Step 2/4: Recommending unknown-property analogs...", expanded=True
                ) as status2:
                    seed_ids = df_known.head(3)["material_id"].tolist()
                    df_unknown = _step2_recommend_unknown(
                        metadata,
                        feature_dict,
                        material_ids,
                        seed_ids=seed_ids,
                        top_k=top_k_unknown,
                    )
                    if not df_unknown.empty:
                        # Enhancement 2: Step2 likewise backfills missing properties for top-N and marks the LLM source
                        df_unknown, _n_bk2 = _backfill_llm_props(
                            df_unknown, collect=llm_details
                        )
                        if _n_bk2:
                            st.caption(
                                f"LLM backfilled {_n_bk2} candidate(s) with missing properties (see llm_backfilled column)"
                            )
                    if df_unknown.empty:
                        st.info("No materials with fully unknown properties found.")
                        status2.update(
                            label="Step 2: Complete (0 found)"
                        )
                    else:
                        st.success(
                            f"Step 2 complete — {len(df_unknown)} structurally similar unknown-property materials recommended"
                        )
                        status2.update(label="Step 2: Complete")

                        st.markdown(
                            f"**▼ Step 2: Unknown Candidates (structural analogs of Top 3 known)**"
                        )
                        st.dataframe(
                            df_unknown[
                                ["material_id", "formula", "spacegroup",
                                 "similarity", "llm_backfilled"]
                            ].fillna(float("nan")).style.format(
                                {"similarity": "{:.4f}"}, na_rep="N/A"
                            ),
                            use_container_width=True,
                        )
                        st.session_state.pipeline_state["df_unknown"] = df_unknown

            # ━━━━━━━━━━━━━━━━━━━━ Step 3 ━━━━━━━━━━━━━━━━━━━━
            generated_pass = 0
            generated_total = 0
            if (enable_mattergen or use_existing_cifs) and not df_known.empty:
                if use_existing_cifs and cif_folder and os.path.isdir(cif_folder):
                    label = "Step 3/4: Loading existing CIFs..."
                else:
                    label = "Step 3/4: MatterGen constrained generation..."
                with st.status(label, expanded=True) as status3:
                    prior, summary = _extract_composition_prior(
                        metadata, df_known, df_unknown, n_sample=15
                    )
                    if prior:
                        allowed_for_gen = list(summary["top_elements"])
                        # Show the effective element set (anions + prior)
                        _anions_base = set(
                            _resolve_anion_base(anion_policy, anion_whitelist)
                        )
                        _effective = sorted(_anions_base | set(allowed_for_gen))
                        st.info(
                            f"Element pre-restriction: {len(_effective)} elements — "
                            f"{', '.join(_effective)}\n\n"
                            f"Anions ({', '.join(sorted(_anions_base))}) always included; "
                            f"cations strictly restricted to composition prior."
                        )
                    else:
                        allowed_for_gen = None
                        st.info(
                            "No composition prior extractable — using unrestricted generation."
                        )

                    if use_existing_cifs and cif_folder and os.path.isdir(cif_folder):
                        # ── Load existing structures (CIF or extxyz, recursive) ──
                        structures = []
                        cif_files = sorted(
                            glob.glob(
                                os.path.join(cif_folder, "**", "*.cif"), recursive=True
                            )
                        )
                        for cf in cif_files:
                            try:
                                structures.append(Structure.from_file(cf))
                            except Exception:
                                pass
                        if not structures:
                            xyz_files = sorted(
                                glob.glob(
                                    os.path.join(
                                        cif_folder, "**", "generated_crystals*.extxyz"
                                    ),
                                    recursive=True,
                                )
                            )
                            if xyz_files:
                                try:
                                    from ase.io import read as ase_read

                                    atoms_list = ase_read(xyz_files[-1], index=":")
                                    structures = [
                                        AseAtomsAdaptor().get_structure(at)
                                        for at in atoms_list
                                    ]
                                except Exception:
                                    pass
                        if not structures:
                            st.warning(
                                f"No .cif or generated_crystals*.extxyz found recursively in `{cif_folder}`"
                            )
                            status3.update(
                                label="Step 3: No structures found"
                            )
                        else:
                            # Filter: keep only ceramic-relevant (must contain oxygen)
                            n_before = len(structures)
                            structures = _filter_chemically_plausible(structures)
                            n_after = len(structures)
                            st.success(
                                f"Loaded {n_after} ceramic structures from `{cif_folder}`"
                                + (
                                    f" ({n_before - n_after} non-ceramic filtered out)"
                                    if n_before > n_after
                                    else ""
                                )
                            )
                            status3.update(
                                label="Step 3: Structures loaded"
                            )
                            st.session_state.pipeline_state["gen_structures"] = (
                                structures
                            )
                            st.session_state.pipeline_state["gen_out_dir"] = cif_folder
                            generated_total = len(structures)
                            gen_data = []
                            for i, s in enumerate(structures):
                                gen_data.append(
                                    {
                                        "#": i + 1,
                                        "Formula": s.composition.reduced_formula,
                                        "Atoms": len(s),
                                    }
                                )
                            if gen_data:
                                st.dataframe(
                                    pd.DataFrame(gen_data), use_container_width=True
                                )
                    else:
                        try:
                            # resolve structure-guidance options
                            sg_cond = None
                            if sg_mode in ("Same SG as seed", "Same crystal system (relaxed)") and not df_known.empty:
                                _mode = "same" if sg_mode == "Same SG as seed" else "system"
                                seed_sg = df_known.iloc[0].get("spacegroup", None)
                                sg_cond = _sg_relax(seed_sg, mode=_mode)
                                if sg_cond is None:
                                    st.warning(
                                        "Failed to parse the seed space group; structure constraint skipped for this run (fallback to unconstrained generation)."
                                    )
                            # MACE seed-similarity config for Step 4 re-ranking
                            st.session_state.pipeline_state["mg_sim_cfg"] = {
                                "mode": mace_sim_mode,
                                "w": float(w_sim),
                            }
                            structures, out_dir = _run_mattergen(
                                num_structures=mg_num,
                                batch_size=mg_bs,
                                output_subdir=f"gen_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                                allowed_elements=allowed_for_gen,
                                anion_policy=anion_policy,
                                anion_whitelist=anion_whitelist,
                                space_group_cond=sg_cond,
                            )
                            if sg_cond is not None:
                                _sg_disp = (
                                    str(sg_cond[0])
                                    if len(sg_cond) == 1
                                    else f"{sg_cond[0]}–{sg_cond[-1]} ({len(sg_cond)} SG)"
                                )
                                st.info(f"Space-group condition applied: SG {_sg_disp}")
                            st.success(
                                f"Generated {len(structures)} structures → `{out_dir}`"
                            )
                            status3.update(
                                label="Step 3: Generation complete"
                            )
                            st.session_state.pipeline_state["gen_structures"] = (
                                structures
                            )
                            st.session_state.pipeline_state["gen_out_dir"] = out_dir
                            generated_total = len(structures)
                        except FileNotFoundError as e:
                            st.error(f"MatterGen checkpoint not found: {e}")
                            status3.update(
                                label="Step 3: Failed — checkpoint missing"
                            )
                        except Exception as e:
                            st.error(f"MatterGen generation failed: {e}")
                            status3.update(label="Step 3: Failed")

            # ━━━━━━━━━━━━━━━━━━━━ Step 4 ━━━━━━━━━━━━━━━━━━━━
            need_unknown_pred = (
                top_k_unknown > 0 and not df_unknown.empty and mace_available
            )
            need_gen_pred = generated_total > 0 and mace_available

            if need_unknown_pred or need_gen_pred:
                with st.status(
                    "Step 4/4: Forward prediction validation...", expanded=True
                ) as status4:
                    # -------- 4a: Predict unknown candidates --------
                    df_unknown_scored = pd.DataFrame()
                    if need_unknown_pred:
                        st.markdown("**Predicting unknown candidates...**")
                        unk_results = []
                        for _, row in df_unknown.iterrows():
                            cif_path = row.get("cif_path", None)
                            if not cif_path or not isinstance(cif_path, str):
                                continue
                            try:
                                fp = _get_vector_from_cif(
                                    cif_path, mace_calc, adaptor, FEAT_DIM
                                )
                                neighbors = _search_by_vector(
                                    fp,
                                    1,
                                    index,
                                    material_ids,
                                    metadata,
                                    db_filter=None,
                                    min_sim=0.0,
                                )
                                nearest_formula = (
                                    neighbors[0]["formula"] if neighbors else "N/A"
                                )
                            except Exception:
                                nearest_formula = "N/A"
                            try:
                                struct = Structure.from_file(cif_path)
                                pred = _run_forward_prediction(struct)
                            except Exception:
                                pred = {}
                            unk_results.append(
                                {
                                    "material_id": row["material_id"],
                                    "Formula": row["formula"],
                                    "spacegroup": row.get("spacegroup", "N/A"),
                                    "similarity_to_seed": row["similarity"],
                                    "nearest_known": nearest_formula,
                                    "pred_er": pred.get("pred_er", None),
                                    "pred_qf": pred.get("pred_qf", None),
                                    "pred_tcf": pred.get("pred_tcf", None),
                                    "pred_tm": pred.get("pred_tm", None),
                                }
                            )
                            if len(unk_results) % _STEP4_CKP_INTERVAL == 0:
                                _save_step4_checkpoint("unknown", unk_results)
                        if unk_results:
                            df_unk = pd.DataFrame(unk_results)
                            # Score against targets
                            if not df_unk.empty:
                                has_any = (
                                    df_unk[
                                        ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
                                    ]
                                    .notna()
                                    .sum(axis=1)
                                    > 0
                                )
                                df_unknown_scored = df_unk[has_any].copy()
                                if not df_unknown_scored.empty:
                                    total_d = np.zeros(len(df_unknown_scored))
                                    for col, target, std, w in zip(
                                        ["pred_tm", "pred_er", "pred_qf", "pred_tcf"],
                                        targets,
                                        prop_stds,
                                        weights,
                                    ):
                                        if w > 0:
                                            mask = df_unknown_scored[col].notna()
                                            if mask.any():
                                                vals = (
                                                    df_unknown_scored.loc[
                                                        mask, col
                                                    ]
                                                    .values
                                                )
                                                d = np.array(
                                                    [
                                                        _spec_distance(
                                                            float(v), target,
                                                            float(std),
                                                        )
                                                        for v in vals
                                                    ]
                                                )
                                                total_d[mask.values] += w * d
                                    df_unknown_scored["score"] = 1.0 / (1.0 + total_d)
                                    df_unknown_scored = df_unknown_scored.sort_values(
                                        "score", ascending=False
                                    )
                                    st.success(
                                        f"Predicted {len(df_unknown_scored)} unknown candidates (from Step 2)"
                                    )
                                    st.markdown("**▼ Step 2 Candidates — Now Scored**")
                                    st.dataframe(
                                        df_unknown_scored[
                                            [
                                                "material_id",
                                                "Formula",
                                                "spacegroup",
                                                "pred_er",
                                                "pred_qf",
                                                "pred_tcf",
                                                "pred_tm",
                                                "similarity_to_seed",
                                                "nearest_known",
                                                "score",
                                            ]
                                        ].fillna(float("nan")).style.format(
                                            {
                                                "score": "{:.4f}",
                                                "similarity_to_seed": "{:.4f}",
                                                "pred_er": "{:.2f}",
                                                "pred_qf": "{:.0f}",
                                                "pred_tcf": "{:.2f}",
                                                "pred_tm": "{:.1f}",
                                            },
                                            na_rep="N/A",
                                        ),
                                        use_container_width=True,
                                    )
                                    st.session_state.pipeline_state[
                                        "df_unknown_scored"
                                    ] = df_unknown_scored

                    # -------- 4b: Predict generated structures --------
                    gen_results = []
                    # MACE seed-similarity (structure-guided re-ranking) setup
                    seed_vec_mace = None
                    _sim_cfg = st.session_state.pipeline_state.get(
                        "mg_sim_cfg", {}
                    ) or {}
                    _sim_mode = _sim_cfg.get("mode", "OFF")
                    _sim_w = float(_sim_cfg.get("w", 0.5))
                    _do_sim = _sim_mode != "OFF" and not df_known.empty
                    if _do_sim:
                        _seed_rows = df_known.head(min(3, len(df_known)))
                        _vecs = []
                        for _, _r in _seed_rows.iterrows():
                            _cp = _r.get("cif_path", None)
                            if not _cp or not isinstance(_cp, str):
                                continue
                            try:
                                _v = _get_vector_from_cif(
                                    _cp, mace_calc, adaptor, FEAT_DIM
                                )
                                _vecs.append(_v[:256])  # MACE structure part only
                            except Exception:
                                pass
                        if _vecs:
                            _arr = np.stack(_vecs)
                            _arr /= (
                                np.linalg.norm(_arr, axis=1, keepdims=True) + 1e-9
                            )
                            seed_vec_mace = np.mean(_arr, axis=0)
                            seed_vec_mace /= (
                                np.linalg.norm(seed_vec_mace) + 1e-9
                            )
                    if need_gen_pred:
                        structs = st.session_state.pipeline_state["gen_structures"]
                        for i, s in enumerate(structs):
                            try:
                                formula = s.composition.reduced_formula
                                fp = _get_vector_from_structure(
                                    s, mace_calc, adaptor, FEAT_DIM
                                )
                                neighbors = _search_by_vector(
                                    fp,
                                    1,
                                    index,
                                    material_ids,
                                    metadata,
                                    db_filter=None,
                                    min_sim=0.0,
                                )
                                nearest = (
                                    neighbors[0]["formula"] if neighbors else "N/A"
                                )
                                nearest_sim = (
                                    neighbors[0]["similarity"] if neighbors else 0.0
                                )
                                # cosine to aggregate seed fingerprint (MACE 256-dim)
                                seed_sim = (
                                    float(np.dot(fp[:256], seed_vec_mace))
                                    if seed_vec_mace is not None
                                    else None
                                )
                            except Exception:
                                nearest = "N/A"
                                nearest_sim = 0.0
                                seed_sim = None

                            # Check if formula already exists in DB — skip prediction
                            cached = metadata[metadata["formula"] == formula]
                            if (
                                not cached.empty
                                and cached[
                                    ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
                                ]
                                .notna()
                                .any(axis=1)
                                .any()
                            ):
                                row = cached.iloc[0]
                                pred = {
                                    "pred_er": row.get("pred_er"),
                                    "pred_qf": row.get("pred_qf"),
                                    "pred_tcf": row.get("pred_tcf"),
                                    "pred_tm": row.get("pred_tm"),
                                    "source": "DB lookup",
                                }
                            else:
                                try:
                                    pred = _run_forward_prediction(s)
                                    pred["source"] = "fresh prediction"
                                except Exception:
                                    pred = {"source": "failed"}

                            gen_results.append(
                                {
                                    "#": i + 1,
                                    "Formula": formula,
                                    "Atoms": len(s),
                                    "pred_er": pred.get("pred_er", None),
                                    "pred_qf": pred.get("pred_qf", None),
                                    "pred_tcf": pred.get("pred_tcf", None),
                                    "pred_tm": pred.get("pred_tm", None),
                                    "nearest_known": nearest,
                                    "nearest_sim": nearest_sim,
                                    "seed_sim": seed_sim,
                                }
                            )

                    df_gen = pd.DataFrame(gen_results)

                    # Re-score against targets
                    if not df_gen.empty:
                        has_any = (
                            df_gen[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                            .notna()
                            .sum(axis=1)
                            > 0
                        )
                        df_gen_scored = df_gen[has_any].copy()
                        if not df_gen_scored.empty:
                            total_d = np.zeros(len(df_gen_scored))
                            for col, target, std, w in zip(
                                ["pred_tm", "pred_er", "pred_qf", "pred_tcf"],
                                targets,
                                prop_stds,
                                weights,
                            ):
                                if w > 0:
                                    mask = df_gen_scored[col].notna()
                                    if mask.any():
                                        vals = df_gen_scored.loc[mask, col].values
                                        d = np.array(
                                            [
                                                _spec_distance(
                                                    float(v), target, float(std)
                                                )
                                                for v in vals
                                            ]
                                        )
                                        total_d[mask.values] += w * d
                            df_gen_scored["score"] = 1.0 / (1.0 + total_d)
                            # structure-guided re-ranking: fold seed similarity in
                            if (
                                _sim_mode == "Into composite score"
                                and "seed_sim" in df_gen_scored.columns
                            ):
                                _sim_vals = (
                                    df_gen_scored["seed_sim"]
                                    .fillna(0.0)
                                    .clip(0.0, 1.0)
                                    .values
                                )
                                df_gen_scored["score"] = (
                                    df_gen_scored["score"].values
                                    / (1.0 + _sim_w * (1.0 - _sim_vals))
                                )
                                df_gen_scored = df_gen_scored.sort_values(
                                    "score", ascending=False
                                )
                            else:
                                df_gen_scored = df_gen_scored.sort_values(
                                    "score", ascending=False
                                )
                            top_score = df_gen_scored.iloc[0].get("score", 0)
                            generated_pass = len(df_gen_scored)

                            st.success(
                                f"Step 4 complete — {generated_pass}/{generated_total} generated structures scored"
                            )
                            status4.update(
                                label="Step 4: Validation complete"
                            )

                            st.markdown(
                                "**▼ Generated & Validated Candidates (sorted by score)**"
                            )
                            display_cols = [
                                "#",
                                "Formula",
                                "pred_er",
                                "pred_qf",
                                "pred_tcf",
                                "pred_tm",
                                "nearest_known",
                                "nearest_sim",
                                "seed_sim",
                                "score",
                            ]
                            st.dataframe(
                                df_gen_scored[display_cols]
                                .fillna(float("nan"))
                                .style.format(
                                    {
                                        "score": "{:.4f}",
                                        "nearest_sim": "{:.4f}",
                                        "seed_sim": "{:.4f}",
                                        "pred_er": "{:.2f}",
                                        "pred_qf": "{:.0f}",
                                        "pred_tcf": "{:.2f}",
                                        "pred_tm": "{:.1f}",
                                    },
                                    na_rep="N/A",
                                ),
                                use_container_width=True,
                            )
                            st.session_state.pipeline_state["df_gen"] = df_gen_scored
                        else:
                            st.warning("No generated structures had valid predictions")
                            status4.update(
                                label="Step 4: No valid predictions"
                            )
                    else:
                        status4.update(
                            label="Step 4: No generated data"
                        )

            # ━━━━━━━━━━━━━━━━━━━━ LLM Explanation ━━━━━━━━━━━━━━━━━━━━
            if not df_known.empty:
                st.divider()
                st.markdown("### 🧠 LLM Analysis")

                with st.status(
                    "Generating chemical interpretation & strategy...", expanded=True
                ) as status_llm:
                    # Build combined explanation: database + generated candidates
                    gen_for_llm = None
                    if "df_gen" in st.session_state.pipeline_state:
                        gen_for_llm = st.session_state.pipeline_state["df_gen"]

                    explanation = _explain_candidates(
                        df_known,
                        targets,
                        weights,
                        prop_stds,
                        source_label="known",
                        n_explain=min(5, len(df_known)),
                        gen_df=gen_for_llm,
                    )
                    if explanation and not explanation.startswith("*LLM"):
                        st.markdown("#### 📊 Why These Candidates Scored Well")
                        st.markdown(explanation)
                    else:
                        st.info(explanation)

                    # Strategy suggestion
                    strategy = _strategy_suggestion(
                        targets,
                        weights,
                        prop_stds,
                        known_count=len(df_known),
                        unknown_count=len(df_unknown) if top_k_unknown > 0 else 0,
                        generated_count=generated_total,
                        generated_pass_count=generated_pass,
                        total_known=(
                            ~metadata[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                            .isna()
                            .all(axis=1)
                        ).sum(),
                        gen_df=gen_for_llm,
                    )
                    if strategy and not strategy.startswith("*LLM"):
                        st.markdown("#### 🎯 Strategy Recommendations")
                        st.markdown(strategy)
                    else:
                        st.info(strategy)

                    status_llm.update(label="LLM analysis complete")

            # ── Summary stats ──
            if not df_known.empty:
                st.divider()
                st.markdown("### 📈 Pipeline Summary")
                known_total = (
                    ~metadata[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                    .isna()
                    .all(axis=1)
                ).sum()
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Known scored", f"{len(df_known)}")
                with sc2:
                    st.metric("Unknown recomm.", f"{len(df_unknown)}")
                with sc3:
                    st.metric("Generated", f"{generated_total}")
                with sc4:
                    st.metric("Passed validation", f"{generated_pass}")

                # -- Enhancement 3: auto-save prediction results (following the gen_* pattern) --
                _df_unknown_scored = locals().get(
                    "df_unknown_scored", pd.DataFrame()
                )
                _df_gen_scored = locals().get("df_gen_scored", pd.DataFrame())
                _save_dir = _save_prediction_run(
                    targets,
                    weights,
                    prop_stds,
                    df_known=df_known,
                    df_unknown_scored=(
                        _df_unknown_scored
                        if _df_unknown_scored is not None
                        and not _df_unknown_scored.empty
                        else None
                    ),
                    df_gen=(
                        _df_gen_scored
                        if _df_gen_scored is not None
                        and not _df_gen_scored.empty
                        else None
                    ),
                    llm_details=llm_details,
                    structures_dir=st.session_state.pipeline_state.get(
                        "gen_out_dir"
                    ),
                    llm_analysis={
                        "explanation": locals().get("explanation"),
                        "strategy": locals().get("strategy"),
                    },
                    cfg={
                        "top_k_known": top_k_known,
                        "top_k_unknown": top_k_unknown,
                        "generated_total": generated_total,
                        "generated_pass": generated_pass,
                    },
                )
                if _save_dir:
                    st.success(f"Prediction record auto-saved to `{_save_dir}`")


# ===================================================================
#  Forward prediction wrapper (reuses Section 1 plugin pipeline)
# ===================================================================

_forward_agent = None


def _get_forward_agent():
    global _forward_agent
    if _forward_agent is None:
        from forward_prediction import ForwardPredictionAgent

        _forward_agent = ForwardPredictionAgent()
    return _forward_agent


# ===================================================================
#  Step 4 checkpoint - interruption protection for batch forward prediction (F6)
#  Every _STEP4_CKP_INTERVAL candidates, computed results are persisted to CSV (overwrite),
#  so already-computed intermediate results survive Streamlit reruns / process interruption, avoiding full recomputation amplifying missing values.
# ===================================================================
_STEP4_CKP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "step4_checkpoints"
)
# Root directory for auto-saving prediction results (following the mattergen_output/gen_* pattern)
_PRED_SAVE_DIR = os.path.join(config.INVERSE_DB_DIR, "ido_predictions")
_STEP4_CKP_INTERVAL = 50


def _save_step4_checkpoint(tag, records):
    """Periodically persist collected prediction results to CSV (checkpoint only; not used in scoring)."""
    try:
        if not records:
            return
        os.makedirs(_STEP4_CKP_DIR, exist_ok=True)
        df = pd.DataFrame(records)
        path = os.path.join(_STEP4_CKP_DIR, f"step4_{tag}_checkpoint.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(
            f"   [Step4 checkpoint] {tag}: saved {len(records)} rows -> {path}",
            flush=True,
        )
    except Exception as e:
        print(f"   [Step4 checkpoint] {tag} failed: {e}", flush=True)


def _run_forward_prediction(structure):
    try:
        agent = _get_forward_agent()
        from pymatgen.core import Structure as MgStructure

        if not isinstance(structure, MgStructure):
            return {}
        formula = structure.composition.reduced_formula
        if not formula:
            return {}
        data = agent.analyze_single(formula, external_struct=structure)
        result = {}
        # analyze_single returns flat dict: {"er_mean": val, "qxf_mean": val, "tcf_mean": val, ...}
        _prop_map = {"er_mean": "pred_er", "qxf_mean": "pred_qf", "tcf_mean": "pred_tcf", "tm_mean": "pred_tm"}
        if data and isinstance(data, dict):
            for src_key, dst_key in _prop_map.items():
                val = data.get(src_key)
                if val is not None:
                    result[dst_key] = val

        # Fallback: for properties not predicted by ML, extract backfill values from enhanced_results (LLM/Lit).
        # For example, non-perovskite structures can only have er predicted by the GENERAL plugin; Qxf/tf/Tm need LLM backfill.
        # F3: backfill extraction must filter out LLM-failed placeholder fake values (default_pred / confidence<=0 / status=error),
        #     so that fake values such as qf=20000 / tcf=0 are not written into the CSV as valid predictions.
        enhanced_results = data.get('enhanced_results', []) if data and isinstance(data, dict) else []
        _enhanced_target_map = {"er": "pred_er", "qf": "pred_qf", "tcf": "pred_tcf", "tm": "pred_tm"}
        llm_details = []
        for entry in enhanced_results:
            target = entry.get('target', '')
            dst = _enhanced_target_map.get(target)
            if dst and dst not in result:
                status = entry.get('status', 'ok')
                if status in ('error', 'out_of_range'):
                    continue
                conf = entry.get('confidence', 0)
                if isinstance(conf, str):
                    # 'low'/'medium'/'high' string labels come from EnhancedPredictor._conf_label;
                    # the upstream _call_llm_for_method already filters status=error/out_of_range and confidence<=0,
                    # so anything reaching enhanced_results is a valid LLM output -- pass it through directly.
                    pass
                else:
                    try:
                        if conf is None or float(conf) <= 0:
                            continue
                    except (TypeError, ValueError):
                        continue
                value = entry.get('value')
                if value is not None:
                    result[dst] = value
                    llm_details.append({
                        "target": target,
                        "method": entry.get("method", ""),
                        "value": value,
                        "confidence": str(conf),
                        "status": "ok",
                        "route": entry.get("method", ""),
                        "reasoning": entry.get("reasoning", ""),
                        "neighbor_summary": entry.get("neighbor_summary", ""),
                    })
        result["_llm_details"] = llm_details

        print(f"   [DEBUG _run_forward_prediction] formula={formula} result={result}", flush=True)
        return result
    except Exception:
        import traceback
        traceback.print_exc()
        return {}


# ===================================================================
#  UI-free Step 4 + full-pipeline orchestration (CADR chat path)
#  Added 2026-09-03 — callable from chat_module without Streamlit widgets.
# ===================================================================


def _apply_scoring(df, targets, weights, prop_stds):
    """Scoring shared by Step-4 branches: score = 1 / (1 + sum(w * dist / std)).
    Supports eq/gte/lte/range constraints (spec dict); legacy scalar targets are treated as eq for compatibility."""
    specs = _normalize_target_specs(targets)
    total_d = np.zeros(len(df))
    for col, spec, std, w in zip(
        ["pred_tm", "pred_er", "pred_qf", "pred_tcf"], specs, prop_stds, weights
    ):
        if w > 0:
            mask = df[col].notna()
            if mask.any():
                vals = df.loc[mask, col].values
                d = np.array([_spec_distance(v, spec, std) for v in vals], dtype=float)
                total_d[mask.values] += w * d
    df["score"] = 1.0 / (1.0 + total_d)
    return df


def _backfill_llm_props(df, prop_cols=None, collect=None):
    """Run _run_forward_prediction on the top-N candidates via their inline cif_path to backfill missing properties.

    - Fill as many as are missing: only fills columns in prop_cols that are NaN/None
    - Backfilled results are marked in the `llm_backfilled` column (comma-separated property names); the source is LLM
    - Any row that fails degrades to skip without blocking the flow
    - When collect is a list, collects the LLM details of this run (including material_id/formula) for auto-save
    Returns (df, n_backfilled).
    """
    if df is None or df.empty:
        return df, 0
    if prop_cols is None:
        prop_cols = ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
    prop_cols = [c for c in prop_cols if c in df.columns]
    if not prop_cols:
        return df, 0
    if "llm_backfilled" not in df.columns:
        df["llm_backfilled"] = ""
    n_backfilled = 0
    for idx in df.index:
        row = df.loc[idx]
        missing = [c for c in prop_cols if pd.isna(row.get(c))]
        if not missing:
            continue
        cif_path = row.get("cif_path", None)
        if not cif_path or not isinstance(cif_path, str):
            continue
        try:
            struct = Structure.from_file(cif_path)
            pred = _run_forward_prediction(struct)
        except Exception:
            pred = {}
        filled = []
        for c in missing:
            v = pred.get(c, None)
            if v is not None and not (isinstance(v, float) and v != v):
                df.loc[idx, c] = v
                filled.append(c)
        if filled:
            prev = df.loc[idx, "llm_backfilled"]
            df.loc[idx, "llm_backfilled"] = (
                f"{prev},{','.join(filled)}" if prev else ",".join(filled)
            )
            n_backfilled += 1
        if collect is not None:
            det = pred.get("_llm_details", [])
            if det:
                for d in det:
                    d2 = dict(d)
                    d2["material_id"] = row.get("material_id", "")
                    d2["formula"] = row.get("formula", row.get("Formula", ""))
                    d2["stage"] = "backfill"
                    collect.append(d2)
    return df, n_backfilled


def _save_prediction_run(targets, weights, prop_stds, df_known=None,
                         df_unknown_scored=None, df_gen=None,
                         llm_details=None, cfg=None,
                         structures_dir=None, llm_analysis=None):
    """Auto-save prediction results following the gen_* pattern.

    Directory: ido_predictions/run_{YYYYMMDD_HHMMSS}/
      - predictions_scored.csv : scored results merged from three sources (with source / llm_backfilled markers)
      - llm_details.csv        : LLM backfill details (target/method/value/confidence/status/route/reasoning)
      - run_config.json        : target specs / weights / prop_stds / key cfg
      - structures/            : structure files generated/loaded in this run (*.cif / *.extxyz / *.zip, optional)
      - llm_analysis.md        : LLM chemical interpretation + strategy suggestion text (optional)
    Returns the absolute directory path; returns None on failure (does not block the main flow).
    """
    try:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_PRED_SAVE_DIR, f"run_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        frames = []
        if df_known is not None and not df_known.empty:
            k = df_known.copy()
            k["source"] = "known"
            frames.append(k)
        if df_unknown_scored is not None and not df_unknown_scored.empty:
            u = df_unknown_scored.copy()
            u["source"] = "unknown"
            frames.append(u)
        if df_gen is not None and not df_gen.empty:
            g = df_gen.copy()
            g["source"] = "generated"
            frames.append(g)
        if frames:
            merged = pd.concat(frames, ignore_index=True, sort=False)
            # Unify the formula column: known/unknown use lowercase 'formula', generated uses uppercase 'Formula';
            # after concat both columns coexist -> merge into a single formula column ('formula' takes priority, falling back to 'Formula' when empty).
            if "Formula" in merged.columns:
                if "formula" not in merged.columns:
                    merged["formula"] = merged["Formula"]
                else:
                    merged["formula"] = merged["formula"].where(
                        merged["formula"].notna()
                        & (merged["formula"].astype(str).str.strip() != ""),
                        merged["Formula"],
                    )
                merged = merged.drop(columns=["Formula"])
            if "llm_backfilled" not in merged.columns:
                merged["llm_backfilled"] = ""
            merged.to_csv(
                os.path.join(out_dir, "predictions_scored.csv"),
                index=False, encoding="utf-8-sig",
            )

        if llm_details:
            pd.DataFrame(llm_details).to_csv(
                os.path.join(out_dir, "llm_details.csv"),
                index=False, encoding="utf-8-sig",
            )

        cfg_save = {
            "targets": targets,
            "weights": weights,
            "prop_stds": prop_stds,
        }
        if cfg:
            cfg_save["cfg"] = {
                k: v for k, v in cfg.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
        with open(os.path.join(out_dir, "run_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(cfg_save, f, ensure_ascii=False, indent=2, default=str)

        # Archive the structure files generated/loaded in this run (*.cif / *.extxyz / *.zip -> structures/)
        if structures_dir and os.path.isdir(structures_dir):
            struct_out = os.path.join(out_dir, "structures")
            os.makedirs(struct_out, exist_ok=True)
            _pat_files = []
            for _ext in ("*.cif", "*.extxyz"):
                _pat_files.extend(
                    glob.glob(
                        os.path.join(structures_dir, "**", _ext), recursive=True
                    )
                )
            _pat_files.extend(glob.glob(os.path.join(structures_dir, "*.zip")))
            _seen = set()
            _copied = 0
            for _src in sorted(set(_pat_files)):
                _base = os.path.basename(_src)
                _dst = os.path.join(struct_out, _base)
                _n = 1
                while _dst in _seen or os.path.exists(_dst):
                    _root, _ext = os.path.splitext(_base)
                    _dst = os.path.join(struct_out, f"{_root}_{_n}{_ext}")
                    _n += 1
                try:
                    shutil.copy2(_src, _dst)
                    _seen.add(_dst)
                    _copied += 1
                except Exception:
                    pass
            if _copied:
                print(
                    f"   [Prediction run saved] structures: {_copied} files -> {struct_out}",
                    flush=True,
                )

        # Persist the LLM Analysis (chemical interpretation + strategy suggestion)
        if llm_analysis:
            try:
                if isinstance(llm_analysis, str):
                    _md = str(llm_analysis)
                else:
                    _parts = []
                    if llm_analysis.get("explanation"):
                        _parts.append(
                            "## Why These Candidates Scored Well\n\n"
                            + str(llm_analysis["explanation"])
                        )
                    if llm_analysis.get("strategy"):
                        _parts.append(
                            "## Strategy Recommendations\n\n"
                            + str(llm_analysis["strategy"])
                        )
                    _md = "\n\n".join(_parts)
                if _md.strip():
                    with open(os.path.join(out_dir, "llm_analysis.md"), "w",
                              encoding="utf-8") as f:
                        f.write(_md)
            except Exception:
                pass

        print(f"   [Prediction run saved] {out_dir}", flush=True)
        return out_dir
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _aggregate_seed_fingerprint(df_known, mace_calc, adaptor, feat_dim):
    """Mean of top-3 seed MACE 256-dim fingerprints (L2-normalised)."""
    vecs = []
    for _, r in df_known.head(min(3, len(df_known))).iterrows():
        cp = r.get("cif_path", None)
        if not cp or not isinstance(cp, str):
            continue
        try:
            v = _get_vector_from_cif(cp, mace_calc, adaptor, feat_dim)
            vecs.append(v[:256])
        except Exception:
            pass
    if not vecs:
        return None
    arr = np.stack(vecs)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    m = np.mean(arr, axis=0)
    m /= np.linalg.norm(m) + 1e-9
    return m


def _validate_and_score(
    df_known,
    df_unknown,
    targets,
    weights,
    prop_stds,
    mace_calc=None,
    adaptor=None,
    index=None,
    material_ids=None,
    metadata=None,
    gen_structures=None,
    sim_mode="OFF",
    sim_w=0.5,
    feat_dim=None,
):
    """UI-free Step 4: forward prediction + re-ranking.
    - 4a: score unknown analog candidates (from Step 2)
    - 4b: validate generated structures, nearest-known search + optional seed-similarity re-rank
    Mirrors the in-render Step-4 logic without any streamlit widget calls.
    Returns dict with df_unknown_scored / df_gen / gen_results / generated_pass / generated_total.
    """
    mace_ok = (
        mace_calc is not None
        and adaptor is not None
        and index is not None
        and material_ids is not None
        and metadata is not None
    )
    df_unknown_scored = pd.DataFrame()
    df_gen = pd.DataFrame()
    gen_results = []
    generated_pass = 0
    generated_total = len(gen_structures) if gen_structures else 0

    # -------- 4a: unknown analog candidates --------
    if mace_ok and df_unknown is not None and not df_unknown.empty:
        unk_results = []
        for _, row in df_unknown.iterrows():
            cif_path = row.get("cif_path", None)
            if not cif_path or not isinstance(cif_path, str):
                continue
            try:
                fp = _get_vector_from_cif(cif_path, mace_calc, adaptor, feat_dim)
                neighbors = _search_by_vector(
                    fp, 1, index, material_ids, metadata,
                    db_filter=None, min_sim=0.0,
                )
                nearest_formula = (
                    neighbors[0]["formula"] if neighbors else "N/A"
                )
            except Exception:
                nearest_formula = "N/A"
            try:
                struct = Structure.from_file(cif_path)
                pred = _run_forward_prediction(struct)
            except Exception:
                pred = {}
            unk_results.append(
                {
                    "material_id": row["material_id"],
                    "Formula": row["formula"],
                    "spacegroup": row.get("spacegroup", "N/A"),
                    "similarity_to_seed": row["similarity"],
                    "nearest_known": nearest_formula,
                    "pred_er": pred.get("pred_er", None),
                    "pred_qf": pred.get("pred_qf", None),
                    "pred_tcf": pred.get("pred_tcf", None),
                    "pred_tm": pred.get("pred_tm", None),
                }
            )
            if len(unk_results) % _STEP4_CKP_INTERVAL == 0:
                _save_step4_checkpoint("unknown", unk_results)
        if unk_results:
            df_unk = pd.DataFrame(unk_results)
            has_any = (
                df_unk[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                .notna()
                .sum(axis=1)
                > 0
            )
            df_unknown_scored = df_unk[has_any].copy()
            if not df_unknown_scored.empty:
                _apply_scoring(df_unknown_scored, targets, weights, prop_stds)
                df_unknown_scored = df_unknown_scored.sort_values(
                    "score", ascending=False
                )

    # -------- 4b: generated structures --------
    if mace_ok and gen_structures:
        seed_vec_mace = None
        _do_sim = sim_mode != "OFF" and df_known is not None and not df_known.empty
        if _do_sim:
            seed_vec_mace = _aggregate_seed_fingerprint(
                df_known, mace_calc, adaptor, feat_dim
            )
        for s in gen_structures:
            try:
                struct = s if isinstance(s, Structure) else Structure.from_file(s)
                formula = struct.composition.reduced_formula
                fp = _get_vector_from_structure(struct, mace_calc, adaptor, feat_dim)
                neighbors = _search_by_vector(
                    fp, 1, index, material_ids, metadata,
                    db_filter=None, min_sim=0.0,
                )
                nearest = neighbors[0]["formula"] if neighbors else "N/A"
                nearest_sim = neighbors[0]["similarity"] if neighbors else 0.0
                seed_sim = (
                    float(np.dot(fp[:256], seed_vec_mace))
                    if seed_vec_mace is not None
                    else None
                )
            except Exception:
                struct = None
                formula = ""
                nearest, nearest_sim, seed_sim = "N/A", 0.0, None

            cached = (
                metadata[metadata["formula"] == formula]
                if (metadata is not None and formula)
                else pd.DataFrame()
            )
            if (
                formula
                and not cached.empty
                and cached[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                .notna()
                .any(axis=1)
                .any()
            ):
                crow = cached.iloc[0]
                pred = {
                    "pred_er": crow.get("pred_er"),
                    "pred_qf": crow.get("pred_qf"),
                    "pred_tcf": crow.get("pred_tcf"),
                    "pred_tm": crow.get("pred_tm"),
                    "source": "DB lookup",
                }
            else:
                try:
                    pred = _run_forward_prediction(struct) if struct is not None else {}
                    pred["source"] = "fresh prediction"
                except Exception:
                    pred = {"source": "failed"}

            gen_results.append(
                {
                    "#": len(gen_results) + 1,
                    "Formula": formula,
                    "Atoms": len(struct) if struct is not None else 0,
                    "pred_er": pred.get("pred_er", None),
                    "pred_qf": pred.get("pred_qf", None),
                    "pred_tcf": pred.get("pred_tcf", None),
                    "pred_tm": pred.get("pred_tm", None),
                    "nearest_known": nearest,
                    "nearest_sim": nearest_sim,
                    "seed_sim": seed_sim,
                }
            )
            if len(gen_results) % _STEP4_CKP_INTERVAL == 0:
                _save_step4_checkpoint("gen", gen_results)

        if gen_results:
            df_gen = pd.DataFrame(gen_results)
            has_any = (
                df_gen[["pred_tm", "pred_er", "pred_qf", "pred_tcf"]]
                .notna()
                .sum(axis=1)
                > 0
            )
            df_gen = df_gen[has_any].copy()
            if not df_gen.empty:
                _apply_scoring(df_gen, targets, weights, prop_stds)
                if (
                    sim_mode == "Into composite score"
                    and "seed_sim" in df_gen.columns
                ):
                    _sim_vals = (
                        df_gen["seed_sim"].fillna(0.0).clip(0.0, 1.0).values
                    )
                    df_gen["score"] = (
                        df_gen["score"].values / (1.0 + sim_w * (1.0 - _sim_vals))
                    )
                df_gen = df_gen.sort_values("score", ascending=False)
                generated_pass = len(df_gen)

    return {
        "df_unknown_scored": df_unknown_scored,
        "df_gen": df_gen,
        "gen_results": gen_results,
        "generated_pass": generated_pass,
        "generated_total": generated_total,
    }


def run_full_inverse(cfg):
    """UI-free end-to-end inverse-design pipeline (CADR chat path).

    cfg keys:
      formula            : str or None (optional seed / literature anchor)
      targets            : {tm, er, qf, tcf} subset with finite float values (>=1 required)
      weights            : {tm, er, qf, tcf} optional non-negative; defaults to equal split
      mattergen          : bool (default False — avoids long blocking in chat)
      mg_num             : int (default 8) max structures to keep (batch size is auto)
      mg_bs              : int or None (default None) legacy batch-size override;
                           None → batch size derived automatically by
                           _plan_generation, which is what the IDO UI does
      sg_mode            : "OFF" | "same" | "system" (structure guidance, default "OFF")
      mace_sim_mode      : "OFF" | "Reference (display only)" | "Into composite score"
      w_sim              : float in [0,1] (default 0.5)
      top_k_known        : int (default 10)
      top_k_unknown      : int (default 10)

    Returns a dict with raw DataFrames + meta; never raises (errors captured in 'error').
    """
    import traceback

    result = {
        "formula": cfg.get("formula") or None,
        "cfg": cfg,
        "df_known": pd.DataFrame(),
        "df_unknown": pd.DataFrame(),
        "df_unknown_scored": pd.DataFrame(),
        "df_gen": pd.DataFrame(),
        "gen_out_dir": None,
        "generated_total": 0,
        "generated_pass": 0,
        "targets": [],
        "weights": [],
        "prop_stds": [],
        "mace_available": False,
        "sg_applied": None,
        "error": None,
        "steps": {},
    }
    try:
        index, material_ids, metadata, feature_dict = _load_all()
        try:
            mace_calc, adaptor = _load_mace()
            mace_ok = True
        except Exception:
            mace_calc, adaptor, mace_ok = None, None, False
        result["mace_available"] = mace_ok
        feat_dim = len(feature_dict[material_ids[0]])

        # ---- targets / weights / prop_stds ----
        _KEYS = ["tm", "er", "qf", "tcf"]
        # Default constraints (same as the UI defaults, expressed as bounds only): er [25,35] / Qxf lo=50000, no upper bound / tf [-10,10] / Tm [800,1400]
        _SPEC_DEFAULTS = {
            "er": {"lo": 25.0, "hi": 35.0},
            "qf": {"lo": 50000.0, "hi": None},
            "tcf": {"lo": -10.0, "hi": 10.0},
            "tm": {"lo": 800.0, "hi": 1400.0},
        }
        tg = cfg.get("targets") or {}
        targets = []
        for k in _KEYS:
            raw = tg.get(k)
            if raw is None:
                targets.append(dict(_SPEC_DEFAULTS[k]))
            elif isinstance(raw, dict):
                targets.append(raw)
            else:
                # Legacy scalar targets are mapped to eq for compatibility
                targets.append({"op": "eq", "value": float(raw)})
        wg = cfg.get("weights") or {}
        if wg:
            weights = [float(wg.get(k, 0.0)) for k in _KEYS]
        else:
            n = sum(1 for k in _KEYS if tg.get(k) is not None)
            weights = (
                [1.0 / n if tg.get(k) is not None else 0.0 for k in _KEYS]
                if n > 0
                else [0.25] * 4
            )
        prop_stds = [
            metadata["pred_tm"].std() or 1.0,
            metadata["pred_er"].std() or 1.0,
            metadata["pred_qf"].std() or 1.0,
            metadata["pred_tcf"].std() or 1.0,
        ]
        result["targets"] = targets
        result["weights"] = weights
        result["prop_stds"] = prop_stds

        # ---- Step 1 ----
        top_k_known = int(cfg.get("top_k_known", 10))
        df_known, _ = _step1_score(
            metadata, targets, weights, prop_stds, top_k=top_k_known
        )
        _llm_details_all = []
        if not df_known.empty:
            # Enhancement 2: LLM backfill of missing properties for Step1 top-N (fill as many as are missing; degrade gracefully without blocking)
            df_known, _n_bk = _backfill_llm_props(
                df_known, collect=_llm_details_all
            )
            if _n_bk:
                _apply_scoring(df_known, targets, weights, prop_stds)
                df_known = df_known.sort_values(
                    "score", ascending=False
                ).reset_index(drop=True)
        result["df_known"] = df_known

        # ---- Step 2 ----
        df_unknown = pd.DataFrame()
        top_k_unknown = int(cfg.get("top_k_unknown", 10))
        if not df_known.empty and top_k_unknown > 0:
            seed_ids = df_known.head(3)["material_id"].tolist()
            try:
                df_unknown = _step2_recommend_unknown(
                    metadata,
                    feature_dict,
                    material_ids,
                    seed_ids=seed_ids,
                    top_k=top_k_unknown,
                )
            except Exception as e:
                result["steps"]["step2_error"] = str(e)
            if not df_unknown.empty:
                # Enhancement 2: Step2 likewise backfills missing properties for top-N and marks the LLM source
                df_unknown, _n_bk2 = _backfill_llm_props(
                    df_unknown, collect=_llm_details_all
                )
            result["df_unknown"] = df_unknown

        # ---- Step 3 (optional MatterGen) ----
        gen_structures = None
        gen_out_dir = None
        if cfg.get("mattergen") and not df_known.empty:
            prior, summary = _extract_composition_prior(
                metadata, df_known, df_unknown, n_sample=15
            )
            allowed_for_gen = list(summary["top_elements"]) if prior else None

            # Configurable anion base (absent → historical O/F/N/S/Cl default)
            anion_policy = cfg.get("anion_policy", ANION_POLICY_DEFAULT)
            anion_whitelist = cfg.get("anion_whitelist", None)

            sg_cond = None
            sg_mode = cfg.get("sg_mode", "OFF")
            if sg_mode in ("same", "system") and not df_known.empty:
                seed_sg = df_known.iloc[0].get("spacegroup", None)
                sg_cond = _sg_relax(
                    seed_sg, mode="same" if sg_mode == "same" else "system"
                )
            result["sg_applied"] = sg_cond

            structures, out_dir = _run_mattergen(
                num_structures=int(cfg.get("mg_num", 8)),
                # None → auto batch size; legacy explicit mg_bs is still honoured
                batch_size=(
                    int(cfg["mg_bs"]) if cfg.get("mg_bs") not in (None, "", 0) else None
                ),
                output_subdir=(
                    "chat_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                ),
                allowed_elements=allowed_for_gen,
                anion_policy=anion_policy,
                anion_whitelist=anion_whitelist,
                space_group_cond=sg_cond,
            )
            gen_structures = structures
            gen_out_dir = out_dir
            result["gen_out_dir"] = gen_out_dir
        result["generated_total"] = len(gen_structures) if gen_structures else 0

        # ---- Step 4 ----
        res4 = _validate_and_score(
            df_known,
            df_unknown,
            targets,
            weights,
            prop_stds,
            mace_calc=mace_calc,
            adaptor=adaptor,
            index=index,
            material_ids=material_ids,
            metadata=metadata,
            gen_structures=gen_structures,
            sim_mode=cfg.get("mace_sim_mode", "OFF"),
            sim_w=float(cfg.get("w_sim", 0.5)),
            feat_dim=feat_dim,
        )
        result["df_unknown_scored"] = res4["df_unknown_scored"]
        result["df_gen"] = res4["df_gen"]
        result["generated_pass"] = res4["generated_pass"]
        result["steps"] = {
            "top_k_known": top_k_known,
            "top_k_unknown": top_k_unknown,
            "mattergen": bool(cfg.get("mattergen")),
            "n_generated": res4["generated_total"],
            "n_scored": res4["generated_pass"],
            "mace_available": mace_ok,
        }

        # Enhancement 3: auto-save prediction results (following the gen_* pattern)
        _r4u = res4.get("df_unknown_scored")
        _r4g = res4.get("df_gen")
        _llm_analysis = None
        try:
            _llm_analysis = {
                "explanation": _explain_candidates(
                    df_known, targets, weights, prop_stds,
                    source_label="known", n_explain=min(5, len(df_known)),
                    gen_df=res4.get("df_gen"),
                ),
                "strategy": _strategy_suggestion(
                    targets, weights, prop_stds,
                    known_count=len(df_known),
                    unknown_count=(0 if _r4u is None else len(_r4u)),
                    generated_count=res4["generated_total"],
                    generated_pass_count=res4["generated_pass"],
                    total_known=int(
                        (
                            ~metadata[
                                ["pred_tm", "pred_er", "pred_qf", "pred_tcf"]
                            ]
                            .isna()
                            .all(axis=1)
                        ).sum()
                    ),
                    gen_df=res4.get("df_gen"),
                ),
            }
        except Exception:
            _llm_analysis = None
        result["prediction_save_dir"] = _save_prediction_run(
            targets,
            weights,
            prop_stds,
            df_known=df_known,
            df_unknown_scored=(
                _r4u if _r4u is not None and not _r4u.empty else None
            ),
            df_gen=(_r4g if _r4g is not None and not _r4g.empty else None),
            llm_details=_llm_details_all,
            structures_dir=gen_out_dir,
            llm_analysis=_llm_analysis,
            cfg={
                "top_k_known": top_k_known,
                "top_k_unknown": top_k_unknown,
                "mattergen": bool(cfg.get("mattergen")),
                "mode": "chat",
            },
        )
    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"

    return result
