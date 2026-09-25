import numpy as np
import warnings
from pymatgen.core import Element, Species
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from matminer.featurizers.structure import StructuralHeterogeneity

class FeatureCalculator:
    def __init__(self, data_loader):
        self.data = data_loader
        self.het_feat = StructuralHeterogeneity()
        self.cnn = CrystalNN(x_diff_weight=0, porous_adjustment=False)
        self.bv_analyzer = BVAnalyzer()

    # =========================================================================
    # Part 1: Original logic (port of v1.0) - specifically handles ordered / undoped structures
    # =========================================================================
    def get_valences_ordered(self, structure):
        try:
            return self.bv_analyzer.get_valences(structure)
        except Exception:
            pass
        # After BV fails, try several fallback strategies
        guesses = structure.composition.oxi_state_guesses()
        if guesses:
            return [guesses[0].get(s.specie.symbol, 0) for s in structure]
        # Last resort: assign values by common oxidation states
        common_oxi = {
            "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
            "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
            "B": 3, "Al": 3, "Ga": 3, "In": 3, "Tl": 1,
            "C": 4, "Si": 4, "Ge": 4, "Sn": 2, "Pb": 2,
            "N": -3, "P": -3, "As": -3, "Sb": -3, "Bi": 3,
            "O": -2, "S": -2, "Se": -2, "Te": -2,
            "F": -1, "Cl": -1, "Br": -1, "I": -1,
            "Sc": 3, "Y": 3, "La": 3,
            "Ti": 4, "Zr": 4, "Hf": 4,
            "V": 5, "Nb": 5, "Ta": 5,
            "Cr": 3, "Mo": 6, "W": 6,
            "Mn": 2, "Tc": 7, "Re": 7,
            "Fe": 2, "Ru": 4, "Os": 8,
            "Co": 2, "Rh": 3, "Ir": 4,
            "Ni": 2, "Pd": 2, "Pt": 2,
            "Cu": 1, "Ag": 1, "Au": 1,
            "Zn": 2, "Cd": 2, "Hg": 2,
            "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 2,
            "Eu": 2, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3,
            "Er": 3, "Tm": 3, "Yb": 2, "Lu": 3,
            "Th": 4, "U": 6, "Np": 5, "Pu": 4,
        }
        return [
            common_oxi.get(s.specie.symbol, 0)
            for s in structure
        ]

    def get_zhang_en_ordered(self, element_symbol, valence):
        val_int = int(round(valence))
        return self.data.zhang_en_dict.get((element_symbol, val_int), Element(element_symbol).X)

    def get_polar_smart_ordered(self, element, valence, target_cn):
        df = self.data.mlr_df
        row = df[(df['Element']==element) & (df['Valence']==valence) & (df['CN']==target_cn)]
        if not row.empty: return row.iloc[0]['Polarizability']
        sub = df[(df['Element']==element) & (df['Valence']==valence)]
        if not sub.empty: return sub.iloc[0]['Polarizability']
        return 0.0

    def get_robust_cn_and_nn_ordered(self, structure, site_idx):
        site = structure[site_idx]
        try: nn_info = self.cnn.get_nn_info(structure, site_idx); cn = len(nn_info)
        except: nn_info=[]; cn=0
        fallback=False
        if cn==0: fallback=True
        target_el = site.specie.symbol
        if target_el in ['V','P','Si','W','Mo','Nb','Ta'] and cn not in [4,6]: fallback=True
        if fallback:
            dists = [site.distance(oxygen_site) for oxygen_site in structure if oxygen_site.specie.symbol=='O']
            if dists:
                cut = min(dists)*1.25
                cn = sum(1 for d in dists if d<=cut)
        return cn, nn_info

    def get_specific_shannon_radius_ordered(self, element_symbol, valence, target_cn_str):
        try: return Species(element_symbol, int(round(valence))).get_shannon_radius(target_cn_str)
        except: return 0.0

    def calc_features_ordered(self, structure):
        """Core calculation logic of the 1.0 version (kept verbatim)"""
        reduced_comp, z_factor = structure.composition.get_reduced_composition_and_factor()
        mass = structure.composition.weight / z_factor
        va = structure.volume / structure.num_sites
        asd = np.std(structure.lattice.angles)
        try: nbr_dist_var = self.het_feat.featurize(structure)[6]
        except: nbr_dist_var = 0
        
        total_polar = 0.0
        all_raw_bond_lengths = []
        valences = self.get_valences_ordered(structure)
        
        for i, site in enumerate(structure):
            el = site.specie.symbol
            val = int(round(valences[i]))
            cn, nn_info = self.get_robust_cn_and_nn_ordered(structure, i)
            if cn>0 and nn_info:
                dists = [site.distance(structure[n['site_index']], jimage=n['image']) for n in nn_info]
                all_raw_bond_lengths.extend(dists)
            total_polar += self.get_polar_smart_ordered(el, val, cn)
            
        p_mlr_pv = total_polar / structure.volume if structure.volume>0 else 0
        bond_abs_dev = np.mean(np.abs(np.array(all_raw_bond_lengths)-np.mean(all_raw_bond_lengths))) if all_raw_bond_lengths else 0
        
        feats = {"Mass": mass, "Va": va, "P_MLR_pv": p_mlr_pv, "ASD": asd, "Nbr_dist_var": nbr_dist_var, "Bond_abs_dev": bond_abs_dev}
        return feats, structure

    def calc_abo4_ordered(self, structure, p_mlr_pv):
        z_factor = structure.composition.num_atoms / 6.0 
        vm = structure.volume / z_factor
        p = p_mlr_pv * vm 
        valences = self.get_valences_ordered(structure)
        cations_en = [self.get_zhang_en_ordered(s.specie.symbol, valences[i]) for i, s in enumerate(structure) if s.specie.symbol != "O"]
        en = np.mean(cations_en) if cations_en else 0
        qf_feats = {"vm": vm, "p": p, "en": en}

        try: sp = structure.get_space_group_info()[1]
        except: sp = 0
        total_bvs = 0.0; total_cvd = 0.0; cation_count = 0

        for i, site in enumerate(structure):
            if site.specie.symbol == "O": continue
            el = site.specie.symbol
            val = int(round(valences[i]))
            cation_count += 1
            
            r0 = self.data.bond_val_dict.get((el, val), 0)
            if r0 == 0: 
                candidates = [v for (e, v), r in self.data.bond_val_dict.items() if e == el]
                if candidates: r0 = self.data.bond_val_dict[(el, candidates[0])]
            if r0 > 0:
                neighbors = structure.get_neighbors(site, r=6.0)
                for entry in neighbors:
                    if hasattr(entry, "nn_distance"): dist = entry.nn_distance; neighbor = entry
                    else: neighbor = entry[0]; dist = entry[1]
                    if neighbor.specie.symbol == "O": total_bvs += np.exp((r0 - dist) / 0.37)

            nn_info = self.cnn.get_nn_info(structure, i)
            bonding_dists = [site.distance(structure[n['site_index']], jimage=n['image']) for n in nn_info if structure[n['site_index']].specie.symbol == 'O']
            bl_params = self.data.bond_len_dict.get((el, val))
            bc_params = self.data.bond_cov_dict.get(el)
            if bl_params and bc_params and bonding_dists:
                R_avg = np.mean(bonding_dists)
                if R_avg > 0:
                    S = (R_avg / bl_params['R1']) ** (-bl_params['N'])
                    fc = bc_params['a'] * (S ** bc_params['M'])
                    if S > 0: total_cvd += (fc / S)

        tcf_feats = {"sp": sp, "pm": p, "cvd": total_cvd/cation_count, "bvs": total_bvs/cation_count} if cation_count else {}
        return qf_feats, tcf_feats

    def calc_abo3_ordered(self, structure, p_mlr_pv):
        z_factor = structure.composition.num_atoms / 5.0
        m = structure.composition.weight / z_factor
        pm = p_mlr_pv * (structure.volume / z_factor)
        valences = self.get_valences_ordered(structure)
        
        cation_bond_lengths = []
        total_vi_cell = 0.0

        for i, site in enumerate(structure):
            el = site.specie.symbol
            val = int(round(valences[i]))
            
            radius = 1.40 if el == 'O' else 0.0
            if el != 'O':
                r_12 = self.get_specific_shannon_radius_ordered(el, val, "XII")
                r_6  = self.get_specific_shannon_radius_ordered(el, val, "VI")
                if r_12 > 0.90: radius = r_12
                elif r_6 > 0: radius = r_6
                else: 
                    try: radius = Species(el, val).average_ionic_radius
                    except: radius = 0.6
            total_vi_cell += (4/3) * np.pi * (radius ** 3)

            if el != "O":
                nn_info = self.cnn.get_nn_info(structure, i)
                dists = [site.distance(structure[n['site_index']], jimage=n['image']) for n in nn_info if structure[n['site_index']].specie.symbol == 'O']
                if dists: cation_bond_lengths.append(np.mean(dists))

        Vi = total_vi_cell / z_factor
        if len(cation_bond_lengths) >= 2:
            cation_bond_lengths.sort(reverse=True)
            mid = len(cation_bond_lengths) // 2
            tt = np.mean(cation_bond_lengths[:mid]) / (np.sqrt(2) * np.mean(cation_bond_lengths[mid:]))
        else: tt = 1.0

        return {"m": m, "Vi": Vi, "tt": tt, "pm": pm}


    # =========================================================================
    # Part 2: New logic (robust version) - specifically handles disordered / doped structures
    # =========================================================================
    def get_element_valences_robust(self, structure):
        val_dict = {}
        try:
            guesses = structure.composition.oxi_state_guesses()
            if guesses: val_dict = {k: int(round(v)) for k, v in guesses[0].items()}
        except: pass
        
        for el in structure.composition.elements:
            if el.symbol not in val_dict:
                try: val_dict[el.symbol] = max(el.common_oxidation_states) if el.symbol != 'O' else -2
                except: val_dict[el.symbol] = 0
        return val_dict

    def get_polar_val_robust(self, el, val, cn=None):
        df = self.data.mlr_df
        if df.empty: return 0.0
        el_df = df[df['Element'].astype(str).str.strip() == el]
        if el_df.empty: return 0.0
        
        val_df = el_df[el_df['Valence'] == val]
        if val_df.empty: return el_df['Polarizability'].mean()
        
        if cn is not None:
            cn_df = val_df[val_df['CN'] == cn]
            if not cn_df.empty: return cn_df.iloc[0]['Polarizability']
        
        return val_df['Polarizability'].mean()

    def get_weighted_property(self, site, prop_func, valence_dict, **kwargs):
        try:
            if hasattr(site.species, "as_dict"): species_dict = site.species.as_dict()
            else: species_dict = {str(site.specie): 1.0}
            
            total_prop, total_occ = 0.0, 0.0
            for el_str, occ in species_dict.items():
                el_sym = "".join([c for c in el_str if c.isalpha()])
                current_val = valence_dict.get(el_sym, 0)
                total_prop += prop_func(el_sym, current_val, **kwargs) * occ
                total_occ += occ
            return total_prop / total_occ if total_occ > 0 else 0.0
        except: return 0.0

    def get_robust_cn_and_nn_disordered(self, structure, site_idx):
        # CN acquisition optimized for disordered structures
        site = structure[site_idx]
        try: nn_info = self.cnn.get_nn_info(structure, site_idx); cn = len(nn_info)
        except: nn_info=[]; cn=0
        if cn == 0:
            dists = [site.distance(s) for j, s in enumerate(structure) if site_idx != j]
            if dists:
                cut = min(dists) * 1.4
                cn = sum(1 for d in dists if d <= cut)
                nn_info = [{'site_index': k, 'image': (0,0,0)} for k, d in enumerate(dists) if d <= cut]
        return cn, nn_info

    def calc_features_disordered(self, structure):
        # Robust calculation logic
        valence_dict = self.get_element_valences_robust(structure)
        comp = structure.composition
        reduced_comp, z_factor = comp.get_reduced_composition_and_factor()
        
        mass = comp.weight / z_factor 
        va = structure.volume / (comp.num_atoms / z_factor)
        asd = np.std(structure.lattice.angles)
        try: nbr_dist_var = self.het_feat.featurize(structure)[6]
        except: nbr_dist_var = 0
        
        total_polar = 0.0
        all_raw_bond_lengths = []
        
        for i, site in enumerate(structure):
            cn, nn_info = self.get_robust_cn_and_nn_disordered(structure, i)
            def _polar_wrapper(e, v, **kwargs):
                return self.get_polar_val_robust(e, v, cn=kwargs.get('cn'))
            
            total_polar += self.get_weighted_property(site, _polar_wrapper, valence_dict, cn=cn)
            
            if cn > 0 and nn_info:
                dists = [site.distance(structure[n['site_index']], jimage=n['image']) for n in nn_info]
                all_raw_bond_lengths.extend(dists)
            
        p_mlr_pv = total_polar / structure.volume if structure.volume>0 else 0
        bond_abs_dev = np.mean(np.abs(np.array(all_raw_bond_lengths)-np.mean(all_raw_bond_lengths))) if all_raw_bond_lengths else 0
        
        feats = {"Mass": mass, "Va": va, "P_MLR_pv": p_mlr_pv, "ASD": asd, "Nbr_dist_var": nbr_dist_var, "Bond_abs_dev": bond_abs_dev}
        return feats, structure

    def calc_abo4_disordered(self, structure, p_mlr_pv):
        z_factor = structure.composition.num_atoms / 6.0 
        vm = structure.volume / z_factor
        p = p_mlr_pv * vm 
        valence_dict = self.get_element_valences_robust(structure)
        
        cations_en = []
        for site in structure:
            is_oxygen = False
            try:
                for el, occ in site.species.items():
                    if el.symbol == "O" and occ > 0.5: is_oxygen = True
            except: 
                if site.specie.symbol == "O": is_oxygen = True
            
            if not is_oxygen:
                def _en_func(e, v, **kwargs): return self.data.zhang_en_dict.get((e, v), Element(e).X)
                cations_en.append(self.get_weighted_property(site, _en_func, valence_dict))
                
        en = np.mean(cations_en) if cations_en else 0
        qf_feats = {"vm": vm, "p": p, "en": en}
        try: sp = structure.get_space_group_info()[1]
        except: sp = 0
        return qf_feats, {"sp": sp, "pm": p, "cvd": 0, "bvs": 0} 

    def calc_abo3_disordered(self, structure, p_mlr_pv):
        # Simplified robust ABO3 calculation
        z_factor = structure.composition.num_atoms / 5.0
        m = structure.composition.weight / z_factor
        pm = p_mlr_pv * (structure.volume / z_factor)
        return {"m": m, "Vi": 0, "tt": 1.0, "pm": pm} # complex geometric features temporarily use defaults for disordered cases


    # =========================================================================
    # Part 3: Intelligent dispatcher (Dispatcher)
    # =========================================================================
    def process_structure_uniformly(self, structure):
        # Only generic handling (Niggli); do not force Order
        try: structure = structure.get_niggli_reduced_structure()
        except: pass 
        return structure

    def calc_basic_features(self, original_structure):
        structure = self.process_structure_uniformly(original_structure)
        
        # === Intelligent dispatch ===
        if structure.is_ordered:
            # print("   🔧 Mode: Ordered (Classic 1.0)")
            return self.calc_features_ordered(structure)
        else:
            # print("   🔧 Mode: Disordered (Robust VCA)")
            return self.calc_features_disordered(structure)

    def calc_abo4_features(self, structure, p_mlr_pv):
        if structure.is_ordered: return self.calc_abo4_ordered(structure, p_mlr_pv)
        else: return self.calc_abo4_disordered(structure, p_mlr_pv)

    def calc_abo3_features(self, structure, p_mlr_pv):
        if structure.is_ordered: return self.calc_abo3_ordered(structure, p_mlr_pv)
        else: return self.calc_abo3_disordered(structure, p_mlr_pv)