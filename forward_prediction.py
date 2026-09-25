import sys
import os
import re
import json
import requests

from pymatgen.core import Composition, Structure

import config
from data_loader import DataLoader
from features import FeatureCalculator
from get_cod_cif import CODRester
from plugins import load_all_plugins, global_registry
from structure_classifier import determine_structure_type

try:
    from prediction.enhanced_predictor import EnhancedPredictor
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False

try:
    from mp_api.client import MPRester

    MP_API_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] mp_api not available: {e}")
    MPRester = None
    MP_API_AVAILABLE = False

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class ForwardPredictionAgent:
    def __init__(self):
        self.loader = DataLoader()
        self.calculator = FeatureCalculator(self.loader)
        self.cod_rester = CODRester()

        load_all_plugins()

        print(f"\n[📦] Loaded Plugins: {len(global_registry.list_all())}")
        for p in global_registry.list_all():
            meta = p.metadata
            name = meta.name if meta else p.id
            print(f"   - {p.id}: {name} ({p.structure_type})")

        self.enhanced_predictor = None
        if _ENHANCED_AVAILABLE:
            try:
                self.enhanced_predictor = EnhancedPredictor()
                print(f"   [+] Enhanced predictor (8-method router) initialized")
            except Exception as e:
                print(f"   [!] Enhanced predictor init failed: {e}")

    def get_plugin(self, structure_type: str):
        return global_registry.get_by_structure(structure_type)

    def reload_external_plugins(self):
        from plugins import load_external_plugins

        load_external_plugins()
        print(
            f"[📦] Reloaded external plugins. Total: {len(global_registry.list_all())}"
        )

    def select_plugins_with_llm(
        self,
        formula: str,
        stype: str = "Unknown",
        has_structure: bool = True,
        property_name: str = "",
    ) -> list:
        all_plugins = global_registry.list_all()
        if not all_plugins:
            return []

        feasible_plugins = []
        for p in all_plugins:
            if not p.enabled:
                continue
            meta = p.metadata
            plugin_stype = p.structure_type.upper() if p.structure_type else "GENERAL"
            target_stype = stype.upper()

            if not has_structure and plugin_stype != "GENERAL":
                continue
            if plugin_stype != "GENERAL" and target_stype != "UNKNOWN":
                if plugin_stype != target_stype:
                    continue
            feasible_plugins.append(p)

        if not feasible_plugins:
            return []

        plugin_descriptions = []
        for p in feasible_plugins:
            meta = p.metadata
            supported = meta.supported_structures if meta else []
            desc = f"- {p.id}: {meta.name if meta else 'N/A'}, type={p.structure_type}, works_for={supported}"
            if hasattr(p.predictor, "get_required_features"):
                try:
                    feats = p.predictor.get_required_features()
                    desc += f", required_features={feats[:8]}..."
                except:
                    pass
            plugin_descriptions.append(desc)

        prompt = f"""You are a Materials Science AI router. 
    Target Material: {formula}
    Identified System: {stype}
    Structure Data Available: {has_structure}
    Goal: Predict {property_name if property_name else "dielectric properties (er, qxf, tcf)"}
    
    From the feasible plugins below, select the most relevant one(s).
    Strict Rules:
    1. If the material fits a specific system (e.g., ABO3), prioritize that specific plugin.
    2. If multiple plugins are relevant, you can select more than one.
    3. Return ONLY a JSON list of IDs.
    
    Feasible Plugins:
    {chr(10).join(plugin_descriptions)}
    
    Output Example: ["plugin_id_1", "plugin_id_2"]"""

        try:
            headers = {
                "Authorization": f"Bearer {config.DS_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.DS_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a materials science expert. Output ONLY JSON list.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post(
                f"{config.DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )

            if resp.status_code == 200:
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    json_str = match.group().strip()
                    try:
                        selected_ids = json.loads(json_str)
                    except json.JSONDecodeError:
                        selected_ids = re.findall(r'"([^"]+)"', json_str)

                    if isinstance(selected_ids, list) and selected_ids:
                        print(f"   [AI Router] Selected: {selected_ids}")
                        ai_selected = [
                            p for p in feasible_plugins if p.id in selected_ids
                        ]
                        general_plugins = [
                            p
                            for p in feasible_plugins
                            if p.structure_type.upper() == "GENERAL"
                        ]
                        same_type_plugins = [
                            p
                            for p in feasible_plugins
                            if p.structure_type.upper() == target_stype
                        ]
                        final_selection = (
                            ai_selected + general_plugins + same_type_plugins
                        )
                        seen = set()
                        unique_selection = []
                        for p in final_selection:
                            if p.id not in seen:
                                seen.add(p.id)
                                unique_selection.append(p)
                        if unique_selection:
                            return unique_selection

        except Exception as e:
            print(
                f"   [!] AI Routing failed, falling back to rule-based selection. Error: {e}"
            )

        fallback = [
            p for p in feasible_plugins if p.structure_type.upper() == stype.upper()
        ]
        if not fallback:
            fallback = [p for p in feasible_plugins if p.structure_type == "general"]

        return fallback

    def normalize_formula(self, formula):
        try:

            def fraction_to_decimal(match):
                return str(float(match.group(1)) / float(match.group(2)))

            formula_decimal = re.sub(r"(\d+)/(\d+)", fraction_to_decimal, formula)
            comp = Composition(formula_decimal)
            integer_formula, _ = comp.get_integer_formula_and_factor()
            return str(integer_formula).replace(" ", "")
        except:
            return None

    def fetch_structure(self, query, db_pref=None):
        if not query:
            return None, None
        clean_query = query.strip()

        if os.path.exists(clean_query) and os.path.isfile(clean_query):
            try:
                return Structure.from_file(
                    clean_query
                ), f"Local-{os.path.basename(clean_query)}"
            except:
                return None, None

        if db_pref in [None, "mp"] and "cod-" not in clean_query.lower():
            if not MP_API_AVAILABLE or MPRester is None:
                print("[WARN] MP API not available, skipping Materials Project search")
            else:
                mp_failed = False
                try:
                    with MPRester(config.MP_API_KEY) as mpr:

                        def _decode_alpha_id(raw_id):
                            """Decode alpha-encoded MPID (e.g. mp-aaacktyk) to legacy (mp-1103190)."""
                            ALPHA = "abcdefghijklmnopqrstuvwxyz"
                            prefix, _, encoded = raw_id.partition("-")
                            if not encoded:
                                return raw_id
                            # Already a numeric ID
                            if encoded.isdigit():
                                return raw_id
                            # Decode base-26 alpha to integer
                            val = 0
                            for ch in encoded:
                                val = val * 26 + ALPHA.index(ch)
                            return f"{prefix}-{val}"

                        def _do_mp_search(search_str):
                            """Search MP; alpha-ID strings skip pydantic directly to raw HTTP + alpha decode."""
                            # Pre-check: alpha IDs (mp-xxx where xxx contains letters) skip pydantic
                            _is_alpha_id = bool(re.match(r"^mp-[a-z]+$", search_str))
                            if not _is_alpha_id:
                                try:
                                    docs = mpr.materials.summary.search(
                                        material_ids=[search_str]
                                        if "mp-" in search_str
                                        else None,
                                        formula=search_str if "mp-" not in search_str else None,
                                        fields=[
                                            "structure",
                                            "material_id",
                                            "energy_above_hull",
                                        ],
                                    )
                                    return docs
                                except Exception:
                                    pass  # fall through to raw HTTP

                            # --- Raw HTTP fallback with alpha-ID decoding ---
                            from pymatgen.core import Structure
                            from types import SimpleNamespace

                            url = "https://api.materialsproject.org/materials/summary/"
                            params = (
                                {"material_ids": search_str}
                                if "mp-" in search_str
                                else {"formula": search_str}
                            )
                            params["_fields"] = (
                                "structure,material_id,energy_above_hull"
                            )
                            try:
                                resp = requests.get(
                                    url,
                                    params=params,
                                    headers={"X-API-KEY": config.MP_API_KEY},
                                    timeout=30,
                                )
                                resp.raise_for_status()
                            except Exception as e2:
                                print(f"   ⚠️ Raw HTTP fallback failed: {e2}")
                                return []

                            data = resp.json()
                            results = []
                            skipped = 0
                            for doc in data.get("data", []):
                                raw_mid = str(doc.get("material_id", ""))
                                # Try to decode alpha ID; skip if doesn't look like mp-xxx
                                mid = _decode_alpha_id(raw_mid)
                                if not re.match(r"^mp-\d+$", mid):
                                    skipped += 1
                                    continue
                                try:
                                    struct = Structure.from_dict(doc["structure"])
                                except Exception:
                                    skipped += 1
                                    continue
                                hull = doc.get("energy_above_hull", 999.0)
                                results.append(
                                    SimpleNamespace(
                                        structure=struct,
                                        material_id=mid,
                                        energy_above_hull=hull,
                                    )
                                )
                            if skipped:
                                print(
                                    f"   ⚠️ Filtered {skipped} invalid entries (raw HTTP)"
                                )
                            return results

                        docs = None
                        try:
                            print(f"   🔎 Searching MP for: {clean_query} ...")
                            docs = _do_mp_search(clean_query)
                        except Exception as e:
                            print(f"   ⚠️ MP search failed for '{clean_query}': {e}")

                        if not docs and "mp-" not in clean_query:
                            norm_query = self.normalize_formula(clean_query)
                            if norm_query and norm_query != clean_query:
                                try:
                                    print(f"   🔎 Retrying MP with normalized: {norm_query} ...")
                                    docs = _do_mp_search(norm_query)
                                except Exception as e:
                                    print(f"   ⚠️ MP search failed for '{norm_query}': {e}")

                        if docs:
                            ground_states = [
                                d for d in docs if d.energy_above_hull < 0.0001
                            ]
                            targets = ground_states if ground_states else docs
                            targets.sort(key=lambda x: x.energy_above_hull)
                            print(
                                f"   ✅ Found in MP: {targets[0].material_id} (E_hull: {targets[0].energy_above_hull:.4f})"
                            )
                            return targets[0].structure, targets[0].material_id
                        else:
                            print(f"   ⚠️ MP returned 0 results for '{clean_query}', falling back to COD")
                except Exception as e:
                    mp_failed = True
                    print(f"   ⚠️ MP API unavailable ({e}), falling back to COD")

        if (db_pref in [None, "cod"]) and "mp-" not in clean_query:
            struct, cod_id = self.cod_rester.get_structure_interactive(clean_query)
            if struct:
                return struct, cod_id

        return None, None

    def check_structure_type(self, structure):
        try:
            comp = structure.composition
            el_dict = comp.get_el_amt_dict()
            if "O" not in el_dict or len(el_dict) < 3:
                return "Unknown"
            o_amt = el_dict["O"]

            factor_3 = 3.0 / o_amt
            cations_3 = {k: v * factor_3 for k, v in el_dict.items() if k != "O"}
            if (
                1.95 < sum(cations_3.values()) < 2.05
                and 0.85 < max(cations_3.values()) < 1.15
            ):
                return "ABO3"

            factor_4 = 4.0 / o_amt
            cations_4 = {k: v * factor_4 for k, v in el_dict.items() if k != "O"}
            if (
                1.95 < sum(cations_4.values()) < 2.05
                and 0.85 < max(cations_4.values()) < 1.15
            ):
                return "ABO4"

            return "Unknown"
        except:
            return "Unknown"

    def analyze_single(self, query, manual_feats=None, db_pref=None,
                       user_context=None, enabled_methods=None,
                       external_struct=None):
        if manual_feats is None:
            manual_feats = {}

        result_data = {
            "Input_Query": query,
            "Material_ID": "Manual",
            "Structure_Type": "Unknown",
            "Formula": query,
            "er_mean": None,
            "er_dev": None,
            "qxf_mean": None,
            "qxf_dev": None,
            "tcf_mean": None,
            "tcf_dev": None,
            "tm_mean": None,
            "tm_dev": None,
            "Error": None,
        }

        # external_struct: structure object passed in directly from outside (e.g. generated by MatterGen);
        # skips the database query and is used directly for feature calculation and ML prediction
        has_external = external_struct is not None
        if has_external:
            struct = external_struct
            mid = "Generated"
            has_structure = True
            formula = struct.composition.reduced_formula
            # Extract space group + lattice information from the structure, and combine with the LLM to determine the structure type
            try:
                sg = struct.get_space_group_info()[0] if hasattr(struct, 'get_space_group_info') else "Unknown"
                lattice = struct.lattice
                struct_info = f"Space group: {sg}, a={lattice.a:.3f} b={lattice.b:.3f} c={lattice.c:.3f}, α={lattice.alpha:.1f}° β={lattice.beta:.1f}° γ={lattice.gamma:.1f}°"
            except Exception:
                struct_info = ""
            stype = determine_structure_type(formula, struct_info, use_llm=True)
        else:
            struct, mid = self.fetch_structure(query, db_pref)
            has_structure = struct is not None
            stype = determine_structure_type(query, use_llm=True)

        result_data["Structure_Type"] = stype

        if has_structure:
            result_data["Material_ID"] = mid
            result_data["Formula"] = struct.composition.reduced_formula
            result_data["_structure_obj"] = struct
            if has_external:
                print(f"   [*] Using external structure: {formula} | System: {stype}")
            else:
                print(f"   [*] Structure found: {mid} | System: {stype}")
        else:
            print(
                f"   [!] Structure not found. Switching to Composition-only mode | System: {stype}"
            )

        selected_plugins = self.select_plugins_with_llm(
            formula=query, stype=stype, has_structure=has_structure
        )

        if not selected_plugins:
            result_data["Error"] = (
                "No suitable prediction models found for this material."
            )
            return result_data

        try:
            if has_structure:
                feats, final_struct = self.calculator.calc_basic_features(struct)
            else:
                try:
                    comp = Composition(query)
                    feats = {"Mass": comp.weight, "num_atoms": comp.num_atoms}
                    final_struct = None
                except:
                    feats = {}
                    final_struct = None

            feats.update(manual_feats)

            for plugin in selected_plugins:
                if not plugin.enabled:
                    continue

                print(f"   [+] Running Plugin: {plugin.id}")

                spec_feats = feats.copy()
                if plugin.feature_calculator:
                    try:
                        calc_res = plugin.feature_calculator.calculate(
                            final_struct, feats
                        )
                        if isinstance(calc_res, dict):
                            spec_feats.update(calc_res)
                    except Exception as fe:
                        print(
                            f"      [!] Feature calculation error in {plugin.id}: {fe}"
                        )

                if plugin.predictor:
                    try:
                        prediction = plugin.predictor.predict(spec_feats)
                        if isinstance(prediction, dict):
                            for k, v in prediction.items():
                                if v is not None:
                                    result_data[k] = v
                    except Exception as pe:
                        print(f"      [!] Prediction error in {plugin.id}: {pe}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            result_data["Error"] = f"Critical error in analysis: {str(e)}"

        # --- Enhanced prediction (LLM + Lit route, 8-method) ---
        if self.enhanced_predictor is not None:
            try:
                target_props = []
                ml_predictions = {}
                # Dynamically build the property mapping: all _mean keys -> enhanced predictor target names
                # Note: val may be None (ML failed without CIF), but it must still be passed to the enhanced predictor to fall back to LLM-Full
                _SPECIAL_MAP = {"qxf": "qf"}
                for key, val in result_data.items():
                    if key.endswith("_mean") and not key.startswith("_"):
                        base = key[:-5].lower()  # strip the trailing _mean and lowercase
                        prop_name = _SPECIAL_MAP.get(base, base)
                        target_props.append(prop_name)
                        ml_predictions[prop_name] = val if val is not None else None

                if target_props:
                    has_struct = result_data.get("_structure_obj") is not None
                    enhanced = self.enhanced_predictor.predict(
                        formula=result_data['Formula'],
                        target_properties=target_props,
                        user_context=user_context,
                        has_structure=has_struct,
                        ml_predictions=ml_predictions,
                        enabled_methods=enabled_methods,
                    )
                    if enhanced:
                        result_data['enhanced_results'] = enhanced.get('results', [])
                        result_data['route_info'] = enhanced.get('route_info', {})
                        print(f"   [✓] Enhanced prediction completed: {len(enhanced.get('results',[]))} routes")
            except Exception as ee:
                print(f"   [!] Enhanced prediction error: {ee}")
                result_data['_enhanced_error'] = str(ee)

        return result_data
