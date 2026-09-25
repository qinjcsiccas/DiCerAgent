# -*- coding: utf-8 -*-
"""
HEC_ThermalConductivity (Cornell2023 高熵陶瓷热导率) 特征计算脚本
用于插件转换：给定化学式 + 温度，输出最终模型（RF/XGB/KRR）所需的特征向量。

特征管线（复刻 data_process.ipynb + magpie 特征定义）：
1. 化学式解析 -> 元素与摩尔系数
2. 22 项 magpie 元素属性 -> mean/maxdiff/dev/max/min/most 六统计量（132 个）
3. 附加特征：NComp（组分数）、temperature
4. 筛选（VarianceThreshold 0.10 + 相关性 0.90）-> 最终特征子集

已验证：165/166 唯一组成与 full_data.csv 全量特征一致（唯一差异为原数据脏行）。
"""
import os
import re
import numpy as np
import pandas as pd

# ---------------- 常量表 ----------------
_CONST_DIR = os.path.dirname(os.path.abspath(__file__))
_MAGPIE_CSV = os.path.join(_CONST_DIR, 'magpie_element_properties.csv')

# 最终模型特征子集（对应筛选后 50 特征，按 data_process.ipynb CELL 6 复算得到）
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

# ---------------- 化学式解析（复刻 notebook decompose_formula_II，容忍空格，支持括号块） ----------------
def _expand_brackets(formula):
    """展开简单括号块：(A0.2B0.2)2C1O3 -> A0.4B0.4C1O3。仅支持一对括号。"""
    m = re.search(r'\(([^()]*)\)([\d.]+)?', formula)
    if not m:
        return formula
    inner, mult = m.group(1), m.group(2)
    mult = float(mult) if mult else 1.0
    # inner 内每段 元素+系数 同乘 mult
    parts = re.findall(r'([A-Z][a-z]?)([\d.]+)?', inner)
    expanded = ''
    for ele, num in parts:
        n = float(num) if num else 1.0
        n *= mult
        expanded += f"{ele}{n:g}"
    return formula[:m.start()] + expanded + formula[m.end():]


def decompose_formula(formula):
    """返回 (元素列表, 摩尔系数列表)。支持 (A0.2B0.2...)2C1O3 括号公式与空格。"""
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


# ---------------- magpie 六统计量 ----------------
def magpie_stats(values, weights):
    """magpie 六统计量：mean / maxdiff / dev(加权平均绝对偏差) / max / min / most(权重最大元素属性值,并列取平均)"""
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


# ---------------- 属性表加载 ----------------
_PROP_TABLE = None
_ATTRS = None


def _load_prop_table():
    global _PROP_TABLE, _ATTRS
    if _PROP_TABLE is not None:
        return
    if not os.path.exists(_MAGPIE_CSV):
        raise FileNotFoundError(f"Constant table not found: {_MAGPIE_CSV}")
    _PROP_TABLE = pd.read_csv(_MAGPIE_CSV)
    _PROP_TABLE['Symbol'] = _PROP_TABLE['Symbol'].str.strip()
    _ATTRS = [c for c in _PROP_TABLE.columns if c not in ('Symbol', 'OxidationStates')]
    _PROP_TABLE.set_index('Symbol', inplace=True)


def _ele_prop(ele):
    if ele not in _PROP_TABLE.index:
        raise KeyError(f"Missing element property: ['{ele}']")
    return _PROP_TABLE.loc[ele]


# ---------------- 特征计算 ----------------
def calculate(formula, temperature=298.0):
    """主入口：输入化学式与温度(K)，返回最终模型特征 dict。"""
    _load_prop_table()
    eles, nums = decompose_formula(formula)
    if not eles:
        raise ValueError(f"Cannot parse formula: {formula}")
    total = sum(nums)
    weights = [n / total for n in nums]

    feats = {'temperature': float(temperature), 'NComp': len(set(eles))}

    # 22 项属性六统计量
    for attr in _ATTRS:
        try:
            vals = [float(_ele_prop(e)[attr]) for e in eles]
        except (TypeError, ValueError) as ex:
            raise ValueError(f"Property {attr} has missing values and cannot be computed (may contain superheavy elements / noble gases)") from ex
        mean, maxdiff, dev, vmax, vmin, vmost = magpie_stats(vals, weights)
        feats[f'mean_{attr}'] = mean
        feats[f'maxdiff_{attr}'] = maxdiff
        feats[f'dev_{attr}'] = dev
        feats[f'max_{attr}'] = vmax
        feats[f'min_{attr}'] = vmin
        feats[f'most_{attr}'] = vmost

    # CanFormIonic（magpie OxidationStateGuesser 官方逻辑）：
    # 对每个元素取其可能氧化态列表，枚举全部组合，存在电荷平衡组合(Σ state_i*frac_i 绝对值<1E-6)则为 1
    if len(eles) == 1:
        feats['CanFormIonic'] = 0
    else:
        states = []
        ok = True
        for e in eles:
            os_ = _ele_prop(e)['OxidationStates']
            if not isinstance(os_, str) or not os_.strip():
                ok = False
                break
            states.append([float(x) for x in os_.split()])
        if not ok:
            feats['CanFormIonic'] = 0
        else:
            import itertools
            found = False
            for combo in itertools.product(*states):
                if abs(np.dot(combo, weights)) < 1E-6:
                    found = True
                    break
            feats['CanFormIonic'] = 1 if found else 0

    # 筛选最终特征子集
    out = {k: feats[k] for k in FINAL_FEATURES}
    return out


# ---------------- 直接运行示例 ----------------
if __name__ == '__main__':
    demo = ['(Y0.2Gd0.2Er0.2Yb0.2Lu0.2)2Zr2O7', 'Hf0.2Zr0.2Ta0.2Nb0.2Ti0.2C', 'Y4Al2O9']
    for f in demo:
        try:
            feat = calculate(f, temperature=1000.0)
            print(f"\n{f} @1000K -> {len(feat)} features")
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in list(feat.items())[:8]})
            print("  ...")
        except Exception as ex:
            print(f"\n{f} -> failed: {ex}")
