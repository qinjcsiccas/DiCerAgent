import streamlit as st
try:
    from stmol import showmol
except ImportError:
    showmol = None
try:
    import py3Dmol
except ImportError:
    py3Dmol = None
from pymatgen.core.periodic_table import Element

# === Jmol standard color palette ===
JMOL_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000",
    "S": "#FFFF30", "Cl": "#1FF01F", "K": "#8F40D4", "Ca": "#3DFF00", "Sc": "#E6E6E6",
    "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633",
    "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
    "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1",
    "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9",
    "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C", "Pd": "#006985",
    "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5",
    "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F", "Ba": "#00C900",
    "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7", "Pm": "#A3FFC7",
    "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7", "Dy": "#1FFFC7",
    "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24",
    "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696",
    "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0", "Tl": "#A6544D",
    "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45", "Rn": "#428296"
}

def get_element_color(symbol):
    return JMOL_COLORS.get(symbol, "#CCCCCC") 

def render_structure_card(structure_obj, key_suffix=""):
    """
    Receive a pymatgen structure object and render an interactive 3D card
    """
    if not structure_obj:
        st.error("No structure data available.")
        return

    if showmol is None or py3Dmol is None:
        st.warning("⚠️ 3D visualization requires 'stmol' and 'py3Dmol' packages. Install with: pip install stmol py3Dmol")
        st.info("Showing structure data instead:")
        lat = structure_obj.lattice
        st.markdown(f"""
        - **Formula**: {structure_obj.composition.reduced_formula}
        - **a**: {lat.a:.3f} Å, **b**: {lat.b:.3f} Å, **c**: {lat.c:.3f} Å
        - **α**: {lat.alpha:.1f}°, **β**: {lat.beta:.1f}°, **γ**: {lat.gamma:.1f}°
        - **Volume**: {lat.volume:.2f} Å³
        """)
        return

    # 1. Layout control: Reset button
    c_label, c_btn = st.columns([4, 1])
    with c_label:
        st.caption("🖱️ Scroll to Zoom | Drag to Rotate")
    with c_btn:
        # Clicking this button triggers a Rerun, re-executing the code below to reset the view
        if st.button("🔄 Reset", key=f"rst_btn_{key_suffix}", help="Reset Camera View"):
            pass 

    # 2. Prepare CIF data
    cif_str = structure_obj.to(fmt="cif")
    
    # 3. Create viewer
    view = py3Dmol.view(width=600, height=400)
    view.addModel(cif_str, 'cif')
    
    # Force white background
    view.setBackgroundColor('#ffffff') 
    # ------------------------------------

    # 4. Set style (Ball & Stick)
    view.setStyle({
        'sphere': {'scale': 0.28, 'colorscheme': 'Jmol'}, 
        'stick': {'radius': 0.15, 'colorscheme': 'Jmol'}
    })
    
    # 5. Add unit cell box
    view.addUnitCell()
    
    # 6. Set view
    view.zoomTo() 
    
    # 7. Render
    showmol(view, height=400, width=600)

    # 8. === Generate Legend ===
    # Core fix: use list concatenation to build a clean single-line HTML string, avoiding Markdown parsing errors
    elements = sorted([e.symbol for e in structure_obj.composition.elements])
    
    legend_items = []
    for el in elements:
        color = get_element_color(el)
        # HTML for each legend item (single line)
        item_str = (
            f"<div style='display: flex; align-items: center; gap: 6px; margin-right: 15px;'>"
            f"<div style='width: 12px; height: 12px; background-color: {color}; border-radius: 50%; "
            f"border: 1px solid #999; box-shadow: 1px 1px 2px rgba(0,0,0,0.1);'></div>"
            f"<span style='font-weight: 500; font-size: 14px; color: #444; font-family: sans-serif;'>{el}</span>"
            f"</div>"
        )
        legend_items.append(item_str)
    
    # Concatenate all items
    all_items_str = "".join(legend_items)
        
    final_legend_html = (
        f"<div style='display: flex; flex-wrap: wrap; justify-content: center; padding: 8px; "
        f"background-color: transparent !important; " # must be transparent
        f"margin-top: -5px; border: none !important;'>" # must have no border
        f"{all_items_str}"
        f"</div>"
    )
    
    # Render (unsafe_allow_html=True is required)
    st.markdown(final_legend_html, unsafe_allow_html=True)

    # 9. Lattice parameter display
    with st.expander("📏 Lattice Parameters"):
        lat = structure_obj.lattice
        st.markdown(f"""
        - **a**: {lat.a:.3f} Å, **b**: {lat.b:.3f} Å, **c**: {lat.c:.3f} Å
        - **α**: {lat.alpha:.1f}°, **β**: {lat.beta:.1f}°, **γ**: {lat.gamma:.1f}°
        - **Volume**: {lat.volume:.2f} Å³
        """)
