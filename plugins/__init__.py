"""Plugin system for DiCerAgent

Usage:
    from plugins.registry import global_registry, register_plugin
    from plugins.base import Plugin, BaseFeatureCalculator, BasePredictor

    # Get all registered plugins
    plugins = global_registry.list_all()
"""

from plugins.base import (
    BaseFeatureCalculator,
    BasePredictor,
    ModelMetadata,
    ModelType,
    Plugin,
)
from plugins.registry import global_registry, register_plugin, PluginRegistry

__all__ = [
    "BaseFeatureCalculator",
    "BasePredictor",
    "ModelMetadata",
    "ModelType",
    "Plugin",
    "global_registry",
    "register_plugin",
    "PluginRegistry",
]


def load_builtin_plugins():
    """Load all built-in plugins"""
    import os
    import importlib

    builtin_dir = os.path.join(os.path.dirname(__file__), "builtin")
    for filename in os.listdir(builtin_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = f"plugins.builtin.{filename[:-3]}"
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"[!] Failed to load {module_name}: {e}")


def load_external_plugins():
    """Load all external plugins from plugins/external/"""
    import os
    import importlib

    external_dir = os.path.join(os.path.dirname(__file__), "external")
    if not os.path.exists(external_dir):
        return

    for filename in os.listdir(external_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            plugin_name = filename[:-3]
            module_name = f"plugins.external.{plugin_name}"

            # Model-path fallback: probe model files under the plugin-named subdirectory,
            # and expose them to the plugin via environment variables + module attributes, reducing reliance on LLM-generated hard-coded paths.
            model_dir = os.path.join(external_dir, plugin_name)
            model_files = []
            if os.path.isdir(model_dir):
                for mf in sorted(os.listdir(model_dir)):
                    if mf.endswith((".pkl", ".joblib")):
                        model_files.append(os.path.join(model_dir, mf))
            if model_files:
                os.environ["EXTERNAL_PLUGIN_MODEL_DIR"] = model_dir
                os.environ["EXTERNAL_PLUGIN_MODEL_FILES"] = os.pathsep.join(model_files)
            else:
                os.environ["EXTERNAL_PLUGIN_MODEL_DIR"] = ""
                os.environ["EXTERNAL_PLUGIN_MODEL_FILES"] = ""

            try:
                module = importlib.import_module(module_name)
                # Module-attribute fallback (more reliable than environment variables; unaffected by multi-plugin import order):
                # the plugin can fall back to os.path.join(
                #     os.path.dirname(os.path.abspath(__file__)), plugin_name, 'model.pkl')
                module._external_model_dir = model_dir if model_files else None
                module._external_model_files = model_files
            except Exception as e:
                print(f"[X] Failed to load {module_name}: {e}")


def load_enhanced_plugin():
    """Load the enhanced prediction plugin (8-method router)"""
    try:
        from prediction.plugin import create_plugin
        create_plugin()
    except ImportError:
        print("[!] Enhanced prediction plugin not available (prediction module missing)")
    except Exception as e:
        print(f"[!] Enhanced prediction plugin init failed: {e}")


def load_all_plugins():
    """Load all plugins (builtin + external + enhanced)"""
    load_builtin_plugins()
    load_external_plugins()
    load_enhanced_plugin()
