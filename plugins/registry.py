from typing import Dict, Optional, List, Any, Callable
import importlib
import os

# The Plugin class needs to be available here for builtin plugins to import
from plugins.base import Plugin

class PluginRegistry:
    """Plugin registration center"""
    
    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._structure_map: Dict[str, str] = {}  # structure_type -> plugin_id
    
    def register(self, plugin):
        """Register a plugin"""
        self._plugins[plugin.id] = plugin
        if plugin.structure_type:
            self._structure_map[plugin.structure_type] = plugin.id
        print(f"[+] Plugin Registered: [{plugin.id}] for structure [{plugin.structure_type}]")
    
    def get(self, plugin_id: str):
        return self._plugins.get(plugin_id)
    
    def get_by_structure(self, structure_type: str):
        """Get a plugin by structure type"""
        plugin_id = self._structure_map.get(structure_type)
        return self._plugins.get(plugin_id) if plugin_id else None
    
    def list_all(self) -> List:
        return list(self._plugins.values())
    
    def enable(self, plugin_id: str):
        """Enable a plugin"""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].enabled = True
    
    def disable(self, plugin_id: str):
        """Disable a plugin"""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].enabled = False
    
    def load_from_directory(self, plugin_dir: str, base_module: str = "plugins.builtin"):
        """Dynamically load plugins from a directory"""
        if not os.path.exists(plugin_dir):
            print(f"[!] Plugin directory not found: {plugin_dir}")
            return
        
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                full_module_path = f"{base_module}.{module_name}"
                try:
                    importlib.import_module(full_module_path)
                    print(f"   [*] Loaded plugin module: {module_name}")
                except Exception as e:
                    print(f"   [!] Failed to load {module_name}: {e}")

# Global registry instance
global_registry = PluginRegistry()

def register_plugin(plugin):
    """Convenience registration function"""
    global_registry.register(plugin)
