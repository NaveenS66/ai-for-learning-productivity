"""Plugin system for AI Learning Accelerator."""

from .plugin_manager import PluginManager, plugin_manager
from .base_plugin import BasePlugin, PluginMetadata, PluginType
from .webhook_manager import WebhookManager, webhook_manager

__all__ = [
    "PluginManager",
    "plugin_manager", 
    "BasePlugin",
    "PluginMetadata",
    "PluginType",
    "WebhookManager",
    "webhook_manager"
]