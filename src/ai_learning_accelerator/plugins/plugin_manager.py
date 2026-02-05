"""Plugin management system."""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type
from uuid import UUID

from ..logging_config import get_logger
from .base_plugin import BasePlugin, PluginInfo, PluginMetadata, PluginStatus, PluginType


class PluginManager:
    """Manages plugin lifecycle and operations."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.logger = get_logger(__name__)
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_directories: List[Path] = []
        self._api_routes: Dict[str, Dict[str, Any]] = {}
        self._webhook_handlers: Dict[str, List[str]] = {}  # webhook_type -> plugin_names
        self._event_handlers: Dict[str, List[str]] = {}   # event_type -> plugin_names
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_dependency_graph: Dict[str, Set[str]] = {}
        
        # Plugin loading state
        self._loading_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
    
    def add_plugin_directory(self, directory: Path) -> None:
        """Add a directory to search for plugins.
        
        Args:
            directory: Directory path to search for plugins
        """
        if directory.exists() and directory.is_dir():
            self._plugin_directories.append(directory)
            self.logger.info(f"Added plugin directory: {directory}")
        else:
            self.logger.warning(f"Plugin directory does not exist: {directory}")
    
    async def discover_plugins(self) -> List[str]:
        """Discover available plugins in configured directories.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for directory in self._plugin_directories:
            try:
                for item in directory.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        plugin_file = item / "plugin.py"
                        metadata_file = item / "metadata.json"
                        
                        if plugin_file.exists():
                            plugin_name = item.name
                            if await self._validate_plugin_structure(item):
                                discovered.append(plugin_name)
                                self.logger.info(f"Discovered plugin: {plugin_name}")
                            else:
                                self.logger.warning(f"Invalid plugin structure: {plugin_name}")
                                
            except Exception as e:
                self.logger.error(f"Error discovering plugins in {directory}: {e}")
        
        return discovered
    
    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """Load a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Plugin configuration
            
        Returns:
            bool: True if plugin loaded successfully
        """
        async with self._loading_lock:
            if plugin_name in self._plugins:
                self.logger.warning(f"Plugin {plugin_name} is already loaded")
                return True
            
            try:
                # Find plugin directory
                plugin_dir = self._find_plugin_directory(plugin_name)
                if not plugin_dir:
                    self.logger.error(f"Plugin directory not found: {plugin_name}")
                    return False
                
                # Load plugin module
                plugin_module = await self._load_plugin_module(plugin_dir)
                if not plugin_module:
                    return False
                
                # Create plugin instance
                plugin_class = self._find_plugin_class(plugin_module)
                if not plugin_class:
                    self.logger.error(f"No plugin class found in {plugin_name}")
                    return False
                
                plugin_instance = plugin_class(config or {})
                
                # Create plugin info
                plugin_info = PluginInfo(
                    metadata=plugin_instance.metadata,
                    status=PluginStatus.LOADING,
                    instance=plugin_instance,
                    config=config or {}
                )
                
                # Validate dependencies
                if not await self._validate_dependencies(plugin_info.metadata):
                    self.logger.error(f"Dependency validation failed for {plugin_name}")
                    return False
                
                # Initialize plugin
                if not await plugin_instance.initialize():
                    self.logger.error(f"Plugin initialization failed: {plugin_name}")
                    return False
                
                # Register plugin
                self._plugins[plugin_name] = plugin_info
                self._register_plugin_capabilities(plugin_name, plugin_instance)
                self._update_dependency_graph(plugin_name, plugin_info.metadata)
                
                plugin_info.status = PluginStatus.INACTIVE
                plugin_instance._set_status(PluginStatus.INACTIVE)
                
                self.logger.info(f"Plugin loaded successfully: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading plugin {plugin_name}: {e}")
                return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if plugin unloaded successfully
        """
        async with self._loading_lock:
            if plugin_name not in self._plugins:
                self.logger.warning(f"Plugin {plugin_name} is not loaded")
                return True
            
            try:
                plugin_info = self._plugins[plugin_name]
                plugin_instance = plugin_info.instance
                
                # Check for dependent plugins
                dependents = self._reverse_dependency_graph.get(plugin_name, set())
                if dependents:
                    active_dependents = [
                        dep for dep in dependents 
                        if self._plugins.get(dep, {}).status == PluginStatus.ACTIVE
                    ]
                    if active_dependents:
                        self.logger.error(
                            f"Cannot unload {plugin_name}: active dependents {active_dependents}"
                        )
                        return False
                
                # Stop plugin if active
                if plugin_info.status == PluginStatus.ACTIVE:
                    await self.stop_plugin(plugin_name)
                
                # Set status to unloading
                plugin_info.status = PluginStatus.UNLOADING
                plugin_instance._set_status(PluginStatus.UNLOADING)
                
                # Cleanup plugin
                await plugin_instance.cleanup()
                
                # Unregister plugin capabilities
                self._unregister_plugin_capabilities(plugin_name)
                
                # Remove from dependency graph
                self._remove_from_dependency_graph(plugin_name)
                
                # Remove plugin
                del self._plugins[plugin_name]
                
                self.logger.info(f"Plugin unloaded successfully: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
                return False
    
    async def start_plugin(self, plugin_name: str) -> bool:
        """Start a loaded plugin.
        
        Args:
            plugin_name: Name of the plugin to start
            
        Returns:
            bool: True if plugin started successfully
        """
        if plugin_name not in self._plugins:
            self.logger.error(f"Plugin {plugin_name} is not loaded")
            return False
        
        plugin_info = self._plugins[plugin_name]
        
        if plugin_info.status == PluginStatus.ACTIVE:
            self.logger.warning(f"Plugin {plugin_name} is already active")
            return True
        
        if plugin_info.status != PluginStatus.INACTIVE:
            self.logger.error(f"Plugin {plugin_name} is in invalid state: {plugin_info.status}")
            return False
        
        try:
            # Start dependencies first
            for dep_name in plugin_info.metadata.dependencies:
                if dep_name in self._plugins:
                    dep_info = self._plugins[dep_name]
                    if dep_info.status != PluginStatus.ACTIVE:
                        if not await self.start_plugin(dep_name):
                            self.logger.error(f"Failed to start dependency {dep_name} for {plugin_name}")
                            return False
            
            # Start plugin
            if await plugin_info.instance.start():
                plugin_info.status = PluginStatus.ACTIVE
                plugin_info.instance._set_status(PluginStatus.ACTIVE)
                plugin_info.started_at = datetime.utcnow()
                
                self.logger.info(f"Plugin started successfully: {plugin_name}")
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.instance._set_status(PluginStatus.ERROR)
                self.logger.error(f"Plugin start failed: {plugin_name}")
                return False
                
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.instance._set_status(PluginStatus.ERROR)
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            self.logger.error(f"Error starting plugin {plugin_name}: {e}")
            return False
    
    async def stop_plugin(self, plugin_name: str) -> bool:
        """Stop an active plugin.
        
        Args:
            plugin_name: Name of the plugin to stop
            
        Returns:
            bool: True if plugin stopped successfully
        """
        if plugin_name not in self._plugins:
            self.logger.error(f"Plugin {plugin_name} is not loaded")
            return False
        
        plugin_info = self._plugins[plugin_name]
        
        if plugin_info.status != PluginStatus.ACTIVE:
            self.logger.warning(f"Plugin {plugin_name} is not active")
            return True
        
        try:
            # Stop dependent plugins first
            dependents = self._reverse_dependency_graph.get(plugin_name, set())
            for dep_name in dependents:
                if dep_name in self._plugins and self._plugins[dep_name].status == PluginStatus.ACTIVE:
                    await self.stop_plugin(dep_name)
            
            # Stop plugin
            if await plugin_info.instance.stop():
                plugin_info.status = PluginStatus.INACTIVE
                plugin_info.instance._set_status(PluginStatus.INACTIVE)
                
                self.logger.info(f"Plugin stopped successfully: {plugin_name}")
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.instance._set_status(PluginStatus.ERROR)
                self.logger.error(f"Plugin stop failed: {plugin_name}")
                return False
                
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.instance._set_status(PluginStatus.ERROR)
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            self.logger.error(f"Error stopping plugin {plugin_name}: {e}")
            return False
    
    async def restart_plugin(self, plugin_name: str) -> bool:
        """Restart a plugin.
        
        Args:
            plugin_name: Name of the plugin to restart
            
        Returns:
            bool: True if plugin restarted successfully
        """
        if await self.stop_plugin(plugin_name):
            return await self.start_plugin(plugin_name)
        return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin information or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None, status: Optional[PluginStatus] = None) -> List[str]:
        """List loaded plugins.
        
        Args:
            plugin_type: Filter by plugin type
            status: Filter by plugin status
            
        Returns:
            List of plugin names matching criteria
        """
        plugins = []
        
        for name, info in self._plugins.items():
            if plugin_type and info.metadata.plugin_type != plugin_type:
                continue
            if status and info.status != status:
                continue
            plugins.append(name)
        
        return plugins
    
    async def call_plugin_api(self, plugin_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a plugin API endpoint.
        
        Args:
            plugin_name: Name of the plugin
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            
        Returns:
            Response data
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        plugin_info = self._plugins[plugin_name]
        
        if plugin_info.status != PluginStatus.ACTIVE:
            raise ValueError(f"Plugin {plugin_name} is not active")
        
        try:
            result = await plugin_info.instance.handle_api_call(endpoint, method, data)
            plugin_info.api_calls += 1
            plugin_info.last_activity = datetime.utcnow()
            return result
            
        except Exception as e:
            plugin_info.error_count += 1
            plugin_info.last_error = str(e)
            self.logger.error(f"Plugin API call failed {plugin_name}.{endpoint}: {e}")
            raise
    
    async def send_webhook(self, webhook_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send webhook to all registered handlers.
        
        Args:
            webhook_type: Type of webhook
            data: Webhook data
            
        Returns:
            List of responses from handlers
        """
        responses = []
        
        plugin_names = self._webhook_handlers.get(webhook_type, [])
        
        for plugin_name in plugin_names:
            if plugin_name in self._plugins:
                plugin_info = self._plugins[plugin_name]
                
                if plugin_info.status == PluginStatus.ACTIVE:
                    try:
                        response = await plugin_info.instance.handle_webhook(webhook_type, data)
                        responses.append({
                            "plugin": plugin_name,
                            "success": True,
                            "response": response
                        })
                        
                        plugin_info.last_activity = datetime.utcnow()
                        
                    except Exception as e:
                        plugin_info.error_count += 1
                        plugin_info.last_error = str(e)
                        responses.append({
                            "plugin": plugin_name,
                            "success": False,
                            "error": str(e)
                        })
                        self.logger.error(f"Webhook handler failed {plugin_name}.{webhook_type}: {e}")
        
        return responses
    
    async def send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send event to all registered handlers.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        plugin_names = self._event_handlers.get(event_type, [])
        
        for plugin_name in plugin_names:
            if plugin_name in self._plugins:
                plugin_info = self._plugins[plugin_name]
                
                if plugin_info.status == PluginStatus.ACTIVE:
                    try:
                        await plugin_info.instance.handle_event(event_type, data)
                        plugin_info.last_activity = datetime.utcnow()
                        
                    except Exception as e:
                        plugin_info.error_count += 1
                        plugin_info.last_error = str(e)
                        self.logger.error(f"Event handler failed {plugin_name}.{event_type}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown plugin manager and all plugins."""
        self.logger.info("Shutting down plugin manager")
        
        # Set shutdown event
        self._shutdown_event.set()
        
        # Stop all active plugins
        active_plugins = self.list_plugins(status=PluginStatus.ACTIVE)
        for plugin_name in active_plugins:
            await self.stop_plugin(plugin_name)
        
        # Unload all plugins
        loaded_plugins = list(self._plugins.keys())
        for plugin_name in loaded_plugins:
            await self.unload_plugin(plugin_name)
        
        self.logger.info("Plugin manager shutdown complete")
    
    # Private methods
    
    def _find_plugin_directory(self, plugin_name: str) -> Optional[Path]:
        """Find plugin directory by name."""
        for directory in self._plugin_directories:
            plugin_dir = directory / plugin_name
            if plugin_dir.exists() and plugin_dir.is_dir():
                return plugin_dir
        return None
    
    async def _validate_plugin_structure(self, plugin_dir: Path) -> bool:
        """Validate plugin directory structure."""
        required_files = ["plugin.py"]
        
        for file_name in required_files:
            if not (plugin_dir / file_name).exists():
                return False
        
        return True
    
    async def _load_plugin_module(self, plugin_dir: Path):
        """Load plugin module from directory."""
        try:
            # Add plugin directory to Python path
            sys.path.insert(0, str(plugin_dir.parent))
            
            # Import plugin module
            module_name = plugin_dir.name
            spec = importlib.util.spec_from_file_location(
                f"plugin_{module_name}",
                plugin_dir / "plugin.py"
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            
        except Exception as e:
            self.logger.error(f"Error loading plugin module from {plugin_dir}: {e}")
        
        finally:
            # Remove from Python path
            if str(plugin_dir.parent) in sys.path:
                sys.path.remove(str(plugin_dir.parent))
        
        return None
    
    def _find_plugin_class(self, module) -> Optional[Type[BasePlugin]]:
        """Find plugin class in module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BasePlugin) and 
                obj != BasePlugin and 
                not inspect.isabstract(obj)):
                return obj
        return None
    
    async def _validate_dependencies(self, metadata: PluginMetadata) -> bool:
        """Validate plugin dependencies."""
        for dep_name in metadata.dependencies:
            if dep_name not in self._plugins:
                self.logger.error(f"Missing dependency: {dep_name}")
                return False
        
        # Check for conflicts
        for conflict_name in metadata.conflicts:
            if conflict_name in self._plugins:
                conflict_info = self._plugins[conflict_name]
                if conflict_info.status in [PluginStatus.ACTIVE, PluginStatus.INACTIVE]:
                    self.logger.error(f"Conflicting plugin loaded: {conflict_name}")
                    return False
        
        return True
    
    def _register_plugin_capabilities(self, plugin_name: str, plugin_instance: BasePlugin) -> None:
        """Register plugin capabilities."""
        # Register API routes
        routes = plugin_instance.get_api_routes()
        for route in routes:
            endpoint = route.get("endpoint")
            if endpoint:
                self._api_routes[endpoint] = {
                    "plugin": plugin_name,
                    "route": route
                }
        
        # Register webhook handlers
        webhook_handlers = plugin_instance.get_webhook_handlers()
        for webhook_type in webhook_handlers.keys():
            if webhook_type not in self._webhook_handlers:
                self._webhook_handlers[webhook_type] = []
            self._webhook_handlers[webhook_type].append(plugin_name)
        
        # Register event handlers
        event_handlers = plugin_instance.get_event_handlers()
        for event_type in event_handlers.keys():
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(plugin_name)
    
    def _unregister_plugin_capabilities(self, plugin_name: str) -> None:
        """Unregister plugin capabilities."""
        # Unregister API routes
        routes_to_remove = [
            endpoint for endpoint, info in self._api_routes.items()
            if info["plugin"] == plugin_name
        ]
        for endpoint in routes_to_remove:
            del self._api_routes[endpoint]
        
        # Unregister webhook handlers
        for webhook_type, plugin_names in self._webhook_handlers.items():
            if plugin_name in plugin_names:
                plugin_names.remove(plugin_name)
        
        # Unregister event handlers
        for event_type, plugin_names in self._event_handlers.items():
            if plugin_name in plugin_names:
                plugin_names.remove(plugin_name)
    
    def _update_dependency_graph(self, plugin_name: str, metadata: PluginMetadata) -> None:
        """Update dependency graph."""
        self._dependency_graph[plugin_name] = set(metadata.dependencies)
        
        # Update reverse dependency graph
        for dep_name in metadata.dependencies:
            if dep_name not in self._reverse_dependency_graph:
                self._reverse_dependency_graph[dep_name] = set()
            self._reverse_dependency_graph[dep_name].add(plugin_name)
    
    def _remove_from_dependency_graph(self, plugin_name: str) -> None:
        """Remove plugin from dependency graph."""
        # Remove from dependency graph
        if plugin_name in self._dependency_graph:
            dependencies = self._dependency_graph[plugin_name]
            del self._dependency_graph[plugin_name]
            
            # Update reverse dependency graph
            for dep_name in dependencies:
                if dep_name in self._reverse_dependency_graph:
                    self._reverse_dependency_graph[dep_name].discard(plugin_name)
        
        # Remove from reverse dependency graph
        if plugin_name in self._reverse_dependency_graph:
            del self._reverse_dependency_graph[plugin_name]


# Global plugin manager instance
plugin_manager = PluginManager()


# Import datetime at the top level to avoid issues
from datetime import datetime