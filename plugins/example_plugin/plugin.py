"""Example plugin demonstrating the plugin architecture."""

from typing import Any, Dict, List
from uuid import UUID

from src.ai_learning_accelerator.plugins.base_plugin import (
    BasePlugin, PluginMetadata, PluginType, PluginStatus
)


class ExamplePlugin(BasePlugin):
    """Example plugin for demonstration purposes."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="example-plugin",
            version="1.0.0",
            description="Example plugin demonstrating the plugin architecture",
            author="AI Learning Accelerator Team",
            plugin_type=PluginType.EXTENSION,
            category="example",
            tags=["example", "demo", "tutorial"],
            min_api_version="1.0.0",
            capabilities=["api_endpoints", "webhooks", "events"],
            permissions=["read_user_data"],
            api_endpoints=["/example/hello", "/example/status"],
            config_schema={
                "type": "object",
                "properties": {
                    "greeting": {
                        "type": "string",
                        "default": "Hello"
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True
                    }
                }
            },
            default_config={
                "greeting": "Hello",
                "enabled": True
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing example plugin")
            
            # Initialize plugin state
            self._call_count = 0
            self._initialized_at = None
            
            # Validate configuration
            if not isinstance(self.config.get("greeting"), str):
                self.logger.error("Invalid greeting configuration")
                return False
            
            self.logger.info("Example plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing example plugin: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the plugin."""
        try:
            self.logger.info("Starting example plugin")
            
            from datetime import datetime
            self._initialized_at = datetime.utcnow()
            
            self.logger.info("Example plugin started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting example plugin: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the plugin."""
        try:
            self.logger.info("Stopping example plugin")
            
            # Cleanup any resources
            self._initialized_at = None
            
            self.logger.info("Example plugin stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping example plugin: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up example plugin")
            
            # Reset state
            self._call_count = 0
            self._initialized_at = None
            
            self.logger.info("Example plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up example plugin: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform plugin health check."""
        return {
            "status": self.status.value,
            "healthy": self.status == PluginStatus.ACTIVE,
            "last_error": None,
            "uptime": str(self._initialized_at) if self._initialized_at else None,
            "call_count": self._call_count,
            "config": self.config
        }
    
    def get_api_routes(self) -> List[Dict[str, Any]]:
        """Get API routes exposed by this plugin."""
        return [
            {
                "endpoint": "/example/hello",
                "methods": ["GET", "POST"],
                "description": "Get a greeting message"
            },
            {
                "endpoint": "/example/status",
                "methods": ["GET"],
                "description": "Get plugin status"
            }
        ]
    
    def get_webhook_handlers(self) -> Dict[str, Any]:
        """Get webhook handlers provided by this plugin."""
        return {
            "user.created": self._handle_user_created,
            "learning.progress": self._handle_learning_progress
        }
    
    def get_event_handlers(self) -> Dict[str, Any]:
        """Get event handlers provided by this plugin."""
        return {
            "system.startup": self._handle_system_startup,
            "system.shutdown": self._handle_system_shutdown
        }
    
    async def handle_api_call(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call to plugin endpoint."""
        self._call_count += 1
        
        if endpoint == "/example/hello":
            if method == "GET":
                return {
                    "message": f"{self.config.get('greeting', 'Hello')} from Example Plugin!",
                    "call_count": self._call_count,
                    "status": self.status.value
                }
            elif method == "POST":
                name = data.get("name", "World")
                return {
                    "message": f"{self.config.get('greeting', 'Hello')} {name} from Example Plugin!",
                    "call_count": self._call_count,
                    "status": self.status.value
                }
        
        elif endpoint == "/example/status":
            if method == "GET":
                return await self.health_check()
        
        raise NotImplementedError(f"API endpoint {method} {endpoint} not implemented")
    
    async def _handle_user_created(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user created webhook."""
        self.logger.info(f"User created webhook received: {data}")
        
        user_id = data.get("user_id")
        user_email = data.get("email", "unknown")
        
        return {
            "plugin": "example-plugin",
            "message": f"Welcome new user {user_email}!",
            "user_id": user_id,
            "processed_at": str(datetime.utcnow())
        }
    
    async def _handle_learning_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning progress webhook."""
        self.logger.info(f"Learning progress webhook received: {data}")
        
        user_id = data.get("user_id")
        progress = data.get("progress", 0)
        
        return {
            "plugin": "example-plugin",
            "message": f"User progress updated to {progress}%",
            "user_id": user_id,
            "progress": progress,
            "processed_at": str(datetime.utcnow())
        }
    
    async def _handle_system_startup(self, data: Dict[str, Any]) -> None:
        """Handle system startup event."""
        self.logger.info("System startup event received")
    
    async def _handle_system_shutdown(self, data: Dict[str, Any]) -> None:
        """Handle system shutdown event."""
        self.logger.info("System shutdown event received")


# Import datetime at module level
from datetime import datetime