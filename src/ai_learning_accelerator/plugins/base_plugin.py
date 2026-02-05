"""Base plugin interface and metadata structures."""

import abc
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field


class PluginType(str, Enum):
    """Types of plugins supported by the system."""
    IDE_INTEGRATION = "ide_integration"
    LEARNING_CONTENT = "learning_content"
    ANALYTICS = "analytics"
    AUTOMATION = "automation"
    DEBUGGING = "debugging"
    CONTEXT_ANALYSIS = "context_analysis"
    MULTIMODAL = "multimodal"
    PRIVACY = "privacy"
    ENCRYPTION = "encryption"
    WORKFLOW = "workflow"
    EXTENSION = "extension"


class PluginStatus(str, Enum):
    """Plugin status states."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    LOADING = "loading"
    UNLOADING = "unloading"


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration."""
    
    # Basic identification
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    
    # Plugin classification
    plugin_type: PluginType = Field(..., description="Type of plugin")
    category: str = Field(..., description="Plugin category")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    
    # Compatibility and requirements
    min_api_version: str = Field(..., description="Minimum API version required")
    max_api_version: Optional[str] = Field(None, description="Maximum API version supported")
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")
    conflicts: List[str] = Field(default_factory=list, description="Conflicting plugins")
    
    # Capabilities and permissions
    capabilities: List[str] = Field(default_factory=list, description="Plugin capabilities")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")
    api_endpoints: List[str] = Field(default_factory=list, description="Exposed API endpoints")
    
    # Configuration
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    default_config: Dict[str, Any] = Field(default_factory=dict, description="Default configuration")
    
    # Lifecycle
    auto_start: bool = Field(default=False, description="Auto-start plugin on system startup")
    priority: int = Field(default=100, description="Plugin loading priority (lower = higher priority)")
    
    # Metadata
    homepage: Optional[str] = Field(None, description="Plugin homepage URL")
    repository: Optional[str] = Field(None, description="Plugin repository URL")
    license: Optional[str] = Field(None, description="Plugin license")
    keywords: List[str] = Field(default_factory=list, description="Plugin keywords")


class PluginInfo(BaseModel):
    """Runtime plugin information."""
    
    metadata: PluginMetadata
    status: PluginStatus = PluginStatus.INACTIVE
    instance: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Runtime state
    loaded_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    
    # Statistics
    api_calls: int = 0
    last_activity: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True


class BasePlugin(abc.ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin with configuration."""
        self.config = config or {}
        self._metadata: Optional[PluginMetadata] = None
        self._status = PluginStatus.INACTIVE
        self._logger = None
    
    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @property
    def status(self) -> PluginStatus:
        """Get plugin status."""
        return self._status
    
    @property
    def logger(self):
        """Get plugin logger."""
        if self._logger is None:
            from ..logging_config import get_logger
            self._logger = get_logger(f"plugin.{self.metadata.name}")
        return self._logger
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def start(self) -> bool:
        """Start the plugin.
        
        Returns:
            bool: True if start successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def stop(self) -> bool:
        """Stop the plugin.
        
        Returns:
            bool: True if stop successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup plugin resources.
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        pass
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the plugin.
        
        Args:
            config: Plugin configuration
            
        Returns:
            bool: True if configuration successful, False otherwise
        """
        self.config.update(config)
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform plugin health check.
        
        Returns:
            Dict containing health status information
        """
        return {
            "status": self.status.value,
            "healthy": self.status == PluginStatus.ACTIVE,
            "last_error": None,
            "uptime": None
        }
    
    def get_api_routes(self) -> List[Dict[str, Any]]:
        """Get API routes exposed by this plugin.
        
        Returns:
            List of route definitions
        """
        return []
    
    def get_webhook_handlers(self) -> Dict[str, Any]:
        """Get webhook handlers provided by this plugin.
        
        Returns:
            Dict mapping webhook types to handler functions
        """
        return {}
    
    def get_event_handlers(self) -> Dict[str, Any]:
        """Get event handlers provided by this plugin.
        
        Returns:
            Dict mapping event types to handler functions
        """
        return {}
    
    async def handle_api_call(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call to plugin endpoint.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            
        Returns:
            Response data
        """
        raise NotImplementedError(f"API endpoint {method} {endpoint} not implemented")
    
    async def handle_webhook(self, webhook_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle webhook event.
        
        Args:
            webhook_type: Type of webhook
            data: Webhook data
            
        Returns:
            Response data
        """
        handlers = self.get_webhook_handlers()
        if webhook_type in handlers:
            return await handlers[webhook_type](data)
        raise NotImplementedError(f"Webhook type {webhook_type} not supported")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle system event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        handlers = self.get_event_handlers()
        if event_type in handlers:
            await handlers[event_type](data)
    
    def _set_status(self, status: PluginStatus) -> None:
        """Set plugin status (internal use only)."""
        self._status = status
        self.logger.info(f"Plugin status changed to {status.value}")


class IDEIntegrationPlugin(BasePlugin):
    """Base class for IDE integration plugins."""
    
    @abc.abstractmethod
    async def connect_to_ide(self, ide_type: str, connection_params: Dict[str, Any]) -> bool:
        """Connect to IDE.
        
        Args:
            ide_type: Type of IDE (vscode, intellij, etc.)
            connection_params: Connection parameters
            
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abc.abstractmethod
    async def send_to_ide(self, message_type: str, data: Dict[str, Any]) -> bool:
        """Send message to IDE.
        
        Args:
            message_type: Type of message
            data: Message data
            
        Returns:
            bool: True if message sent successfully
        """
        pass
    
    @abc.abstractmethod
    async def receive_from_ide(self) -> Optional[Dict[str, Any]]:
        """Receive message from IDE.
        
        Returns:
            Message data or None if no message available
        """
        pass


class LearningContentPlugin(BasePlugin):
    """Base class for learning content plugins."""
    
    @abc.abstractmethod
    async def generate_content(self, topic: str, user_level: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning content.
        
        Args:
            topic: Learning topic
            user_level: User skill level
            preferences: User preferences
            
        Returns:
            Generated content
        """
        pass
    
    @abc.abstractmethod
    async def adapt_content(self, content: Dict[str, Any], user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt content based on user feedback.
        
        Args:
            content: Original content
            user_feedback: User feedback
            
        Returns:
            Adapted content
        """
        pass


class AnalyticsPlugin(BasePlugin):
    """Base class for analytics plugins."""
    
    @abc.abstractmethod
    async def collect_metrics(self, user_id: UUID, event_type: str, data: Dict[str, Any]) -> bool:
        """Collect analytics metrics.
        
        Args:
            user_id: User identifier
            event_type: Type of event
            data: Event data
            
        Returns:
            bool: True if metrics collected successfully
        """
        pass
    
    @abc.abstractmethod
    async def generate_insights(self, user_id: UUID, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics insights.
        
        Args:
            user_id: User identifier
            time_range: Time range for analysis
            
        Returns:
            Generated insights
        """
        pass