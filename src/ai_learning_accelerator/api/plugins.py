"""Plugin management API endpoints."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_user, require_admin
from ..models.user import User
from ..plugins import plugin_manager, webhook_manager
from ..plugins.base_plugin import PluginStatus, PluginType
from ..plugins.webhook_manager import WebhookConfig, WebhookEvent, WebhookStatus, WebhookDeliveryStatus

router = APIRouter(prefix="/plugins", tags=["plugins"])


# Plugin Management Schemas

class PluginInfoResponse(BaseModel):
    """Plugin information response."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    status: PluginStatus
    capabilities: List[str]
    api_endpoints: List[str]
    loaded_at: Optional[str] = None
    started_at: Optional[str] = None
    last_error: Optional[str] = None
    error_count: int = 0
    api_calls: int = 0


class PluginListResponse(BaseModel):
    """Plugin list response."""
    plugins: List[PluginInfoResponse]
    total: int


class PluginConfigRequest(BaseModel):
    """Plugin configuration request."""
    config: Dict[str, Any] = Field(..., description="Plugin configuration")


class PluginActionResponse(BaseModel):
    """Plugin action response."""
    success: bool
    message: str
    plugin_name: str


# Webhook Management Schemas

class WebhookCreateRequest(BaseModel):
    """Webhook creation request."""
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    description: Optional[str] = Field(None, description="Webhook description")
    events: List[WebhookEvent] = Field(..., description="Events to subscribe to")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Event filters")
    secret: Optional[str] = Field(None, description="Webhook secret")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retry attempts")


class WebhookResponse(BaseModel):
    """Webhook response."""
    id: str
    name: str
    url: str
    description: Optional[str]
    events: List[WebhookEvent]
    status: WebhookStatus
    created_at: str
    created_by: str


class WebhookListResponse(BaseModel):
    """Webhook list response."""
    webhooks: List[WebhookResponse]
    total: int


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery response."""
    id: str
    webhook_id: str
    event_type: WebhookEvent
    status: WebhookDeliveryStatus
    attempts: int
    max_attempts: int
    created_at: str
    delivered_at: Optional[str]
    response_status: Optional[int]
    error_message: Optional[str]


class WebhookDeliveryListResponse(BaseModel):
    """Webhook delivery list response."""
    deliveries: List[WebhookDeliveryResponse]
    total: int


class WebhookStatsResponse(BaseModel):
    """Webhook statistics response."""
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    retried_deliveries: int
    active_webhooks: int
    total_webhooks: int
    pending_deliveries: int
    queue_size: int


# Plugin Management Endpoints

@router.get("/", response_model=PluginListResponse)
async def list_plugins(
    plugin_type: Optional[PluginType] = None,
    status: Optional[PluginStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """List all loaded plugins."""
    try:
        plugin_names = plugin_manager.list_plugins(plugin_type=plugin_type, status=status)
        plugins = []
        
        for name in plugin_names:
            plugin_info = plugin_manager.get_plugin_info(name)
            if plugin_info:
                plugins.append(PluginInfoResponse(
                    name=name,
                    version=plugin_info.metadata.version,
                    description=plugin_info.metadata.description,
                    author=plugin_info.metadata.author,
                    plugin_type=plugin_info.metadata.plugin_type,
                    status=plugin_info.status,
                    capabilities=plugin_info.metadata.capabilities,
                    api_endpoints=plugin_info.metadata.api_endpoints,
                    loaded_at=plugin_info.loaded_at.isoformat() if plugin_info.loaded_at else None,
                    started_at=plugin_info.started_at.isoformat() if plugin_info.started_at else None,
                    last_error=plugin_info.last_error,
                    error_count=plugin_info.error_count,
                    api_calls=plugin_info.api_calls
                ))
        
        return PluginListResponse(plugins=plugins, total=len(plugins))
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing plugins: {str(e)}"
        )


@router.get("/{plugin_name}", response_model=PluginInfoResponse)
async def get_plugin_info(
    plugin_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get information about a specific plugin."""
    plugin_info = plugin_manager.get_plugin_info(plugin_name)
    
    if not plugin_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin {plugin_name} not found"
        )
    
    return PluginInfoResponse(
        name=plugin_name,
        version=plugin_info.metadata.version,
        description=plugin_info.metadata.description,
        author=plugin_info.metadata.author,
        plugin_type=plugin_info.metadata.plugin_type,
        status=plugin_info.status,
        capabilities=plugin_info.metadata.capabilities,
        api_endpoints=plugin_info.metadata.api_endpoints,
        loaded_at=plugin_info.loaded_at.isoformat() if plugin_info.loaded_at else None,
        started_at=plugin_info.started_at.isoformat() if plugin_info.started_at else None,
        last_error=plugin_info.last_error,
        error_count=plugin_info.error_count,
        api_calls=plugin_info.api_calls
    )


@router.post("/{plugin_name}/load", response_model=PluginActionResponse)
async def load_plugin(
    plugin_name: str,
    config_request: Optional[PluginConfigRequest] = None,
    current_user: User = Depends(require_admin)
):
    """Load a plugin."""
    try:
        config = config_request.config if config_request else {}
        success = await plugin_manager.load_plugin(plugin_name, config)
        
        return PluginActionResponse(
            success=success,
            message=f"Plugin {plugin_name} {'loaded' if success else 'failed to load'}",
            plugin_name=plugin_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading plugin {plugin_name}: {str(e)}"
        )


@router.post("/{plugin_name}/unload", response_model=PluginActionResponse)
async def unload_plugin(
    plugin_name: str,
    current_user: User = Depends(require_admin)
):
    """Unload a plugin."""
    try:
        success = await plugin_manager.unload_plugin(plugin_name)
        
        return PluginActionResponse(
            success=success,
            message=f"Plugin {plugin_name} {'unloaded' if success else 'failed to unload'}",
            plugin_name=plugin_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading plugin {plugin_name}: {str(e)}"
        )


@router.post("/{plugin_name}/start", response_model=PluginActionResponse)
async def start_plugin(
    plugin_name: str,
    current_user: User = Depends(require_admin)
):
    """Start a plugin."""
    try:
        success = await plugin_manager.start_plugin(plugin_name)
        
        return PluginActionResponse(
            success=success,
            message=f"Plugin {plugin_name} {'started' if success else 'failed to start'}",
            plugin_name=plugin_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting plugin {plugin_name}: {str(e)}"
        )


@router.post("/{plugin_name}/stop", response_model=PluginActionResponse)
async def stop_plugin(
    plugin_name: str,
    current_user: User = Depends(require_admin)
):
    """Stop a plugin."""
    try:
        success = await plugin_manager.stop_plugin(plugin_name)
        
        return PluginActionResponse(
            success=success,
            message=f"Plugin {plugin_name} {'stopped' if success else 'failed to stop'}",
            plugin_name=plugin_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping plugin {plugin_name}: {str(e)}"
        )


@router.post("/{plugin_name}/restart", response_model=PluginActionResponse)
async def restart_plugin(
    plugin_name: str,
    current_user: User = Depends(require_admin)
):
    """Restart a plugin."""
    try:
        success = await plugin_manager.restart_plugin(plugin_name)
        
        return PluginActionResponse(
            success=success,
            message=f"Plugin {plugin_name} {'restarted' if success else 'failed to restart'}",
            plugin_name=plugin_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error restarting plugin {plugin_name}: {str(e)}"
        )


@router.post("/{plugin_name}/api/{endpoint:path}")
async def call_plugin_api(
    plugin_name: str,
    endpoint: str,
    request_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Call a plugin API endpoint."""
    try:
        result = await plugin_manager.call_plugin_api(
            plugin_name, endpoint, "POST", request_data
        )
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calling plugin API {plugin_name}.{endpoint}: {str(e)}"
        )


# Webhook Management Endpoints

@router.post("/webhooks", response_model=WebhookResponse)
async def create_webhook(
    webhook_request: WebhookCreateRequest,
    current_user: User = Depends(require_admin)
):
    """Create a new webhook."""
    try:
        webhook_config = WebhookConfig(
            name=webhook_request.name,
            url=webhook_request.url,
            description=webhook_request.description,
            events=webhook_request.events,
            filters=webhook_request.filters,
            secret=webhook_request.secret,
            headers=webhook_request.headers,
            timeout=webhook_request.timeout,
            retry_count=webhook_request.retry_count,
            created_by=current_user.id
        )
        
        webhook_id = webhook_manager.register_webhook(webhook_config)
        
        return WebhookResponse(
            id=str(webhook_id),
            name=webhook_config.name,
            url=webhook_config.url,
            description=webhook_config.description,
            events=webhook_config.events,
            status=webhook_config.status,
            created_at=webhook_config.created_at.isoformat(),
            created_by=str(webhook_config.created_by)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating webhook: {str(e)}"
        )


@router.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    event_type: Optional[WebhookEvent] = None,
    status: Optional[WebhookStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """List all webhooks."""
    try:
        webhooks_data = webhook_manager.list_webhooks(event_type=event_type, status=status)
        webhooks = []
        
        for webhook_id, config in webhooks_data:
            webhooks.append(WebhookResponse(
                id=str(webhook_id),
                name=config.name,
                url=config.url,
                description=config.description,
                events=config.events,
                status=config.status,
                created_at=config.created_at.isoformat(),
                created_by=str(config.created_by)
            ))
        
        return WebhookListResponse(webhooks=webhooks, total=len(webhooks))
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing webhooks: {str(e)}"
        )


@router.get("/webhooks/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: UUID,
    current_user: User = Depends(get_current_user)
):
    """Get webhook information."""
    webhook_config = webhook_manager.get_webhook(webhook_id)
    
    if not webhook_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found"
        )
    
    return WebhookResponse(
        id=str(webhook_id),
        name=webhook_config.name,
        url=webhook_config.url,
        description=webhook_config.description,
        events=webhook_config.events,
        status=webhook_config.status,
        created_at=webhook_config.created_at.isoformat(),
        created_by=str(webhook_config.created_by)
    )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: UUID,
    current_user: User = Depends(require_admin)
):
    """Delete a webhook."""
    success = webhook_manager.unregister_webhook(webhook_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found"
        )
    
    return {"message": f"Webhook {webhook_id} deleted successfully"}


@router.get("/webhooks/{webhook_id}/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    webhook_id: UUID,
    status: Optional[WebhookDeliveryStatus] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """List webhook deliveries."""
    try:
        deliveries_data = webhook_manager.list_deliveries(
            webhook_id=webhook_id, status=status, limit=limit
        )
        
        deliveries = []
        for delivery in deliveries_data:
            deliveries.append(WebhookDeliveryResponse(
                id=str(delivery.id),
                webhook_id=str(delivery.webhook_id),
                event_type=delivery.event_type,
                status=delivery.status,
                attempts=delivery.attempts,
                max_attempts=delivery.max_attempts,
                created_at=delivery.created_at.isoformat(),
                delivered_at=delivery.delivered_at.isoformat() if delivery.delivered_at else None,
                response_status=delivery.response_status,
                error_message=delivery.error_message
            ))
        
        return WebhookDeliveryListResponse(deliveries=deliveries, total=len(deliveries))
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing webhook deliveries: {str(e)}"
        )


@router.post("/webhooks/deliveries/{delivery_id}/retry")
async def retry_webhook_delivery(
    delivery_id: UUID,
    current_user: User = Depends(require_admin)
):
    """Retry a failed webhook delivery."""
    try:
        success = await webhook_manager.retry_delivery(delivery_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Delivery {delivery_id} not found or cannot be retried"
            )
        
        return {"message": f"Delivery {delivery_id} queued for retry"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrying webhook delivery: {str(e)}"
        )


@router.get("/webhooks/stats", response_model=WebhookStatsResponse)
async def get_webhook_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get webhook delivery statistics."""
    try:
        stats = webhook_manager.get_statistics()
        return WebhookStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting webhook statistics: {str(e)}"
        )


@router.post("/webhooks/test")
async def test_webhook_event(
    event_type: WebhookEvent,
    event_data: Dict[str, Any],
    current_user: User = Depends(require_admin)
):
    """Send a test webhook event."""
    try:
        delivery_ids = await webhook_manager.send_webhook(
            event_type, event_data, current_user.id
        )
        
        return {
            "message": f"Test webhook event sent",
            "event_type": event_type.value,
            "delivery_ids": [str(id) for id in delivery_ids]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending test webhook: {str(e)}"
        )