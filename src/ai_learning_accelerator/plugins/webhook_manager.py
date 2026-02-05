"""Webhook management system for external integrations."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from ..logging_config import get_logger


class WebhookStatus(str, Enum):
    """Webhook status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DISABLED = "disabled"


class WebhookEvent(str, Enum):
    """Types of webhook events."""
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    
    LEARNING_PROGRESS = "learning.progress"
    LEARNING_MILESTONE = "learning.milestone"
    LEARNING_COMPLETED = "learning.completed"
    
    CONTENT_CREATED = "content.created"
    CONTENT_UPDATED = "content.updated"
    CONTENT_RATED = "content.rated"
    
    AUTOMATION_TRIGGERED = "automation.triggered"
    AUTOMATION_COMPLETED = "automation.completed"
    
    DEBUG_SESSION_STARTED = "debug.session_started"
    DEBUG_SOLUTION_FOUND = "debug.solution_found"
    
    CONTEXT_ANALYZED = "context.analyzed"
    CONTEXT_RECOMMENDATION = "context.recommendation"
    
    PRIVACY_VIOLATION = "privacy.violation"
    SECURITY_ALERT = "security.alert"
    
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"


class WebhookDeliveryStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    
    # Basic configuration
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    description: Optional[str] = Field(None, description="Webhook description")
    
    # Event configuration
    events: List[WebhookEvent] = Field(..., description="Events to subscribe to")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Event filters")
    
    # Security configuration
    secret: Optional[str] = Field(None, description="Webhook secret for signature verification")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    
    # Delivery configuration
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retry attempts")
    retry_delay: int = Field(default=60, description="Delay between retries in seconds")
    
    # Status and metadata
    status: WebhookStatus = Field(default=WebhookStatus.ACTIVE, description="Webhook status")
    created_by: UUID = Field(..., description="User who created the webhook")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    # Advanced configuration
    batch_size: int = Field(default=1, description="Number of events to batch together")
    batch_timeout: int = Field(default=10, description="Maximum time to wait for batch completion")
    content_type: str = Field(default="application/json", description="Content type for requests")


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    
    id: UUID = Field(default_factory=uuid4, description="Delivery ID")
    webhook_id: UUID = Field(..., description="Webhook ID")
    event_type: WebhookEvent = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(..., description="Event data")
    
    # Delivery details
    status: WebhookDeliveryStatus = Field(default=WebhookDeliveryStatus.PENDING, description="Delivery status")
    attempts: int = Field(default=0, description="Number of delivery attempts")
    max_attempts: int = Field(default=3, description="Maximum delivery attempts")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    scheduled_at: datetime = Field(default_factory=datetime.utcnow, description="Scheduled delivery time")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(days=7), description="Expiration timestamp")
    
    # Response details
    response_status: Optional[int] = Field(None, description="HTTP response status")
    response_headers: Dict[str, str] = Field(default_factory=dict, description="HTTP response headers")
    response_body: Optional[str] = Field(None, description="HTTP response body")
    error_message: Optional[str] = Field(None, description="Error message if delivery failed")
    
    # Metadata
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID for tracking")
    signature: Optional[str] = Field(None, description="Request signature")


class WebhookManager:
    """Manages webhook registrations and deliveries."""
    
    def __init__(self):
        """Initialize webhook manager."""
        self.logger = get_logger(__name__)
        self._webhooks: Dict[UUID, WebhookConfig] = {}
        self._deliveries: Dict[UUID, WebhookDelivery] = {}
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self._stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "retried_deliveries": 0
        }
    
    async def start(self, worker_count: int = 3) -> None:
        """Start webhook manager with delivery workers.
        
        Args:
            worker_count: Number of delivery worker tasks
        """
        self.logger.info(f"Starting webhook manager with {worker_count} workers")
        
        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        # Start delivery workers
        for i in range(worker_count):
            task = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        self.logger.info("Webhook manager started successfully")
    
    async def stop(self) -> None:
        """Stop webhook manager and all workers."""
        self.logger.info("Stopping webhook manager")
        
        # Set shutdown event
        self._shutdown_event.set()
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
        
        self.logger.info("Webhook manager stopped")
    
    def register_webhook(self, config: WebhookConfig) -> UUID:
        """Register a new webhook.
        
        Args:
            config: Webhook configuration
            
        Returns:
            Webhook ID
        """
        webhook_id = uuid4()
        self._webhooks[webhook_id] = config
        
        self.logger.info(f"Registered webhook: {config.name} ({webhook_id})")
        return webhook_id
    
    def unregister_webhook(self, webhook_id: UUID) -> bool:
        """Unregister a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            True if webhook was unregistered
        """
        if webhook_id in self._webhooks:
            webhook_config = self._webhooks[webhook_id]
            del self._webhooks[webhook_id]
            
            self.logger.info(f"Unregistered webhook: {webhook_config.name} ({webhook_id})")
            return True
        
        return False
    
    def get_webhook(self, webhook_id: UUID) -> Optional[WebhookConfig]:
        """Get webhook configuration.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Webhook configuration or None if not found
        """
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self, event_type: Optional[WebhookEvent] = None, status: Optional[WebhookStatus] = None) -> List[tuple[UUID, WebhookConfig]]:
        """List registered webhooks.
        
        Args:
            event_type: Filter by event type
            status: Filter by status
            
        Returns:
            List of (webhook_id, config) tuples
        """
        webhooks = []
        
        for webhook_id, config in self._webhooks.items():
            if event_type and event_type not in config.events:
                continue
            if status and config.status != status:
                continue
            webhooks.append((webhook_id, config))
        
        return webhooks
    
    async def send_webhook(self, event_type: WebhookEvent, event_data: Dict[str, Any], user_id: Optional[UUID] = None) -> List[UUID]:
        """Send webhook event to all registered webhooks.
        
        Args:
            event_type: Type of event
            event_data: Event data
            user_id: User ID for filtering (optional)
            
        Returns:
            List of delivery IDs
        """
        delivery_ids = []
        
        # Find matching webhooks
        matching_webhooks = []
        for webhook_id, config in self._webhooks.items():
            if config.status != WebhookStatus.ACTIVE:
                continue
            
            if event_type not in config.events:
                continue
            
            # Apply filters
            if not self._apply_filters(config.filters, event_data, user_id):
                continue
            
            matching_webhooks.append((webhook_id, config))
        
        # Create deliveries
        for webhook_id, config in matching_webhooks:
            delivery = WebhookDelivery(
                webhook_id=webhook_id,
                event_type=event_type,
                event_data=event_data,
                max_attempts=config.retry_count
            )
            
            self._deliveries[delivery.id] = delivery
            delivery_ids.append(delivery.id)
            
            # Queue for delivery
            await self._delivery_queue.put(delivery.id)
        
        if delivery_ids:
            self.logger.info(f"Queued {len(delivery_ids)} webhook deliveries for event {event_type.value}")
        
        return delivery_ids
    
    def get_delivery(self, delivery_id: UUID) -> Optional[WebhookDelivery]:
        """Get webhook delivery record.
        
        Args:
            delivery_id: Delivery ID
            
        Returns:
            Delivery record or None if not found
        """
        return self._deliveries.get(delivery_id)
    
    def list_deliveries(self, webhook_id: Optional[UUID] = None, status: Optional[WebhookDeliveryStatus] = None, limit: int = 100) -> List[WebhookDelivery]:
        """List webhook deliveries.
        
        Args:
            webhook_id: Filter by webhook ID
            status: Filter by delivery status
            limit: Maximum number of deliveries to return
            
        Returns:
            List of delivery records
        """
        deliveries = []
        
        for delivery in self._deliveries.values():
            if webhook_id and delivery.webhook_id != webhook_id:
                continue
            if status and delivery.status != status:
                continue
            deliveries.append(delivery)
            
            if len(deliveries) >= limit:
                break
        
        # Sort by creation time (newest first)
        deliveries.sort(key=lambda d: d.created_at, reverse=True)
        return deliveries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get webhook delivery statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "active_webhooks": len([w for w in self._webhooks.values() if w.status == WebhookStatus.ACTIVE]),
            "total_webhooks": len(self._webhooks),
            "pending_deliveries": len([d for d in self._deliveries.values() if d.status == WebhookDeliveryStatus.PENDING]),
            "queue_size": self._delivery_queue.qsize()
        }
    
    async def retry_delivery(self, delivery_id: UUID) -> bool:
        """Retry a failed webhook delivery.
        
        Args:
            delivery_id: Delivery ID
            
        Returns:
            True if delivery was queued for retry
        """
        delivery = self._deliveries.get(delivery_id)
        if not delivery:
            return False
        
        if delivery.status not in [WebhookDeliveryStatus.FAILED, WebhookDeliveryStatus.EXPIRED]:
            return False
        
        if delivery.attempts >= delivery.max_attempts:
            return False
        
        # Reset delivery status and queue for retry
        delivery.status = WebhookDeliveryStatus.RETRYING
        delivery.scheduled_at = datetime.utcnow()
        
        await self._delivery_queue.put(delivery_id)
        
        self.logger.info(f"Queued delivery {delivery_id} for retry (attempt {delivery.attempts + 1})")
        return True
    
    # Private methods
    
    async def _delivery_worker(self, worker_name: str) -> None:
        """Webhook delivery worker."""
        self.logger.info(f"Started webhook delivery worker: {worker_name}")
        
        while not self._shutdown_event.is_set():
            try:
                # Get delivery from queue
                try:
                    delivery_id = await asyncio.wait_for(
                        self._delivery_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process delivery
                await self._process_delivery(delivery_id)
                
            except Exception as e:
                self.logger.error(f"Error in webhook delivery worker {worker_name}: {e}")
        
        self.logger.info(f"Stopped webhook delivery worker: {worker_name}")
    
    async def _process_delivery(self, delivery_id: UUID) -> None:
        """Process a webhook delivery."""
        delivery = self._deliveries.get(delivery_id)
        if not delivery:
            return
        
        webhook_config = self._webhooks.get(delivery.webhook_id)
        if not webhook_config:
            delivery.status = WebhookDeliveryStatus.FAILED
            delivery.error_message = "Webhook configuration not found"
            return
        
        # Check if delivery has expired
        if datetime.utcnow() > delivery.expires_at:
            delivery.status = WebhookDeliveryStatus.EXPIRED
            delivery.error_message = "Delivery expired"
            return
        
        # Check if max attempts reached
        if delivery.attempts >= delivery.max_attempts:
            delivery.status = WebhookDeliveryStatus.FAILED
            delivery.error_message = f"Max attempts ({delivery.max_attempts}) reached"
            return
        
        try:
            # Increment attempt counter
            delivery.attempts += 1
            delivery.status = WebhookDeliveryStatus.PENDING
            
            # Prepare request
            payload = {
                "event_type": delivery.event_type.value,
                "event_data": delivery.event_data,
                "delivery_id": str(delivery.id),
                "timestamp": delivery.created_at.isoformat(),
                "request_id": delivery.request_id
            }
            
            # Create signature if secret is configured
            if webhook_config.secret:
                delivery.signature = self._create_signature(
                    webhook_config.secret,
                    json.dumps(payload, sort_keys=True)
                )
            
            # Prepare headers
            headers = {
                "Content-Type": webhook_config.content_type,
                "User-Agent": "AI-Learning-Accelerator-Webhook/1.0",
                "X-Webhook-Event": delivery.event_type.value,
                "X-Webhook-Delivery": str(delivery.id),
                "X-Webhook-Request": delivery.request_id,
                **webhook_config.headers
            }
            
            if delivery.signature:
                headers["X-Webhook-Signature"] = delivery.signature
            
            # Send request
            if self._http_client:
                response = await self._http_client.post(
                    webhook_config.url,
                    json=payload,
                    headers=headers,
                    timeout=webhook_config.timeout
                )
                
                # Record response
                delivery.response_status = response.status_code
                delivery.response_headers = dict(response.headers)
                delivery.response_body = response.text[:1000]  # Limit response body size
                
                # Check if delivery was successful
                if 200 <= response.status_code < 300:
                    delivery.status = WebhookDeliveryStatus.DELIVERED
                    delivery.delivered_at = datetime.utcnow()
                    self._stats["successful_deliveries"] += 1
                    
                    self.logger.info(f"Webhook delivered successfully: {delivery.id}")
                else:
                    delivery.status = WebhookDeliveryStatus.FAILED
                    delivery.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
                    
                    # Schedule retry if attempts remaining
                    if delivery.attempts < delivery.max_attempts:
                        retry_delay = webhook_config.retry_delay * (2 ** (delivery.attempts - 1))  # Exponential backoff
                        delivery.scheduled_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                        delivery.status = WebhookDeliveryStatus.RETRYING
                        
                        # Re-queue for retry
                        await asyncio.sleep(retry_delay)
                        await self._delivery_queue.put(delivery_id)
                        
                        self._stats["retried_deliveries"] += 1
                        self.logger.info(f"Webhook delivery scheduled for retry: {delivery.id} (attempt {delivery.attempts})")
                    else:
                        self._stats["failed_deliveries"] += 1
                        self.logger.error(f"Webhook delivery failed permanently: {delivery.id}")
            
            self._stats["total_deliveries"] += 1
            
        except Exception as e:
            delivery.status = WebhookDeliveryStatus.FAILED
            delivery.error_message = str(e)
            self._stats["failed_deliveries"] += 1
            
            self.logger.error(f"Error processing webhook delivery {delivery.id}: {e}")
    
    def _apply_filters(self, filters: Dict[str, Any], event_data: Dict[str, Any], user_id: Optional[UUID]) -> bool:
        """Apply webhook filters to event data.
        
        Args:
            filters: Webhook filters
            event_data: Event data
            user_id: User ID
            
        Returns:
            True if event passes filters
        """
        if not filters:
            return True
        
        # User ID filter
        if "user_ids" in filters:
            if not user_id or str(user_id) not in filters["user_ids"]:
                return False
        
        # Field filters
        for field_path, expected_value in filters.items():
            if field_path == "user_ids":
                continue
            
            # Get field value from event data
            field_value = self._get_nested_field(event_data, field_path)
            
            if field_value != expected_value:
                return False
        
        return True
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value from data using dot notation.
        
        Args:
            data: Data dictionary
            field_path: Field path (e.g., "user.profile.level")
            
        Returns:
            Field value or None if not found
        """
        try:
            value = data
            for key in field_path.split("."):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def _create_signature(self, secret: str, payload: str) -> str:
        """Create HMAC signature for webhook payload.
        
        Args:
            secret: Webhook secret
            payload: Payload string
            
        Returns:
            HMAC signature
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"


# Global webhook manager instance
webhook_manager = WebhookManager()