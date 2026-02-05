"""Deployment configuration management for AI Learning Accelerator."""

import os
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

from pydantic_settings import BaseSettings
from pydantic import Field, validator

from ..logging_config import get_logger

logger = get_logger(__name__)


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ServiceMode(str, Enum):
    """Service deployment modes."""
    STANDALONE = "standalone"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"


@dataclass
class ServiceConfig:
    """Configuration for individual services."""
    name: str
    enabled: bool = True
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    dependencies: List[str] = field(default_factory=list)


class DeploymentSettings(BaseSettings):
    """Deployment-specific settings."""
    
    # Environment
    environment: DeploymentEnvironment = Field(
        default=DeploymentEnvironment.DEVELOPMENT,
        env="DEPLOYMENT_ENVIRONMENT"
    )
    
    # Service configuration
    service_mode: ServiceMode = Field(
        default=ServiceMode.STANDALONE,
        env="SERVICE_MODE"
    )
    
    # Scaling
    auto_scaling_enabled: bool = Field(default=False, env="AUTO_SCALING_ENABLED")
    min_replicas: int = Field(default=1, env="MIN_REPLICAS")
    max_replicas: int = Field(default=10, env="MAX_REPLICAS")
    target_cpu_utilization: int = Field(default=70, env="TARGET_CPU_UTILIZATION")
    
    # Load balancing
    load_balancer_enabled: bool = Field(default=False, env="LOAD_BALANCER_ENABLED")
    load_balancer_type: str = Field(default="round_robin", env="LOAD_BALANCER_TYPE")
    
    # Monitoring
    monitoring_enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    metrics_collection_interval: int = Field(default=30, env="METRICS_COLLECTION_INTERVAL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    tls_enabled: bool = Field(default=False, env="TLS_ENABLED")
    tls_cert_path: Optional[str] = Field(default=None, env="TLS_CERT_PATH")
    tls_key_path: Optional[str] = Field(default=None, env="TLS_KEY_PATH")
    
    # Database
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    # Redis/Cache
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # AI/ML
    model_cache_enabled: bool = Field(default=True, env="MODEL_CACHE_ENABLED")
    model_cache_size: int = Field(default=1000, env="MODEL_CACHE_SIZE")
    gpu_enabled: bool = Field(default=False, env="GPU_ENABLED")
    gpu_memory_limit: Optional[str] = Field(default=None, env="GPU_MEMORY_LIMIT")
    
    # External services
    external_api_timeout: int = Field(default=30, env="EXTERNAL_API_TIMEOUT")
    external_api_retries: int = Field(default=3, env="EXTERNAL_API_RETRIES")
    
    # Feature flags
    feature_flags: Dict[str, bool] = Field(
        default_factory=lambda: {
            "advanced_analytics": True,
            "real_time_collaboration": False,
            "experimental_ai_models": False,
            "plugin_system": True,
            "workflow_integration": True
        },
        env="FEATURE_FLAGS"
    )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate deployment environment."""
        if v not in DeploymentEnvironment:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator('feature_flags', pre=True)
    def parse_feature_flags(cls, v):
        """Parse feature flags from environment variable."""
        if isinstance(v, str):
            # Parse comma-separated key=value pairs
            flags = {}
            for flag in v.split(','):
                if '=' in flag:
                    key, value = flag.split('=', 1)
                    flags[key.strip()] = value.strip().lower() == 'true'
            return flags
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DeploymentManager:
    """Manages deployment configuration and service orchestration."""
    
    def __init__(self):
        self.settings = DeploymentSettings()
        self.services = self._initialize_service_configs()
        
    def _initialize_service_configs(self) -> Dict[str, ServiceConfig]:
        """Initialize service configurations based on deployment mode."""
        base_services = {
            "api_gateway": ServiceConfig(
                name="api_gateway",
                enabled=True,
                replicas=2 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "500m",
                    "memory": "512Mi",
                    "cpu_limit": "1000m",
                    "memory_limit": "1Gi"
                },
                health_check_path="/health"
            ),
            "learning_engine": ServiceConfig(
                name="learning_engine",
                enabled=True,
                replicas=3 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "1000m",
                    "memory": "2Gi",
                    "cpu_limit": "2000m",
                    "memory_limit": "4Gi"
                },
                environment_variables={
                    "MODEL_CACHE_ENABLED": str(self.settings.model_cache_enabled),
                    "GPU_ENABLED": str(self.settings.gpu_enabled)
                },
                dependencies=["database", "redis"] if self.settings.redis_enabled else ["database"]
            ),
            "debug_assistant": ServiceConfig(
                name="debug_assistant",
                enabled=True,
                replicas=2 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "800m",
                    "memory": "1.5Gi",
                    "cpu_limit": "1500m",
                    "memory_limit": "3Gi"
                },
                dependencies=["learning_engine", "context_analyzer"]
            ),
            "context_analyzer": ServiceConfig(
                name="context_analyzer",
                enabled=True,
                replicas=2 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "600m",
                    "memory": "1Gi",
                    "cpu_limit": "1200m",
                    "memory_limit": "2Gi"
                },
                dependencies=["database"]
            ),
            "automation_engine": ServiceConfig(
                name="automation_engine",
                enabled=self.settings.feature_flags.get("workflow_integration", True),
                replicas=1,
                resources={
                    "cpu": "400m",
                    "memory": "512Mi",
                    "cpu_limit": "800m",
                    "memory_limit": "1Gi"
                },
                dependencies=["context_analyzer"]
            ),
            "content_service": ServiceConfig(
                name="content_service",
                enabled=True,
                replicas=2 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "500m",
                    "memory": "1Gi",
                    "cpu_limit": "1000m",
                    "memory_limit": "2Gi"
                },
                dependencies=["database"]
            ),
            "analytics_service": ServiceConfig(
                name="analytics_service",
                enabled=self.settings.feature_flags.get("advanced_analytics", True),
                replicas=1,
                resources={
                    "cpu": "300m",
                    "memory": "512Mi",
                    "cpu_limit": "600m",
                    "memory_limit": "1Gi"
                },
                dependencies=["database", "redis"] if self.settings.redis_enabled else ["database"]
            ),
            "privacy_service": ServiceConfig(
                name="privacy_service",
                enabled=True,
                replicas=2 if self.settings.environment == DeploymentEnvironment.PRODUCTION else 1,
                resources={
                    "cpu": "200m",
                    "memory": "256Mi",
                    "cpu_limit": "400m",
                    "memory_limit": "512Mi"
                },
                dependencies=["database"]
            )
        }
        
        # Add infrastructure services
        infrastructure_services = {
            "database": ServiceConfig(
                name="database",
                enabled=True,
                replicas=1,
                resources={
                    "cpu": "1000m",
                    "memory": "2Gi",
                    "cpu_limit": "2000m",
                    "memory_limit": "4Gi"
                },
                environment_variables={
                    "POSTGRES_DB": "ai_learning_accelerator",
                    "POSTGRES_USER": "postgres",
                    "POSTGRES_PASSWORD": os.getenv("DATABASE_PASSWORD", "password")
                }
            )
        }
        
        if self.settings.redis_enabled:
            infrastructure_services["redis"] = ServiceConfig(
                name="redis",
                enabled=True,
                replicas=1,
                resources={
                    "cpu": "200m",
                    "memory": "256Mi",
                    "cpu_limit": "400m",
                    "memory_limit": "512Mi"
                }
            )
        
        # Combine all services
        all_services = {**infrastructure_services, **base_services}
        
        # Filter enabled services
        return {name: config for name, config in all_services.items() if config.enabled}
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a specific service."""
        return self.services.get(service_name)
    
    def get_enabled_services(self) -> List[str]:
        """Get list of enabled service names."""
        return list(self.services.keys())
    
    def get_deployment_manifest(self) -> Dict[str, Any]:
        """Generate deployment manifest for the current configuration."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "ai-learning-accelerator-config",
                "namespace": "default"
            },
            "data": {
                "environment": self.settings.environment.value,
                "service_mode": self.settings.service_mode.value,
                "monitoring_enabled": str(self.settings.monitoring_enabled),
                "auto_scaling_enabled": str(self.settings.auto_scaling_enabled),
                "feature_flags": str(self.settings.feature_flags)
            }
        }
    
    def get_docker_compose_config(self) -> Dict[str, Any]:
        """Generate Docker Compose configuration."""
        services = {}
        
        for service_name, config in self.services.items():
            service_config = {
                "image": f"ai-learning-accelerator/{service_name}:latest",
                "restart": "unless-stopped",
                "environment": config.environment_variables,
                "healthcheck": {
                    "test": f"curl -f http://localhost:8000{config.health_check_path} || exit 1",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "40s"
                }
            }
            
            # Add resource limits if specified
            if config.resources:
                service_config["deploy"] = {
                    "resources": {
                        "limits": {
                            "cpus": config.resources.get("cpu_limit", "1.0"),
                            "memory": config.resources.get("memory_limit", "1G")
                        },
                        "reservations": {
                            "cpus": config.resources.get("cpu", "0.5"),
                            "memory": config.resources.get("memory", "512M")
                        }
                    },
                    "replicas": config.replicas
                }
            
            # Add dependencies
            if config.dependencies:
                service_config["depends_on"] = config.dependencies
            
            services[service_name] = service_config
        
        return {
            "version": "3.8",
            "services": services,
            "networks": {
                "ai-learning-accelerator": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {} if self.settings.redis_enabled else None
            }
        }
    
    def get_kubernetes_manifests(self) -> List[Dict[str, Any]]:
        """Generate Kubernetes deployment manifests."""
        manifests = []
        
        # ConfigMap
        manifests.append(self.get_deployment_manifest())
        
        # Services and Deployments
        for service_name, config in self.services.items():
            # Deployment
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"ai-learning-accelerator-{service_name}",
                    "labels": {
                        "app": "ai-learning-accelerator",
                        "component": service_name
                    }
                },
                "spec": {
                    "replicas": config.replicas,
                    "selector": {
                        "matchLabels": {
                            "app": "ai-learning-accelerator",
                            "component": service_name
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "ai-learning-accelerator",
                                "component": service_name
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": service_name,
                                "image": f"ai-learning-accelerator/{service_name}:latest",
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": k, "value": v}
                                    for k, v in config.environment_variables.items()
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": config.resources.get("cpu", "100m"),
                                        "memory": config.resources.get("memory", "128Mi")
                                    },
                                    "limits": {
                                        "cpu": config.resources.get("cpu_limit", "500m"),
                                        "memory": config.resources.get("memory_limit", "512Mi")
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": config.health_check_path,
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": config.health_check_path,
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }]
                        }
                    }
                }
            }
            manifests.append(deployment)
            
            # Service
            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"ai-learning-accelerator-{service_name}",
                    "labels": {
                        "app": "ai-learning-accelerator",
                        "component": service_name
                    }
                },
                "spec": {
                    "selector": {
                        "app": "ai-learning-accelerator",
                        "component": service_name
                    },
                    "ports": [{
                        "port": 80,
                        "targetPort": 8000,
                        "protocol": "TCP"
                    }],
                    "type": "ClusterIP"
                }
            }
            manifests.append(service)
        
        return manifests
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current deployment configuration."""
        issues = []
        warnings = []
        
        # Check for production readiness
        if self.settings.environment == DeploymentEnvironment.PRODUCTION:
            if not self.settings.tls_enabled:
                issues.append("TLS should be enabled in production")
            
            if self.settings.log_level == "DEBUG":
                warnings.append("Debug logging should not be used in production")
            
            if not self.settings.monitoring_enabled:
                issues.append("Monitoring should be enabled in production")
            
            # Check service replicas
            for service_name, config in self.services.items():
                if config.replicas < 2 and service_name not in ["database", "redis"]:
                    warnings.append(f"Service {service_name} should have multiple replicas in production")
        
        # Check dependencies
        for service_name, config in self.services.items():
            for dependency in config.dependencies:
                if dependency not in self.services:
                    issues.append(f"Service {service_name} depends on {dependency} which is not enabled")
        
        # Check resource allocation
        total_cpu = sum(
            float(config.resources.get("cpu", "100m").replace("m", "")) / 1000
            for config in self.services.values()
            if config.resources.get("cpu")
        )
        
        total_memory = sum(
            self._parse_memory(config.resources.get("memory", "128Mi"))
            for config in self.services.values()
            if config.resources.get("memory")
        )
        
        if total_cpu > 16:  # Assuming reasonable limits
            warnings.append(f"Total CPU allocation ({total_cpu:.2f} cores) may be too high")
        
        if total_memory > 32 * 1024:  # 32GB in MB
            warnings.append(f"Total memory allocation ({total_memory/1024:.2f}GB) may be too high")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_services": len(self.services),
            "total_cpu_cores": total_cpu,
            "total_memory_gb": total_memory / 1024
        }
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to MB."""
        if memory_str.endswith("Gi"):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith("Mi"):
            return float(memory_str[:-2])
        elif memory_str.endswith("G"):
            return float(memory_str[:-1]) * 1024
        elif memory_str.endswith("M"):
            return float(memory_str[:-1])
        else:
            return float(memory_str)


# Global deployment manager instance
deployment_manager = DeploymentManager()