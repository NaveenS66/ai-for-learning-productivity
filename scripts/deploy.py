#!/usr/bin/env python3
"""Deployment script for AI Learning Accelerator."""

import asyncio
import argparse
import json
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_learning_accelerator.config.deployment import deployment_manager, DeploymentEnvironment
from ai_learning_accelerator.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


class DeploymentOrchestrator:
    """Orchestrates deployment of AI Learning Accelerator."""
    
    def __init__(self, environment: str, platform: str):
        self.environment = DeploymentEnvironment(environment)
        self.platform = platform
        self.deployment_manager = deployment_manager
        
        # Override environment in deployment manager
        self.deployment_manager.settings.environment = self.environment
        
        # Reinitialize services with new environment
        self.deployment_manager.services = self.deployment_manager._initialize_service_configs()
    
    async def deploy(self, dry_run: bool = False) -> Dict[str, Any]:
        """Deploy the system to the specified platform."""
        logger.info(f"Starting deployment to {self.platform} ({self.environment.value})")
        
        # Validate configuration
        validation = self.deployment_manager.validate_configuration()
        if not validation["valid"]:
            logger.error("Configuration validation failed:")
            for issue in validation["issues"]:
                logger.error(f"  - {issue}")
            return {"status": "failed", "reason": "configuration_invalid", "validation": validation}
        
        if validation["warnings"]:
            logger.warning("Configuration warnings:")
            for warning in validation["warnings"]:
                logger.warning(f"  - {warning}")
        
        if dry_run:
            logger.info("Dry run mode - no actual deployment will be performed")
            return await self._dry_run_deployment()
        
        # Perform actual deployment
        if self.platform == "docker-compose":
            return await self._deploy_docker_compose()
        elif self.platform == "kubernetes":
            return await self._deploy_kubernetes()
        elif self.platform == "local":
            return await self._deploy_local()
        else:
            return {"status": "failed", "reason": f"unsupported_platform: {self.platform}"}
    
    async def _dry_run_deployment(self) -> Dict[str, Any]:
        """Perform a dry run deployment."""
        logger.info("Performing dry run deployment")
        
        # Generate configuration files
        if self.platform == "docker-compose":
            config = self.deployment_manager.get_docker_compose_config()
            logger.info("Generated Docker Compose configuration:")
            logger.info(yaml.dump(config, default_flow_style=False))
        
        elif self.platform == "kubernetes":
            manifests = self.deployment_manager.get_kubernetes_manifests()
            logger.info("Generated Kubernetes manifests:")
            for i, manifest in enumerate(manifests):
                logger.info(f"--- Manifest {i+1} ---")
                logger.info(yaml.dump(manifest, default_flow_style=False))
        
        return {
            "status": "success",
            "dry_run": True,
            "platform": self.platform,
            "environment": self.environment.value,
            "services": list(self.deployment_manager.services.keys()),
            "validation": self.deployment_manager.validate_configuration()
        }
    
    async def _deploy_docker_compose(self) -> Dict[str, Any]:
        """Deploy using Docker Compose."""
        logger.info("Deploying with Docker Compose")
        
        try:
            # Generate docker-compose.yml
            config = self.deployment_manager.get_docker_compose_config()
            compose_file = Path("docker-compose.yml")
            
            with open(compose_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Generated {compose_file}")
            
            # Build images
            logger.info("Building Docker images...")
            build_result = subprocess.run(
                ["docker-compose", "build"],
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                logger.error(f"Docker build failed: {build_result.stderr}")
                return {"status": "failed", "reason": "build_failed", "error": build_result.stderr}
            
            # Deploy services
            logger.info("Starting services...")
            up_result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True
            )
            
            if up_result.returncode != 0:
                logger.error(f"Docker Compose up failed: {up_result.stderr}")
                return {"status": "failed", "reason": "deploy_failed", "error": up_result.stderr}
            
            # Wait for services to be healthy
            await self._wait_for_services_healthy()
            
            logger.info("Docker Compose deployment completed successfully")
            
            return {
                "status": "success",
                "platform": "docker-compose",
                "environment": self.environment.value,
                "services": list(self.deployment_manager.services.keys()),
                "compose_file": str(compose_file)
            }
            
        except Exception as e:
            logger.error(f"Docker Compose deployment failed: {e}")
            return {"status": "failed", "reason": "deployment_error", "error": str(e)}
    
    async def _deploy_kubernetes(self) -> Dict[str, Any]:
        """Deploy to Kubernetes."""
        logger.info("Deploying to Kubernetes")
        
        try:
            # Generate manifests
            manifests = self.deployment_manager.get_kubernetes_manifests()
            manifest_files = []
            
            # Create manifests directory
            manifests_dir = Path("k8s-manifests")
            manifests_dir.mkdir(exist_ok=True)
            
            # Write manifest files
            for i, manifest in enumerate(manifests):
                kind = manifest.get("kind", "Unknown").lower()
                name = manifest.get("metadata", {}).get("name", f"manifest-{i}")
                filename = f"{kind}-{name}.yaml"
                filepath = manifests_dir / filename
                
                with open(filepath, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False)
                
                manifest_files.append(str(filepath))
                logger.info(f"Generated {filepath}")
            
            # Apply manifests
            logger.info("Applying Kubernetes manifests...")
            for manifest_file in manifest_files:
                apply_result = subprocess.run(
                    ["kubectl", "apply", "-f", manifest_file],
                    capture_output=True,
                    text=True
                )
                
                if apply_result.returncode != 0:
                    logger.error(f"Failed to apply {manifest_file}: {apply_result.stderr}")
                    return {
                        "status": "failed",
                        "reason": "manifest_apply_failed",
                        "error": apply_result.stderr,
                        "manifest": manifest_file
                    }
            
            # Wait for deployments to be ready
            await self._wait_for_kubernetes_deployments()
            
            logger.info("Kubernetes deployment completed successfully")
            
            return {
                "status": "success",
                "platform": "kubernetes",
                "environment": self.environment.value,
                "services": list(self.deployment_manager.services.keys()),
                "manifest_files": manifest_files
            }
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return {"status": "failed", "reason": "deployment_error", "error": str(e)}
    
    async def _deploy_local(self) -> Dict[str, Any]:
        """Deploy locally for development."""
        logger.info("Deploying locally")
        
        try:
            # Start database
            logger.info("Starting local database...")
            db_result = subprocess.run(
                ["docker", "run", "-d", "--name", "ai-learning-db",
                 "-e", "POSTGRES_DB=ai_learning_accelerator",
                 "-e", "POSTGRES_USER=postgres",
                 "-e", "POSTGRES_PASSWORD=password",
                 "-p", "5432:5432",
                 "postgres:13"],
                capture_output=True,
                text=True
            )
            
            if db_result.returncode != 0 and "already in use" not in db_result.stderr:
                logger.error(f"Failed to start database: {db_result.stderr}")
                return {"status": "failed", "reason": "database_start_failed", "error": db_result.stderr}
            
            # Run database migrations
            logger.info("Running database migrations...")
            migration_result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True
            )
            
            if migration_result.returncode != 0:
                logger.warning(f"Migration warning: {migration_result.stderr}")
            
            # Start the application
            logger.info("Starting AI Learning Accelerator...")
            
            return {
                "status": "success",
                "platform": "local",
                "environment": self.environment.value,
                "database": "postgresql://postgres:password@localhost:5432/ai_learning_accelerator",
                "api_url": "http://localhost:8000"
            }
            
        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            return {"status": "failed", "reason": "deployment_error", "error": str(e)}
    
    async def _wait_for_services_healthy(self, timeout: int = 300):
        """Wait for Docker Compose services to be healthy."""
        logger.info("Waiting for services to be healthy...")
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check service health
            ps_result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if ps_result.returncode == 0:
                try:
                    services = json.loads(ps_result.stdout)
                    if isinstance(services, dict):
                        services = [services]
                    
                    healthy_services = [
                        s for s in services
                        if s.get("Health") == "healthy" or s.get("State") == "running"
                    ]
                    
                    if len(healthy_services) == len(self.deployment_manager.services):
                        logger.info("All services are healthy")
                        return
                    
                    logger.info(f"Waiting... {len(healthy_services)}/{len(self.deployment_manager.services)} services healthy")
                    
                except json.JSONDecodeError:
                    pass
            
            await asyncio.sleep(10)
        
        logger.warning("Timeout waiting for services to be healthy")
    
    async def _wait_for_kubernetes_deployments(self, timeout: int = 300):
        """Wait for Kubernetes deployments to be ready."""
        logger.info("Waiting for Kubernetes deployments to be ready...")
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check deployment status
            status_result = subprocess.run(
                ["kubectl", "get", "deployments", "-l", "app=ai-learning-accelerator", "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if status_result.returncode == 0:
                try:
                    deployments = json.loads(status_result.stdout)
                    items = deployments.get("items", [])
                    
                    ready_deployments = [
                        d for d in items
                        if d.get("status", {}).get("readyReplicas", 0) >= d.get("spec", {}).get("replicas", 1)
                    ]
                    
                    if len(ready_deployments) == len([s for s in self.deployment_manager.services.values() if s.name != "database"]):
                        logger.info("All deployments are ready")
                        return
                    
                    logger.info(f"Waiting... {len(ready_deployments)} deployments ready")
                    
                except json.JSONDecodeError:
                    pass
            
            await asyncio.sleep(10)
        
        logger.warning("Timeout waiting for deployments to be ready")
    
    async def undeploy(self) -> Dict[str, Any]:
        """Undeploy the system."""
        logger.info(f"Undeploying from {self.platform}")
        
        try:
            if self.platform == "docker-compose":
                result = subprocess.run(
                    ["docker-compose", "down", "-v"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Docker Compose down failed: {result.stderr}")
                    return {"status": "failed", "error": result.stderr}
            
            elif self.platform == "kubernetes":
                # Delete all resources with the app label
                result = subprocess.run(
                    ["kubectl", "delete", "all", "-l", "app=ai-learning-accelerator"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Kubernetes delete failed: {result.stderr}")
                    return {"status": "failed", "error": result.stderr}
            
            elif self.platform == "local":
                # Stop local database
                subprocess.run(["docker", "stop", "ai-learning-db"], capture_output=True)
                subprocess.run(["docker", "rm", "ai-learning-db"], capture_output=True)
            
            logger.info("Undeployment completed successfully")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Undeployment failed: {e}")
            return {"status": "failed", "error": str(e)}


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy AI Learning Accelerator")
    parser.add_argument(
        "action",
        choices=["deploy", "undeploy", "validate", "generate-config"],
        help="Action to perform"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    parser.add_argument(
        "--platform",
        choices=["local", "docker-compose", "kubernetes"],
        default="local",
        help="Deployment platform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual deployment"
    )
    parser.add_argument(
        "--output",
        help="Output file for generated configuration"
    )
    
    args = parser.parse_args()
    
    orchestrator = DeploymentOrchestrator(args.environment, args.platform)
    
    if args.action == "deploy":
        result = await orchestrator.deploy(dry_run=args.dry_run)
    elif args.action == "undeploy":
        result = await orchestrator.undeploy()
    elif args.action == "validate":
        result = deployment_manager.validate_configuration()
    elif args.action == "generate-config":
        if args.platform == "docker-compose":
            config = deployment_manager.get_docker_compose_config()
            output_file = args.output or "docker-compose.yml"
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            result = {"status": "success", "output_file": output_file}
        elif args.platform == "kubernetes":
            manifests = deployment_manager.get_kubernetes_manifests()
            output_dir = Path(args.output or "k8s-manifests")
            output_dir.mkdir(exist_ok=True)
            
            files = []
            for i, manifest in enumerate(manifests):
                kind = manifest.get("kind", "Unknown").lower()
                name = manifest.get("metadata", {}).get("name", f"manifest-{i}")
                filename = f"{kind}-{name}.yaml"
                filepath = output_dir / filename
                
                with open(filepath, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False)
                files.append(str(filepath))
            
            result = {"status": "success", "output_files": files}
        else:
            result = {"status": "failed", "reason": "unsupported_platform_for_config_generation"}
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    if result.get("status") == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())