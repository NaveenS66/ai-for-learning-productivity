#!/usr/bin/env python3
"""Startup script for AI Learning Accelerator system initialization."""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_learning_accelerator.services.system_orchestrator import system_orchestrator
from ai_learning_accelerator.utils.monitoring import monitoring_system
from ai_learning_accelerator.logging_config import configure_logging, get_logger
from ai_learning_accelerator.database import init_db, close_db
from ai_learning_accelerator.config import get_settings

configure_logging()
logger = get_logger(__name__)


class SystemStartup:
    """Manages system startup and shutdown."""
    
    def __init__(self):
        self.settings = get_settings()
        self.shutdown_event = asyncio.Event()
        self.startup_complete = False
        
    async def initialize_system(self) -> bool:
        """Initialize the complete system."""
        logger.info("Starting AI Learning Accelerator system initialization")
        
        try:
            # Step 1: Initialize database
            logger.info("Initializing database connection...")
            await init_db()
            logger.info("Database connection established")
            
            # Step 2: Initialize monitoring system
            logger.info("Starting monitoring system...")
            await monitoring_system.start()
            logger.info("Monitoring system started")
            
            # Step 3: Initialize system orchestrator
            logger.info("Initializing system orchestrator...")
            orchestrator_result = await system_orchestrator.initialize_system()
            
            if orchestrator_result["status"] != "initialized":
                logger.error("System orchestrator initialization failed")
                logger.error(f"Result: {orchestrator_result}")
                return False
            
            logger.info("System orchestrator initialized successfully")
            logger.info(f"Services initialized: {orchestrator_result['successful_initializations']}/{orchestrator_result['total_services']}")
            
            # Step 4: Perform system health check
            logger.info("Performing initial system health check...")
            health_status = await system_orchestrator.get_system_status()
            
            if health_status["overall_status"] in ["critical", "unhealthy"]:
                logger.error(f"System health check failed: {health_status['overall_status']}")
                logger.error(f"Health details: {health_status}")
                return False
            
            logger.info(f"System health check passed: {health_status['overall_status']}")
            
            # Step 5: Log startup summary
            self._log_startup_summary(orchestrator_result, health_status)
            
            self.startup_complete = True
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            logger.exception("Initialization error details:")
            return False
    
    def _log_startup_summary(self, orchestrator_result: dict, health_status: dict):
        """Log startup summary information."""
        logger.info("=== AI Learning Accelerator Startup Summary ===")
        logger.info(f"Environment: {self.settings.debug and 'Development' or 'Production'}")
        logger.info(f"API Host: {self.settings.api_host}:{self.settings.api_port}")
        logger.info(f"Database: Connected")
        logger.info(f"Monitoring: {'Enabled' if monitoring_system._started else 'Disabled'}")
        logger.info(f"Services: {orchestrator_result['successful_initializations']}/{orchestrator_result['total_services']} initialized")
        logger.info(f"Overall Health: {health_status['overall_status']}")
        logger.info(f"Health Ratio: {health_status['health_ratio']:.2%}")
        
        # Log enabled services
        enabled_services = [name for name, status in health_status['services'].items() 
                          if status.get('status') == 'running']
        logger.info(f"Active Services: {', '.join(enabled_services)}")
        
        # Log any warnings
        failed_services = [name for name, status in health_status['services'].items() 
                         if status.get('status') == 'error']
        if failed_services:
            logger.warning(f"Failed Services: {', '.join(failed_services)}")
        
        logger.info("=== Startup Complete ===")
    
    async def shutdown_system(self):
        """Gracefully shutdown the system."""
        logger.info("Starting system shutdown...")
        
        try:
            # Step 1: Shutdown system orchestrator
            logger.info("Shutting down system orchestrator...")
            shutdown_result = await system_orchestrator.shutdown_system()
            
            if shutdown_result["status"] == "shutdown":
                logger.info("System orchestrator shutdown completed")
            else:
                logger.warning(f"System orchestrator shutdown issues: {shutdown_result}")
            
            # Step 2: Stop monitoring system
            logger.info("Stopping monitoring system...")
            await monitoring_system.stop()
            logger.info("Monitoring system stopped")
            
            # Step 3: Close database connections
            logger.info("Closing database connections...")
            await close_db()
            logger.info("Database connections closed")
            
            logger.info("System shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            logger.exception("Shutdown error details:")
    
    async def run_system_checks(self) -> bool:
        """Run comprehensive system checks."""
        logger.info("Running system checks...")
        
        try:
            # Check system status
            status = await system_orchestrator.get_system_status()
            
            logger.info(f"System Status: {status['overall_status']}")
            logger.info(f"Health Ratio: {status['health_ratio']:.2%}")
            logger.info(f"Services: {len(status['services'])}")
            logger.info(f"Background Tasks: {status['background_tasks']}")
            
            # Check individual services
            for service_name, service_status in status['services'].items():
                status_indicator = "✓" if service_status.get('status') == 'running' else "✗"
                logger.info(f"  {status_indicator} {service_name}: {service_status.get('status', 'unknown')}")
            
            # Check monitoring system
            monitoring_status = status['monitoring']
            logger.info(f"Monitoring: {monitoring_status['health']['status']}")
            logger.info(f"Active Alerts: {monitoring_status['alerts']['active_count']}")
            logger.info(f"Recent Requests: {monitoring_status['metrics']['recent_requests']}")
            logger.info(f"Error Rate: {monitoring_status['metrics']['error_rate']:.2%}")
            
            return status['overall_status'] in ['healthy', 'warning']
            
        except Exception as e:
            logger.error(f"System checks failed: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()


async def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="AI Learning Accelerator System Startup")
    parser.add_argument(
        "--mode",
        choices=["start", "check", "init-only"],
        default="start",
        help="Startup mode"
    )
    parser.add_argument(
        "--wait-for-shutdown",
        action="store_true",
        help="Wait for shutdown signal after startup"
    )
    
    args = parser.parse_args()
    
    startup = SystemStartup()
    
    if args.mode in ["start", "init-only"]:
        # Initialize system
        success = await startup.initialize_system()
        
        if not success:
            logger.error("System initialization failed")
            sys.exit(1)
        
        if args.mode == "init-only":
            logger.info("Initialization complete, exiting...")
            await startup.shutdown_system()
            sys.exit(0)
    
    if args.mode in ["start", "check"]:
        # Run system checks
        checks_passed = await startup.run_system_checks()
        
        if not checks_passed:
            logger.error("System checks failed")
            if args.mode == "check":
                sys.exit(1)
    
    if args.mode == "start":
        if args.wait_for_shutdown:
            # Setup signal handlers and wait
            startup.setup_signal_handlers()
            logger.info("System ready. Waiting for shutdown signal...")
            
            try:
                await startup.wait_for_shutdown()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
            finally:
                await startup.shutdown_system()
        else:
            logger.info("System initialization complete")
    
    logger.info("Startup script completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup script failed: {e}")
        logger.exception("Error details:")
        sys.exit(1)