#!/usr/bin/env python3
"""Demo deployment for AI Learning Accelerator."""

import asyncio
import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Main demo deployment function."""
    print("🇮🇳 AI Learning Accelerator - Demo Deployment")
    print("=" * 60)
    
    try:
        # Test basic imports
        print("1. Testing core imports...")
        from ai_learning_accelerator.config import get_settings
        from ai_learning_accelerator.logging_config import configure_logging, get_logger
        
        # Configure logging
        configure_logging()
        logger = get_logger(__name__)
        
        settings = get_settings()
        print(f"✓ Settings loaded: {settings.app_name} v{settings.app_version}")
        
        # Test FastAPI app
        print("2. Testing FastAPI application...")
        from ai_learning_accelerator.main import app
        print(f"✓ FastAPI app created: {app.title}")
        
        # Test database connection (without initialization)
        print("3. Testing database configuration...")
        print(f"✓ Database URL configured: {settings.database_url[:20]}...")
        
        print("=" * 60)
        print("✅ Demo deployment successful!")
        print(f"🚀 Starting server on http://{settings.api_host}:{settings.api_port}")
        print("📚 Available endpoints:")
        print(f"  - Health Check: http://{settings.api_host}:{settings.api_port}/api/v1/health/")
        print(f"  - API Docs: http://{settings.api_host}:{settings.api_port}/docs")
        print(f"  - Root: http://{settings.api_host}:{settings.api_port}/")
        print("=" * 60)
        
        # Start the server
        config = uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_config=None,  # Use our custom logging
            reload=False
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)