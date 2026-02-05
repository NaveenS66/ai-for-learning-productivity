#!/usr/bin/env python3
"""Simple deployment test."""

import sys
import asyncio
sys.path.insert(0, 'src')

async def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        from ai_learning_accelerator.config import get_settings
        settings = get_settings()
        print(f"✓ Settings loaded: {settings.app_name}")
    except Exception as e:
        print(f"✗ Settings failed: {e}")
        return False
    
    try:
        from ai_learning_accelerator.logging_config import configure_logging, get_logger
        configure_logging()
        logger = get_logger(__name__)
        logger.info("Logging configured successfully")
        print("✓ Logging configured")
    except Exception as e:
        print(f"✗ Logging failed: {e}")
        return False
    
    try:
        from ai_learning_accelerator.database import init_db
        print("✓ Database module imported")
    except Exception as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    return True

async def test_fastapi_app():
    """Test FastAPI app creation."""
    print("Testing FastAPI app...")
    
    try:
        from ai_learning_accelerator.main import app
        print(f"✓ FastAPI app created: {app.title}")
        return True
    except Exception as e:
        print(f"✗ FastAPI app failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🧪 AI Learning Accelerator - Simple Deployment Test")
    print("=" * 50)
    
    # Test basic imports
    if not await test_basic_imports():
        print("❌ Basic imports failed")
        return
    
    # Test FastAPI app
    if not await test_fastapi_app():
        print("❌ FastAPI app creation failed")
        return
    
    print("=" * 50)
    print("✅ All basic tests passed!")
    print("🚀 System is ready for deployment")

if __name__ == "__main__":
    asyncio.run(main())