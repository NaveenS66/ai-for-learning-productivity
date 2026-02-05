#!/usr/bin/env python3
"""Temporary fix for import issues."""

import sys
sys.path.insert(0, 'src')

# Test individual imports
print("Testing imports...")

try:
    print("1. Testing workflow_detector...")
    from ai_learning_accelerator.integrations.workflow_detector import DetectedWorkflow
    print("✓ workflow_detector works")
except Exception as e:
    print(f"✗ workflow_detector failed: {e}")

try:
    print("2. Testing workflow_adapter module loading...")
    import ai_learning_accelerator.integrations.workflow_adapter as wa_module
    print("✓ workflow_adapter module loads")
    print(f"Available attributes: {[attr for attr in dir(wa_module) if not attr.startswith('_')]}")
except Exception as e:
    print(f"✗ workflow_adapter module failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing WorkflowAdapter class import...")
    from ai_learning_accelerator.integrations.workflow_adapter import WorkflowAdapter
    print("✓ WorkflowAdapter imports successfully")
except Exception as e:
    print(f"✗ WorkflowAdapter import failed: {e}")
    import traceback
    traceback.print_exc()