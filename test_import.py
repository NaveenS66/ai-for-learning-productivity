#!/usr/bin/env python3
"""Test import of LearningPathStatus."""

import sys
sys.path.insert(0, 'src')

try:
    print("Importing learning module...")
    import ai_learning_accelerator.models.learning as learning_module
    print("Learning module imported successfully")
    
    print("Available attributes:")
    attrs = [attr for attr in dir(learning_module) if not attr.startswith('_')]
    for attr in sorted(attrs):
        print(f"  - {attr}")
    
    print("\nTrying to import LearningPathStatus...")
    from ai_learning_accelerator.models.learning import LearningPathStatus
    print(f"Success! LearningPathStatus = {LearningPathStatus}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()