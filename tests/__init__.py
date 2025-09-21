
"""
Test suite for distributed training framework

This package contains all tests:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance benchmarking tests
"""

import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__all__ = []

# Test utilities
def setup_test_environment():
    """Setup test environment"""
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

def cleanup_test_environment():
    """Cleanup test environment"""
    for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        os.environ.pop(key, None)
