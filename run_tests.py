#!/usr/bin/env python3
"""
Test runner for the indexfund metrics tests using pytest.
"""

import os
import sys

import pytest

if __name__ == "__main__":
    # Add parent directory to path so that 'indexfund' can be imported
    project_root = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(project_root)
    sys.path.insert(0, parent_dir)

    # Run pytest with verbosity
    exit_code = pytest.main(["-v"])

    # Exit with the pytest exit code
    sys.exit(exit_code)
