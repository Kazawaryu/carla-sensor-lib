#!/usr/bin/env python3
"""
Test runner for CARLA sensor library unit tests.
Updated for CARLA 0.10.0 compatibility.

Usage:
    python run_tests.py
    python run_tests.py --specific test.sensor.test_SemanticLidarSensor
"""

import sys
import os
import unittest
import argparse

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test'))


def run_all_tests():
    """Run all unit tests in the test directory."""
    loader = unittest.TestLoader()
    start_dir = 'test'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_module):
    """Run a specific test module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Run CARLA sensor library tests')
    parser.add_argument('--specific', help='Run specific test module')
    args = parser.parse_args()
    
    print("CARLA Sensor Library Test Suite - CARLA 0.10.0")
    print("=" * 50)
    
    if args.specific:
        print(f"Running specific test: {args.specific}")
        success = run_specific_test(args.specific)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()