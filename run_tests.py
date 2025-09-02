#!/usr/bin/env python3
"""
Coscientist Test Runner

Convenient script to run different categories of tests with appropriate configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ pytest not found. Install with: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Coscientist test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit         - Fast unit tests for individual components
  integration  - Medium-speed integration tests  
  e2e          - Slow end-to-end scenario tests
  performance  - Performance and scalability tests
  all          - All tests except real API tests
  real-api     - Tests requiring real API keys (slow)
  
Examples:
  python run_tests.py unit                    # Quick unit tests
  python run_tests.py integration --coverage  # Integration tests with coverage
  python run_tests.py e2e --verbose          # E2E tests with verbose output
  python run_tests.py performance --no-slow  # Performance tests, skip slow ones
  python run_tests.py all --parallel         # All tests in parallel
        """
    )
    
    parser.add_argument(
        "category", 
        choices=["unit", "integration", "e2e", "performance", "all", "real-api"],
        help="Test category to run"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile test execution time"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Drop into debugger on first failure"
    )
    
    args = parser.parse_args()
    
    # Verify we're in the right directory
    if not Path("tests").exists():
        print("❌ tests/ directory not found. Run from project root.")
        sys.exit(1)
    
    # Base command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")  # Quiet by default
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=coscientist",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Add profiling
    if args.profile:
        cmd.append("--durations=10")
    
    # Add debug mode
    if args.debug:
        cmd.append("--pdb")
    
    # Test category specific configurations
    if args.category == "unit":
        cmd.extend([
            "tests/test_*_agent.py",
            "-m", "unit"
        ])
        description = "Unit Tests (Fast)"
        
    elif args.category == "integration":
        cmd.extend([
            "tests/test_*_integration.py", 
            "-m", "integration and not requires_api"
        ])
        description = "Integration Tests (Medium)"
        
    elif args.category == "e2e":
        cmd.extend([
            "tests/test_e2e_scenarios.py",
            "-m", "e2e and mock"
        ])
        if args.no_slow:
            cmd.extend(["-m", "e2e and mock and not slow"])
        description = "End-to-End Tests (Slow)"
        
    elif args.category == "performance":
        cmd.extend([
            "tests/test_performance.py",
            "-m", "performance"
        ])
        if args.no_slow:
            cmd.extend(["-m", "performance and not slow"])
        description = "Performance Tests"
        
    elif args.category == "all":
        cmd.extend([
            "tests/",
            "-m", "not requires_api"
        ])
        if args.no_slow:
            cmd.extend(["-m", "not requires_api and not slow"])
        description = "All Tests (No Real APIs)"
        
    elif args.category == "real-api":
        cmd.extend([
            "tests/",
            "-m", "requires_api"
        ])
        description = "Real API Tests (Requires API Keys)"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated: htmlcov/index.html")
    
    if not success:
        print(f"\n🔧 Troubleshooting tips:")
        print(f"  • Check test output above for specific failures")
        print(f"  • Run single test: pytest tests/test_file.py::test_name -v")
        print(f"  • Debug mode: python run_tests.py {args.category} --debug")
        print(f"  • Check environment: python -c 'import coscientist'")
        sys.exit(1)
    
    print(f"\n🎉 Test run completed successfully!")


if __name__ == "__main__":
    main()