#!/usr/bin/env python3
"""
Test Suite Summary Generator

Analyzes the test suite and provides an overview of coverage and test organization.
"""

import ast
import os
from pathlib import Path
from collections import defaultdict


def count_lines_of_code(file_path):
    """Count lines of code in a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        return len(content.split('\n'))
    except Exception:
        return 0


def extract_test_functions(file_path):
    """Extract test function names and their markers."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        tests = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Extract markers
                markers = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if hasattr(decorator, 'attr'):
                            markers.append(decorator.attr)
                    elif isinstance(decorator, ast.Call):
                        if hasattr(decorator.func, 'attr'):
                            markers.append(decorator.func.attr)
                
                tests.append({
                    'name': node.name,
                    'markers': markers,
                    'line': node.lineno
                })
        
        return tests
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def analyze_test_suite():
    """Analyze the complete test suite."""
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ tests/ directory not found")
        return
    
    print("🧪 Coscientist Test Suite Analysis")
    print("=" * 60)
    
    # Find all test files
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No test files found")
        return
    
    # Analyze each file
    total_tests = 0
    total_lines = 0
    categories = defaultdict(list)
    marker_counts = defaultdict(int)
    
    for test_file in sorted(test_files):
        lines = count_lines_of_code(test_file)
        tests = extract_test_functions(test_file)
        
        total_lines += lines
        total_tests += len(tests)
        
        # Categorize by file type
        file_name = test_file.name
        if "_agent.py" in file_name:
            category = "Unit Tests"
        elif "_integration.py" in file_name:
            category = "Integration Tests"
        elif "_e2e" in file_name or "scenario" in file_name:
            category = "End-to-End Tests"
        elif "performance" in file_name:
            category = "Performance Tests"
        elif "cli" in file_name:
            category = "CLI Tests"
        else:
            category = "Other Tests"
        
        categories[category].append({
            'file': file_name,
            'tests': len(tests),
            'lines': lines,
            'test_details': tests
        })
        
        # Count markers
        for test in tests:
            for marker in test['markers']:
                marker_counts[marker] += 1
    
    # Print summary
    print(f"📁 Total Test Files: {len(test_files)}")
    print(f"🧪 Total Test Functions: {total_tests}")
    print(f"📄 Total Lines of Code: {total_lines:,}")
    print()
    
    # Print by category
    print("📊 Test Categories:")
    print("-" * 40)
    
    for category, files in categories.items():
        category_tests = sum(f['tests'] for f in files)
        category_lines = sum(f['lines'] for f in files)
        print(f"\n{category}:")
        print(f"  Files: {len(files)}")
        print(f"  Tests: {category_tests}")
        print(f"  Lines: {category_lines:,}")
        
        for file_info in files:
            print(f"    • {file_info['file']}: {file_info['tests']} tests ({file_info['lines']:,} lines)")
    
    # Print marker summary
    if marker_counts:
        print(f"\n🏷️  Test Markers:")
        print("-" * 40)
        for marker, count in sorted(marker_counts.items()):
            print(f"  @pytest.mark.{marker}: {count} tests")
    
    # Analyze test distribution
    print(f"\n📈 Test Distribution:")
    print("-" * 40)
    unit_tests = sum(f['tests'] for f in categories.get('Unit Tests', []))
    integration_tests = sum(f['tests'] for f in categories.get('Integration Tests', []))
    e2e_tests = sum(f['tests'] for f in categories.get('End-to-End Tests', []))
    performance_tests = sum(f['tests'] for f in categories.get('Performance Tests', []))
    
    print(f"  Unit Tests: {unit_tests} ({unit_tests/total_tests*100:.1f}%)")
    print(f"  Integration Tests: {integration_tests} ({integration_tests/total_tests*100:.1f}%)")
    print(f"  E2E Tests: {e2e_tests} ({e2e_tests/total_tests*100:.1f}%)")
    print(f"  Performance Tests: {performance_tests} ({performance_tests/total_tests*100:.1f}%)")
    
    # Check for test files
    print(f"\n📋 Test File Details:")
    print("-" * 40)
    
    expected_files = [
        ("test_literature_review_agent.py", "Literature Review Agent"),
        ("test_generation_agent.py", "Generation Agent"),
        ("test_ranking_agent.py", "Ranking Agent"),
        ("test_framework_integration.py", "Framework Integration"),
        ("test_cli_integration.py", "CLI Integration"),
        ("test_e2e_scenarios.py", "End-to-End Scenarios"),
        ("test_performance.py", "Performance Tests"),
    ]
    
    existing_files = [f.name for f in test_files]
    
    for expected_file, description in expected_files:
        status = "✅" if expected_file in existing_files else "❌"
        print(f"  {status} {expected_file}: {description}")
    
    # Configuration files
    print(f"\n⚙️  Configuration Files:")
    print("-" * 40)
    
    config_files = [
        ("pytest.ini", "Pytest configuration"),
        ("conftest.py", "Test fixtures and setup"),
        ("utils.py", "Test utilities"),
        ("README.md", "Test documentation"),
    ]
    
    for config_file, description in config_files:
        config_path = test_dir / config_file
        status = "✅" if config_path.exists() else "❌"
        print(f"  {status} {config_file}: {description}")
    
    # Fixture files
    fixtures_dir = test_dir / "fixtures"
    if fixtures_dir.exists():
        fixture_files = list(fixtures_dir.glob("*.json"))
        print(f"\n📦 Test Fixtures:")
        print("-" * 40)
        for fixture in fixture_files:
            print(f"  • {fixture.name}")
    
    # Quick start guide
    print(f"\n🚀 Quick Start:")
    print("-" * 40)
    print(f"  # Run all unit tests (fast)")
    print(f"  python run_tests.py unit")
    print(f"")
    print(f"  # Run with coverage")
    print(f"  python run_tests.py unit --coverage")
    print(f"")
    print(f"  # Run integration tests")
    print(f"  python run_tests.py integration")
    print(f"")
    print(f"  # Run specific test")
    print(f"  pytest tests/test_generation_agent.py::TestGenerationAgent::test_generate_hypotheses_basic -v")
    print(f"")
    print(f"  # Full test suite (no real APIs)")
    print(f"  python run_tests.py all")


if __name__ == "__main__":
    analyze_test_suite()