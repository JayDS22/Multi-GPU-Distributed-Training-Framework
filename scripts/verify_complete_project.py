#!/usr/bin/env python3
"""
Complete project verification script
Checks all components and ensures the project is fully functional
"""

import os
import sys
from pathlib import Path
import importlib.util
import subprocess

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def check_file_exists(filepath):
    """Check if file exists"""
    if Path(filepath).exists():
        print_success(f"{filepath}")
        return True
    else:
        print_error(f"{filepath} - MISSING")
        return False

def check_python_import(module_path):
    """Check if Python module can be imported"""
    try:
        module = importlib.import_module(module_path)
        print_success(f"Import: {module_path}")
        return True
    except Exception as e:
        print_error(f"Import: {module_path} - {str(e)[:50]}")
        return False

def main():
    print_header("Distributed Training Framework - Complete Verification")
    
    all_checks = []
    
    # Check 1: Core Source Files
    print_header("1. Checking Core Source Files")
    core_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/distributed_training.py",
        "src/core/enhanced_trainer.py",
        "src/core/communication_optimizer.py",
        "src/monitoring/__init__.py",
        "src/monitoring/monitoring_dashboard.py",
        "src/monitoring/health_monitoring.py",
        "src/monitoring/logging_config.py",
        "src/config/__init__.py",
        "src/config/config_manager.py",
        "src/utils/__init__.py",
        "src/utils/helpers.py",
        "src/utils/dataset.py",
    ]
    core_check = all(check_file_exists(f) for f in core_files)
    all_checks.append(("Core Source Files", core_check))
    
    # Check 2: Scripts
    print_header("2. Checking Scripts")
    scripts = [
        "scripts/__init__.py",
        "scripts/production_train.py",
        "scripts/launch_training.sh",
        "scripts/setup_environment.sh",
        "scripts/create_project_structure.sh",
        "scripts/generate_dummy_data.py",
        "scripts/quick_test.sh",
    ]
    scripts_check = all(check_file_exists(f) for f in scripts)
    all_checks.append(("Scripts", scripts_check))
    
    # Check 3: Tests
    print_header("3. Checking Test Files")
    tests = [
        "tests/__init__.py",
        "tests/test_distributed.py",
        "tests/test_integration.py",
        "tests/test_performance.py",
    ]
    tests_check = all(check_file_exists(f) for f in tests)
    all_checks.append(("Test Files", tests_check))
    
    # Check 4: Configs
    print_header("4. Checking Configuration Files")
    configs = [
        "configs/dev.yaml",
        "configs/staging.yaml",
        "configs/production.yaml",
        "configs/README.md",
    ]
    configs_check = all(check_file_exists(f) for f in configs)
    all_checks.append(("Config Files", configs_check))
    
    # Check 5: Deployment
    print_header("5. Checking Deployment Files")
    deployment = [
        "deployment/docker/Dockerfile",
        "deployment/docker/Dockerfile.dev",
        "deployment/docker/docker-compose.yml",
        "deployment/kubernetes/k8s-deployment.yaml",
        "deployment/kubernetes/k8s-service.yaml",
        "deployment/kubernetes/k8s-configmap.yaml",
        "deployment/kubernetes/helm/Chart.yaml",
        "deployment/kubernetes/helm/values.yaml",
    ]
    deployment_check = all(check_file_exists(f) for f in deployment)
    all_checks.append(("Deployment Files", deployment_check))
    
    # Check 6: CI/CD
    print_header("6. Checking CI/CD Files")
    cicd = [
        ".github/workflows/ci-cd.yml",
        ".github/workflows/tests.yml",
        ".github/workflows/security.yml",
    ]
    cicd_check = all(check_file_exists(f) for f in cicd)
    all_checks.append(("CI/CD Files", cicd_check))
    
    # Check 7: Documentation
    print_header("7. Checking Documentation")
    docs = [
        "README.md",
        "DOCS/SETUP_GUIDE.md",
        "DOCS/QUICK_REFERENCE.md",
        "DOCS/IMPLEMENTATION_CHECKLIST.md",
        "DOCS/PROJECT_DESCRIPTION.md",
        "DOCS/GETTING_STARTED.md",
    ]
    docs_check = all(check_file_exists(f) for f in docs)
    all_checks.append(("Documentation", docs_check))
    
    # Check 8: Data Structure
    print_header("8. Checking Data Structure")
    data_files = [
        "data/README.md",
        "data/train/metadata.json",
        "data/val/metadata.json",
    ]
    data_check = all(check_file_exists(f) for f in data_files)
    all_checks.append(("Data Structure", data_check))
    
    # Check 9: Benchmark Files
    print_header("9. Checking Benchmark Files")
    benchmark_files = [
        "benchmarks/__init__.py",
        "benchmarks/run_benchmark.py",
        "benchmarks/results/scalability_results.json",
        "benchmarks/results/benchmark_report.md",
    ]
    benchmark_check = all(check_file_exists(f) for f in benchmark_files)
    all_checks.append(("Benchmark Files", benchmark_check))
    
    # Check 10: Python Imports
    print_header("10. Checking Python Imports")
    imports_to_check = [
        "src",
        "src.core.distributed_training",
        "src.core.enhanced_trainer",
        "src.core.communication_optimizer",
        "src.monitoring.monitoring_dashboard",
        "src.monitoring.health_monitoring",
        "src.monitoring.logging_config",
        "src.config.config_manager",
        "src.utils.helpers",
        "src.utils.dataset",
    ]
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd()))
    
    import_results = []
    for module in imports_to_check:
        result = check_python_import(module)
        import_results.append(result)
    
    imports_check = all(import_results)
    all_checks.append(("Python Imports", imports_check))
    
    # Check 11: Required Dependencies
    print_header("11. Checking Required Dependencies")
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        deps_check = True
    except ImportError:
        print_error("PyTorch not installed")
        deps_check = False
    
    all_checks.append(("Dependencies", deps_check))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total_checks = len(all_checks)
    passed_checks = sum(1 for _, result in all_checks if result)
    
    for check_name, result in all_checks:
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:6}{Colors.NC} - {check_name}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    percentage = (passed_checks / total_checks) * 100
    
    if percentage == 100:
        print(f"{Colors.GREEN}✓ ALL CHECKS PASSED: {passed_checks}/{total_checks} ({percentage:.0f}%){Colors.NC}")
        print(f"{Colors.GREEN}✓ PROJECT IS FULLY FUNCTIONAL!{Colors.NC}")
        return_code = 0
    elif percentage >= 80:
        print(f"{Colors.YELLOW}⚠ MOSTLY FUNCTIONAL: {passed_checks}/{total_checks} ({percentage:.0f}%){Colors.NC}")
        print(f"{Colors.YELLOW}Some components may need attention{Colors.NC}")
        return_code = 1
    else:
        print(f"{Colors.RED}✗ NEEDS WORK: {passed_checks}/{total_checks} ({percentage:.0f}%){Colors.NC}")
        print(f"{Colors.RED}Multiple components are missing{Colors.NC}")
        return_code = 2
    
    # Next Steps
    if percentage < 100:
        print(f"\n{Colors.YELLOW}Next Steps:{Colors.NC}")
        print("1. Run: python scripts/create_project_structure.sh")
        print("2. Run: pip install -e .")
        print("3. Run: python scripts/generate_dummy_data.py")
    else:
        print(f"\n{Colors.GREEN}Ready to use!{Colors.NC}")
        print("Quick test: ./scripts/quick_test.sh")
        print("Start training: python scripts/production_train.py")
    
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
