#!/usr/bin/env python3
"""
Real MATLAB Engine Test - Non-mocked mathematical validation
Windows-compatible version without Unicode characters
"""

import matlab.engine
import math
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

def run_real_matlab_tests():
    """Run actual MATLAB Engine tests with real computations"""
    
    print("=" * 60)
    print("REAL MATLAB ENGINE TEST - NON-MOCKED")
    print("=" * 60)
    
    # Initialize test artifacts
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "MATLAB Engine API Real Tests",
        "environment": "Windows Python from WSL",
        "tests": [],
        "summary": {}
    }
    
    # Start MATLAB Engine
    print("\n[1] Starting MATLAB Engine...")
    start_time = time.time()
    eng = matlab.engine.start_matlab()
    startup_time = time.time() - start_time
    print(f"[OK] MATLAB Engine started in {startup_time:.2f} seconds")
    
    # Test results tracking
    passed = 0
    failed = 0
    test_categories = []
    
    print("\n[2] Running Mathematical Validation Tests...")
    print("-" * 40)
    
    # TEST 1: Basic Arithmetic
    print("\n== Basic Arithmetic Tests ==")
    arithmetic_tests = [
        ("sqrt(64)", 8.0),
        ("2^8", 256.0),
        ("exp(1)", math.e),
        ("log(exp(5))", 5.0),
        ("10 - 7", 3.0),
        ("6 * 7", 42.0),
    ]
    
    category_results = {"name": "Basic Arithmetic", "tests": [], "passed": 0, "failed": 0}
    
    for expr, expected in arithmetic_tests:
        test_start_time = time.time()
        test_case = {
            "name": expr,
            "expected": expected,
            "status": "FAIL",
            "actual": None,
            "error": None,
            "execution_time": 0
        }
        
        try:
            result = eng.eval(expr)
            test_case["actual"] = float(result)
            test_case["execution_time"] = time.time() - test_start_time
            
            if abs(result - expected) < 1e-9:
                print(f"  [PASS] {expr} = {result:.6f} (expected {expected:.6f})")
                test_case["status"] = "PASS"
                passed += 1
                category_results["passed"] += 1
            else:
                print(f"  [FAIL] {expr} = {result:.6f} (expected {expected:.6f})")
                failed += 1
                category_results["failed"] += 1
        except Exception as e:
            test_case["error"] = str(e)
            test_case["execution_time"] = time.time() - test_start_time
            print(f"  [ERROR] {expr} - Error: {e}")
            failed += 1
            category_results["failed"] += 1
        
        category_results["tests"].append(test_case)
    
    test_categories.append(category_results)
    
    # TEST 2: Trigonometric Functions
    print("\n== Trigonometric Functions ==")
    trig_tests = [
        ("sin(pi/2)", 1.0),
        ("cos(0)", 1.0),
        ("tan(pi/4)", 1.0),
        ("sin(pi)", 0.0),
        ("cos(pi/2)", 0.0),
    ]
    
    trig_category = {"name": "Trigonometric Functions", "tests": [], "passed": 0, "failed": 0}
    
    for expr, expected in trig_tests:
        test_start_time = time.time()
        test_case = {
            "name": expr,
            "expected": expected,
            "status": "FAIL",
            "actual": None,
            "error": None,
            "execution_time": 0
        }
        
        try:
            result = eng.eval(expr)
            test_case["actual"] = float(result)
            test_case["execution_time"] = time.time() - test_start_time
            
            if abs(result - expected) < 1e-9:
                print(f"  [PASS] {expr} = {result:.6f} (expected {expected:.6f})")
                test_case["status"] = "PASS"
                passed += 1
                trig_category["passed"] += 1
            else:
                print(f"  [FAIL] {expr} = {result:.6f} (expected {expected:.6f})")
                failed += 1
                trig_category["failed"] += 1
        except Exception as e:
            test_case["error"] = str(e)
            test_case["execution_time"] = time.time() - test_start_time
            print(f"  [ERROR] {expr} - Error: {e}")
            failed += 1
            trig_category["failed"] += 1
        
        trig_category["tests"].append(test_case)
    
    test_categories.append(trig_category)
    
    # TEST 3: Matrix Operations
    print("\n== Matrix Operations ==")
    try:
        # Create matrices
        A = matlab.double([[1.0, 2.0], [3.0, 4.0]])
        B = matlab.double([[5.0, 6.0], [7.0, 8.0]])
        
        # Matrix multiplication
        result = eng.mtimes(A, B)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        if result == expected:
            print(f"  [PASS] Matrix multiplication: A*B correct")
            passed += 1
        else:
            print(f"  [FAIL] Matrix multiplication failed")
            failed += 1
        
        # Determinant
        det_result = eng.det(A)
        if abs(det_result - (-2.0)) < 1e-9:
            print(f"  [PASS] det(A) = {det_result:.2f} (expected -2.0)")
            passed += 1
        else:
            print(f"  [FAIL] det(A) = {det_result:.2f} (expected -2.0)")
            failed += 1
        
        # Inverse
        inv_A = eng.inv(A)
        identity = eng.mtimes(A, inv_A)
        # Check if result is close to identity matrix
        is_identity = True
        for i in range(2):
            for j in range(2):
                expected_val = 1.0 if i == j else 0.0
                if abs(identity[i][j] - expected_val) > 1e-9:
                    is_identity = False
        
        if is_identity:
            print(f"  [PASS] Matrix inverse: A * inv(A) = I")
            passed += 1
        else:
            print(f"  [FAIL] Matrix inverse verification failed")
            failed += 1
            
    except Exception as e:
        print(f"  [ERROR] Matrix operations error: {e}")
        failed += 3
    
    # TEST 4: Complex Numbers
    print("\n== Complex Number Operations ==")
    try:
        # Euler's formula: e^(i*pi) = -1
        result = eng.eval("exp(1i*pi)")
        if abs(result.real - (-1.0)) < 1e-9 and abs(result.imag) < 1e-9:
            print(f"  [PASS] Euler's formula: e^(i*pi) = -1")
            passed += 1
        else:
            print(f"  [FAIL] Euler's formula failed")
            failed += 1
            
        # Complex magnitude
        z = complex(3, 4)
        mag = eng.abs(z)
        if abs(mag - 5.0) < 1e-9:
            print(f"  [PASS] |3+4i| = {mag:.2f} (expected 5.0)")
            passed += 1
        else:
            print(f"  [FAIL] Complex magnitude failed")
            failed += 1
            
    except Exception as e:
        print(f"  [ERROR] Complex operations error: {e}")
        failed += 2
    
    # TEST 5: Statistical Functions
    print("\n== Statistical Functions ==")
    try:
        data = matlab.double([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Mean
        mean_val = eng.mean(data)
        if abs(mean_val - 5.5) < 1e-9:
            print(f"  [PASS] mean([1..10]) = {mean_val:.2f} (expected 5.5)")
            passed += 1
        else:
            print(f"  [FAIL] Mean calculation failed")
            failed += 1
        
        # Standard deviation
        std_val = eng.std(data)
        expected_std = 3.02765035409749
        if abs(std_val - expected_std) < 1e-6:
            print(f"  [PASS] std([1..10]) = {std_val:.6f}")
            passed += 1
        else:
            print(f"  [FAIL] Standard deviation failed")
            failed += 1
            
        # Median
        median_val = eng.median(data)
        if abs(median_val - 5.5) < 1e-9:
            print(f"  [PASS] median([1..10]) = {median_val:.2f} (expected 5.5)")
            passed += 1
        else:
            print(f"  [FAIL] Median calculation failed")
            failed += 1
            
    except Exception as e:
        print(f"  [ERROR] Statistical functions error: {e}")
        failed += 3
    
    # TEST 6: Polynomial Operations
    print("\n== Polynomial Operations ==")
    try:
        # Polynomial roots: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        coeffs = matlab.double([1, -6, 11, -6])
        roots = eng.roots(coeffs)
        expected_roots = [1.0, 2.0, 3.0]
        roots_list = sorted([float(r) for r in roots])
        
        all_correct = all(abs(roots_list[i] - expected_roots[i]) < 1e-9 for i in range(3))
        
        if all_correct:
            print(f"  [PASS] Polynomial roots: {roots_list}")
            passed += 1
        else:
            print(f"  [FAIL] Polynomial roots incorrect")
            failed += 1
            
    except Exception as e:
        print(f"  [ERROR] Polynomial operations error: {e}")
        failed += 1
    
    # TEST 7: Numerical Integration
    print("\n== Numerical Integration ==")
    try:
        # Integrate sin(x) from 0 to pi (should be 2)
        x = eng.linspace(0, float(math.pi), 1000)
        y = eng.sin(x)
        integral = eng.trapz(y, x)
        
        if abs(integral - 2.0) < 1e-3:
            print(f"  [PASS] Integral of sin(x) from 0 to pi = {integral:.4f} (expected 2.0)")
            passed += 1
        else:
            print(f"  [FAIL] Integration failed")
            failed += 1
            
    except Exception as e:
        print(f"  [ERROR] Integration error: {e}")
        failed += 1
    
    # Clean up
    print("\n[3] Cleaning up...")
    eng.quit()
    print("[OK] MATLAB Engine closed")
    
    # Finalize test results
    total_execution_time = time.time() - start_time
    total = passed + failed
    success_rate = (passed/total)*100 if total > 0 else 0
    
    test_results["categories"] = test_categories
    test_results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "success_rate": success_rate,
        "total_execution_time": total_execution_time,
        "startup_time": startup_time
    }
    
    # Generate test artifacts
    generate_test_artifacts(test_results)
    
    # Final Report
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"[PASS] Passed: {passed}/{total}")
    print(f"[FAIL] Failed: {failed}/{total}")
    print(f"[INFO] Success Rate: {success_rate:.1f}%")
    print(f"[TIME] Total Execution Time: {total_execution_time:.2f} seconds")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! MATLAB Engine API is working correctly.")
    else:
        print(f"\n[WARNING] {failed} tests failed. Please review the results above.")
    
    print("\n[ARTIFACTS] Test results saved to:")
    print("  - test_results.json")
    print("  - test_results.xml")
    print("  - performance_metrics.json")
    
    return passed, failed

def generate_test_artifacts(test_results):
    """Generate test result artifacts in multiple formats"""
    
    # Create artifacts directory
    artifacts_dir = Path("test_artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # 1. JSON Test Results
    json_file = artifacts_dir / "test_results.json"
    with open(json_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # 2. XML Test Results (JUnit format)
    xml_file = artifacts_dir / "test_results.xml"
    generate_junit_xml(test_results, xml_file)
    
    # 3. Performance Metrics
    performance_file = artifacts_dir / "performance_metrics.json"
    performance_data = {
        "timestamp": test_results["timestamp"],
        "startup_time": test_results["summary"]["startup_time"],
        "total_execution_time": test_results["summary"]["total_execution_time"],
        "average_test_time": test_results["summary"]["total_execution_time"] / test_results["summary"]["total_tests"],
        "performance_by_category": {}
    }
    
    for category in test_results["categories"]:
        category_time = sum(test["execution_time"] for test in category["tests"])
        performance_data["performance_by_category"][category["name"]] = {
            "total_time": category_time,
            "average_time": category_time / len(category["tests"]) if category["tests"] else 0,
            "test_count": len(category["tests"])
        }
    
    with open(performance_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    # 4. Test Summary Report
    summary_file = artifacts_dir / "test_summary.md"
    generate_markdown_summary(test_results, summary_file)

def generate_junit_xml(test_results, xml_file):
    """Generate JUnit XML format test results"""
    
    root = ET.Element("testsuites")
    root.set("name", test_results["test_suite"])
    root.set("tests", str(test_results["summary"]["total_tests"]))
    root.set("failures", str(test_results["summary"]["failed"]))
    root.set("time", str(test_results["summary"]["total_execution_time"]))
    root.set("timestamp", test_results["timestamp"])
    
    for category in test_results["categories"]:
        testsuite = ET.SubElement(root, "testsuite")
        testsuite.set("name", category["name"])
        testsuite.set("tests", str(len(category["tests"])))
        testsuite.set("failures", str(category["failed"]))
        testsuite.set("time", str(sum(test["execution_time"] for test in category["tests"])))
        
        for test in category["tests"]:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", test["name"])
            testcase.set("classname", f"{test_results['test_suite']}.{category['name']}")
            testcase.set("time", str(test["execution_time"]))
            
            if test["status"] == "FAIL":
                failure = ET.SubElement(testcase, "failure")
                if test["error"]:
                    failure.set("message", test["error"])
                    failure.text = test["error"]
                else:
                    failure.set("message", f"Expected {test['expected']}, got {test['actual']}")
                    failure.text = f"Expected {test['expected']}, got {test['actual']}"
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

def generate_markdown_summary(test_results, summary_file):
    """Generate markdown test summary report"""
    
    with open(summary_file, 'w') as f:
        f.write(f"# MATLAB Engine API Test Results\n\n")
        f.write(f"**Test Suite:** {test_results['test_suite']}\n")
        f.write(f"**Timestamp:** {test_results['timestamp']}\n")
        f.write(f"**Environment:** {test_results['environment']}\n\n")
        
        f.write("## Summary\n\n")
        summary = test_results["summary"]
        f.write(f"- **Total Tests:** {summary['total_tests']}\n")
        f.write(f"- **Passed:** {summary['passed']} ✅\n")
        f.write(f"- **Failed:** {summary['failed']} ❌\n")
        f.write(f"- **Success Rate:** {summary['success_rate']:.1f}%\n")
        f.write(f"- **Execution Time:** {summary['total_execution_time']:.2f} seconds\n")
        f.write(f"- **Startup Time:** {summary['startup_time']:.2f} seconds\n\n")
        
        f.write("## Test Categories\n\n")
        for category in test_results["categories"]:
            f.write(f"### {category['name']}\n\n")
            f.write(f"- Passed: {category['passed']}/{len(category['tests'])}\n")
            f.write(f"- Failed: {category['failed']}/{len(category['tests'])}\n\n")
            
            f.write("| Test | Status | Expected | Actual | Time (s) | Error |\n")
            f.write("|------|--------|----------|--------|----------|-------|\n")
            
            for test in category["tests"]:
                status_emoji = "✅" if test["status"] == "PASS" else "❌"
                error_msg = test.get("error", "")[:50] + "..." if test.get("error") and len(test.get("error", "")) > 50 else test.get("error", "")
                f.write(f"| {test['name']} | {status_emoji} {test['status']} | {test['expected']} | {test.get('actual', 'N/A')} | {test['execution_time']:.3f} | {error_msg} |\n")
            
            f.write("\n")

if __name__ == "__main__":
    try:
        passed, failed = run_real_matlab_tests()
        exit(0 if failed == 0 else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        print("\nPlease ensure:")
        print("1. MATLAB is installed on this system")
        print("2. MATLAB Engine API for Python is installed")
        print("3. MATLAB license is valid")
        
        # Generate error artifact
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "FATAL_ERROR",
            "test_suite": "MATLAB Engine API Real Tests"
        }
        
        artifacts_dir = Path("test_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        with open(artifacts_dir / "error_report.json", 'w') as f:
            json.dump(error_report, f, indent=2)
        
        exit(1)