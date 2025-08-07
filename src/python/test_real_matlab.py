#!/usr/bin/env python3
"""
Real MATLAB Engine Test - Non-mocked mathematical validation
Tests actual MATLAB computations for accuracy
"""

import matlab.engine
import math
import time

def run_real_matlab_tests():
    """Run actual MATLAB Engine tests with real computations"""
    
    print("=" * 60)
    print("REAL MATLAB ENGINE TEST - NON-MOCKED")
    print("=" * 60)
    
    # Start MATLAB Engine
    print("\n[1] Starting MATLAB Engine...")
    start_time = time.time()
    eng = matlab.engine.start_matlab()
    startup_time = time.time() - start_time
    print(f"✅ MATLAB Engine started in {startup_time:.2f} seconds")
    
    # Test results tracking
    passed = 0
    failed = 0
    
    print("\n[2] Running Mathematical Validation Tests...")
    print("-" * 40)
    
    # TEST 1: Basic Arithmetic
    print("\n📐 Basic Arithmetic Tests:")
    tests = [
        ("sqrt(64)", 8.0),
        ("2^8", 256.0),
        ("exp(1)", math.e),
        ("log(exp(5))", 5.0),
        ("10 - 7", 3.0),
        ("6 * 7", 42.0),
    ]
    
    for expr, expected in tests:
        try:
            result = eng.eval(expr)
            if abs(result - expected) < 1e-9:
                print(f"  ✅ {expr} = {result:.6f} (expected {expected:.6f})")
                passed += 1
            else:
                print(f"  ❌ {expr} = {result:.6f} (expected {expected:.6f})")
                failed += 1
        except Exception as e:
            print(f"  ❌ {expr} - Error: {e}")
            failed += 1
    
    # TEST 2: Trigonometric Functions
    print("\n📐 Trigonometric Functions:")
    trig_tests = [
        ("sin(pi/2)", 1.0),
        ("cos(0)", 1.0),
        ("tan(pi/4)", 1.0),
        ("sin(pi)", 0.0),
        ("cos(pi/2)", 0.0),
    ]
    
    for expr, expected in trig_tests:
        try:
            result = eng.eval(expr)
            if abs(result - expected) < 1e-9:
                print(f"  ✅ {expr} = {result:.6f} (expected {expected:.6f})")
                passed += 1
            else:
                print(f"  ❌ {expr} = {result:.6f} (expected {expected:.6f})")
                failed += 1
        except Exception as e:
            print(f"  ❌ {expr} - Error: {e}")
            failed += 1
    
    # TEST 3: Matrix Operations
    print("\n📐 Matrix Operations:")
    try:
        # Create matrices
        A = matlab.double([[1.0, 2.0], [3.0, 4.0]])
        B = matlab.double([[5.0, 6.0], [7.0, 8.0]])
        
        # Matrix multiplication
        result = eng.mtimes(A, B)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        if result == expected:
            print(f"  ✅ Matrix multiplication: A*B correct")
            passed += 1
        else:
            print(f"  ❌ Matrix multiplication failed")
            failed += 1
        
        # Determinant
        det_result = eng.det(A)
        if abs(det_result - (-2.0)) < 1e-9:
            print(f"  ✅ det(A) = {det_result:.2f} (expected -2.0)")
            passed += 1
        else:
            print(f"  ❌ det(A) = {det_result:.2f} (expected -2.0)")
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
            print(f"  ✅ Matrix inverse: A * inv(A) = I")
            passed += 1
        else:
            print(f"  ❌ Matrix inverse verification failed")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ Matrix operations error: {e}")
        failed += 3
    
    # TEST 4: Complex Numbers
    print("\n📐 Complex Number Operations:")
    try:
        # Euler's formula: e^(i*pi) = -1
        result = eng.eval("exp(1i*pi)")
        if abs(result.real - (-1.0)) < 1e-9 and abs(result.imag) < 1e-9:
            print(f"  ✅ Euler's formula: e^(iπ) = -1")
            passed += 1
        else:
            print(f"  ❌ Euler's formula failed")
            failed += 1
            
        # Complex magnitude
        z = complex(3, 4)
        mag = eng.abs(z)
        if abs(mag - 5.0) < 1e-9:
            print(f"  ✅ |3+4i| = {mag:.2f} (expected 5.0)")
            passed += 1
        else:
            print(f"  ❌ Complex magnitude failed")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ Complex operations error: {e}")
        failed += 2
    
    # TEST 5: Statistical Functions
    print("\n📐 Statistical Functions:")
    try:
        data = matlab.double([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Mean
        mean_val = eng.mean(data)
        if abs(mean_val - 5.5) < 1e-9:
            print(f"  ✅ mean([1..10]) = {mean_val:.2f} (expected 5.5)")
            passed += 1
        else:
            print(f"  ❌ Mean calculation failed")
            failed += 1
        
        # Standard deviation
        std_val = eng.std(data)
        expected_std = 3.02765035409749
        if abs(std_val - expected_std) < 1e-6:
            print(f"  ✅ std([1..10]) = {std_val:.6f}")
            passed += 1
        else:
            print(f"  ❌ Standard deviation failed")
            failed += 1
            
        # Median
        median_val = eng.median(data)
        if abs(median_val - 5.5) < 1e-9:
            print(f"  ✅ median([1..10]) = {median_val:.2f} (expected 5.5)")
            passed += 1
        else:
            print(f"  ❌ Median calculation failed")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ Statistical functions error: {e}")
        failed += 3
    
    # TEST 6: Polynomial Operations
    print("\n📐 Polynomial Operations:")
    try:
        # Polynomial roots: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        coeffs = matlab.double([1, -6, 11, -6])
        roots = eng.roots(coeffs)
        expected_roots = [1.0, 2.0, 3.0]
        roots_list = sorted([float(r) for r in roots])
        
        all_correct = all(abs(roots_list[i] - expected_roots[i]) < 1e-9 for i in range(3))
        
        if all_correct:
            print(f"  ✅ Polynomial roots: {roots_list}")
            passed += 1
        else:
            print(f"  ❌ Polynomial roots incorrect")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ Polynomial operations error: {e}")
        failed += 1
    
    # TEST 7: Numerical Integration
    print("\n📐 Numerical Integration:")
    try:
        # Integrate sin(x) from 0 to pi (should be 2)
        x = eng.linspace(0, float(math.pi), 1000)
        y = eng.sin(x)
        integral = eng.trapz(y, x)
        
        if abs(integral - 2.0) < 1e-3:
            print(f"  ✅ ∫sin(x)dx from 0 to π = {integral:.4f} (expected 2.0)")
            passed += 1
        else:
            print(f"  ❌ Integration failed")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ Integration error: {e}")
        failed += 1
    
    # Clean up
    print("\n[3] Cleaning up...")
    eng.quit()
    print("✅ MATLAB Engine closed")
    
    # Final Report
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    total = passed + failed
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    print(f"⏱️  Total Execution Time: {time.time() - start_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MATLAB Engine API is working correctly.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the results above.")
    
    return passed, failed

if __name__ == "__main__":
    try:
        passed, failed = run_real_matlab_tests()
        exit(0 if failed == 0 else 1)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print("\nPlease ensure:")
        print("1. MATLAB is installed on this system")
        print("2. MATLAB Engine API for Python is installed")
        print("3. MATLAB license is valid")
        exit(1)