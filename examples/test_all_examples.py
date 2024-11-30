"""Script to test all example programs in the examples directory.

This script:
1. Finds all Python files in the examples directory
2. Attempts to run each example
3. Reports success/failure for each example
4. Provides a summary of results
"""

import os
import sys
import subprocess
from pathlib import Path

def run_example(example_path):
    """Run a single example and return the result."""
    print(f"\nTesting: {example_path.name}")
    print("-" * (9 + len(example_path.name)))
    
    try:
        # Run the example and capture output
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Check if the program ran successfully
        if result.returncode == 0:
            print("✓ Success")
            return True, None
        else:
            print("✗ Failed with error code:", result.returncode)
            print("\nError output:")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("✗ Failed: Program timed out after 30 seconds")
        return False, "Timeout"
    except Exception as e:
        print("✗ Failed with exception:", str(e))
        return False, str(e)

def main():
    # Get the examples directory
    examples_dir = Path(__file__).parent
    
    # Find all Python files except this test script
    example_files = [
        f for f in examples_dir.glob("*.py")
        if f.name != "test_all_examples.py"
    ]
    
    if not example_files:
        print("No example files found!")
        return
    
    print(f"Found {len(example_files)} example files to test")
    
    # Track results
    results = {}
    
    # Run each example
    for example_path in sorted(example_files):
        success, error = run_example(example_path)
        results[example_path.name] = (success, error)
    
    # Print summary
    print("\nTest Summary")
    print("=" * 50)
    
    successful = [name for name, (success, _) in results.items() if success]
    failed = [name for name, (success, _) in results.items() if not success]
    
    print(f"\nSuccessful ({len(successful)}):")
    for name in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
            error = results[name][1]
            if error:
                first_line = error.split('\n')[0] if error else ''
                print(f"    Error: {first_line}")
    
    # Print final summary
    print("\nFinal Results")
    print("-" * 20)
    print(f"Total examples: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Return non-zero exit code if any tests failed
    sys.exit(len(failed))

if __name__ == "__main__":
    main()
