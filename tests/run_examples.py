#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

examples = [
    'simple_square.py',
    'complex_polygon.py',
    'random_polygons.py'
]

def run_example(example):
    try:
        result = subprocess.run(
            ['python', f'examples/{example}'],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {example} ran successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {example} failed")
        print("Error output:")
        print(e.stderr)
        return False

def main():
    print("Running example tests...\n")
    success_count = 0
    
    for example in examples:
        if run_example(example):
            success_count += 1
    
    print(f"\nTest results: {success_count}/{len(examples)} examples passed")
    if success_count < len(examples):
        sys.exit(1)

if __name__ == "__main__":
    main() 