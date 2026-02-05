#!/usr/bin/env python3
"""
Calculate average metrics from multiple result JSON files.
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_and_average_metrics(directory: str, prefix: str = ""):
    """Load all JSON files and calculate average metrics."""
    path = Path(directory)

    if not path.exists():
        print(f"Error: Directory {directory} does not exist")
        return None

    # Find all JSON files matching the pattern
    if prefix:
        files = sorted(path.glob(f"{prefix}*.json"))
    else:
        files = sorted(path.glob("*.json"))

    if not files:
        print(f"No JSON files found in {directory} with prefix '{prefix}'")
        return None

    ttft_values = []
    ttit_values = []
    throughput_values = []

    print(f"Found {len(files)} files to process:")

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                ttft = data.get('TTFT_sec')
                ttit = data.get('TTIT_avg_sec')
                throughput = data.get('throughput_tok_per_sec')

                if ttft is not None and ttit is not None and throughput is not None:
                    ttft_values.append(ttft)
                    ttit_values.append(ttit)
                    throughput_values.append(throughput)
                    print(f"  ✓ {file.name}")
                else:
                    print(f"  ✗ {file.name} (missing metrics)")
        except Exception as e:
            print(f"  ✗ {file.name} (error: {e})")

    if not ttft_values:
        print("No valid metrics found in any files")
        return None

    # Calculate statistics
    results = {
        "num_samples": len(ttft_values),
        "TTFT_sec": {
            "mean": float(np.mean(ttft_values)),
            "median": float(np.median(ttft_values)),
            "std": float(np.std(ttft_values)),
            "min": float(np.min(ttft_values)),
            "max": float(np.max(ttft_values)),
        },
        "TTIT_avg_sec": {
            "mean": float(np.mean(ttit_values)),
            "median": float(np.median(ttit_values)),
            "std": float(np.std(ttit_values)),
            "min": float(np.min(ttit_values)),
            "max": float(np.max(ttit_values)),
        },
        "throughput_tok_per_sec": {
            "mean": float(np.mean(throughput_values)),
            "median": float(np.median(throughput_values)),
            "std": float(np.std(throughput_values)),
            "min": float(np.min(throughput_values)),
            "max": float(np.max(throughput_values)),
        }
    }

    return results


def print_results(results: dict, name: str = "Results"):
    """Pretty print the results."""
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"{name} (n={results['num_samples']})")
    print(f"{'='*60}")

    print(f"\nTime To First Token (TTFT_sec):")
    print(f"  Mean:   {results['TTFT_sec']['mean']:.6f}")
    print(f"  Median: {results['TTFT_sec']['median']:.6f}")
    print(f"  Std:    {results['TTFT_sec']['std']:.6f}")
    print(f"  Min:    {results['TTFT_sec']['min']:.6f}")
    print(f"  Max:    {results['TTFT_sec']['max']:.6f}")

    print(f"\nAverage Time To Inter-Token (TTIT_avg_sec):")
    print(f"  Mean:   {results['TTIT_avg_sec']['mean']:.6f}")
    print(f"  Median: {results['TTIT_avg_sec']['median']:.6f}")
    print(f"  Std:    {results['TTIT_avg_sec']['std']:.6f}")
    print(f"  Min:    {results['TTIT_avg_sec']['min']:.6f}")
    print(f"  Max:    {results['TTIT_avg_sec']['max']:.6f}")

    print(f"\nThroughput (tok/sec):")
    print(f"  Mean:   {results['throughput_tok_per_sec']['mean']:.2f}")
    print(f"  Median: {results['throughput_tok_per_sec']['median']:.2f}")
    print(f"  Std:    {results['throughput_tok_per_sec']['std']:.2f}")
    print(f"  Min:    {results['throughput_tok_per_sec']['min']:.2f}")
    print(f"  Max:    {results['throughput_tok_per_sec']['max']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Average metrics from result JSON files")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing result JSON files")
    parser.add_argument("--prefix", type=str, default="", help="File prefix to filter (e.g., 'insurance_regulatory')")
    parser.add_argument("--output", type=str, default="", help="Output file name (default: averaged_metrics.json in same dir)")

    args = parser.parse_args()

    # Calculate averages
    results = load_and_average_metrics(args.dir, args.prefix)

    if results is None:
        return

    # Print results
    print_results(results, f"Average Metrics from {args.dir}")

    # Save to file
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.dir) / "averaged_metrics.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved averaged metrics to: {output_path}")


if __name__ == "__main__":
    main()
