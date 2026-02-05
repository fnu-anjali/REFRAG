#!/usr/bin/env python3
"""
Combine all individual result JSON files into a single JSON file.
Excludes the averaged_metrics.json file.
"""

import json
import argparse
from pathlib import Path


def combine_json_files(directory: str, output_name: str = "combined_results.json", exclude_patterns: list = None):
    """Combine all JSON files in a directory into one."""
    path = Path(directory)

    if not path.exists():
        print(f"Error: Directory {directory} does not exist")
        return False

    if exclude_patterns is None:
        exclude_patterns = ["averaged_metrics.json", "combined_results.json", "summary.json"]

    # Find all JSON files
    all_files = sorted(path.glob("*.json"))

    # Filter out excluded files
    json_files = [f for f in all_files if f.name not in exclude_patterns]

    if not json_files:
        print(f"No JSON files found in {directory} (excluding {exclude_patterns})")
        return False

    print(f"Found {len(json_files)} files to combine:")

    combined_data = []
    errors = []

    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add filename to the data for reference
                data['_source_file'] = file.name
                combined_data.append(data)
                print(f"  ✓ {file.name}")
        except Exception as e:
            errors.append(f"{file.name}: {e}")
            print(f"  ✗ {file.name} (error: {e})")

    if not combined_data:
        print("No valid JSON data found to combine")
        return False

    # Save combined results
    output_path = path / output_name
    output_data = {
        "num_results": len(combined_data),
        "source_directory": str(directory),
        "results": combined_data
    }

    if errors:
        output_data["errors"] = errors

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Successfully combined {len(combined_data)} files")
    print(f"✓ Saved to: {output_path}")

    if errors:
        print(f"\n⚠ Encountered {len(errors)} errors (see 'errors' field in output)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Combine multiple JSON result files into one")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing JSON files")
    parser.add_argument("--output", type=str, default="combined_results.json", help="Output filename (default: combined_results.json)")
    parser.add_argument("--exclude", type=str, nargs="+", default=None, help="Additional file patterns to exclude")

    args = parser.parse_args()

    # Default exclusions
    exclude_patterns = ["averaged_metrics.json", "combined_results.json", "summary.json"]

    # Add user-specified exclusions
    if args.exclude:
        exclude_patterns.extend(args.exclude)

    combine_json_files(args.dir, args.output, exclude_patterns)


if __name__ == "__main__":
    main()
