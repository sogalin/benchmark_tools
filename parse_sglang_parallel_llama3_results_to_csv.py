import os
import csv
import numpy as np
from collections import defaultdict
import re
import sys

# Calculate the Geometric Mean
def geometric_mean(numbers):
    numbers = [x for x in numbers if x > 0]
    if not numbers:
        return None
    return np.exp(np.mean(np.log(numbers)))

# Parse the filename to extract parameters
def parse_filename(filename):
    match = re.search(r'_bs(\d+)_in(\d+)_out(\d+)', filename)
    if match:
        batch_size, input_size, output_size = map(int, match.groups())
        return batch_size, input_size, output_size
    return None, None, None

# Read all txt files and compute Geomean
def process_logs(log_folder):
    metrics = defaultdict(lambda: defaultdict(list))

    for file in os.listdir(log_folder):
        if file.endswith('.txt'):
            batch_size, input_size, output_size = parse_filename(file)
            if batch_size is None:
                continue
            
            filepath = os.path.join(log_folder, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Prefill latency (s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Prefill latency'].append(float(line.split()[-1]))
                    elif "Prefill throughput (token/s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Prefill throughput'].append(float(line.split()[-1]))
                    elif "Decode median latency (s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Decode median latency'].append(float(line.split()[-1]))
                    elif "Decode median throughput (token/s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Decode median throughput'].append(float(line.split()[-1]))
                    elif "Total latency (s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Total latency'].append(float(line.split()[-1]))
                    elif "Total throughput (token/s)" in line:
                        metrics[(batch_size, input_size, output_size)]['Total throughput'].append(float(line.split()[-1]))

    results = []
    for (batch_size, input_size, output_size), values in metrics.items():
        result = {
            'Batch size': batch_size,
            'Input size': input_size,
            'Output size': output_size,
            'Prefill latency (s) Geomean': geometric_mean(values['Prefill latency']),
            'Prefill throughput (token/s) Geomean': geometric_mean(values['Prefill throughput']),
            'Decode median latency (s) Geomean': geometric_mean(values['Decode median latency']),
            'Decode median throughput (token/s) Geomean': geometric_mean(values['Decode median throughput']),
            'Total latency (s) Geomean': geometric_mean(values['Total latency']),
            'Total throughput (token/s) Geomean': geometric_mean(values['Total throughput']),
        }
        results.append(result)

    return results

# Save to CSV, grouped by (Input size, Output size), then sorted by Batch size
def save_to_csv(data, output_file):
    fieldnames = [
        'Batch size', 'Input size', 'Output size',
        'Prefill latency (s) Geomean', 'Prefill throughput (token/s) Geomean',
        'Decode median latency (s) Geomean', 'Decode median throughput (token/s) Geomean',
        'Total latency (s) Geomean', 'Total throughput (token/s) Geomean'
    ]

    # Group by (Input size, Output size)
    grouped_data = defaultdict(list)
    for entry in data:
        key = (entry['Input size'], entry['Output size'])
        grouped_data[key].append(entry)

    # Sort groups by (Input size, Output size)
    sorted_group_keys = sorted(grouped_data.keys(), key=lambda x: (x[0], x[1]))

    sorted_data = []
    for key in sorted_group_keys:
        # Sort each group by Batch size in ascending order
        sorted_data.extend(sorted(grouped_data[key], key=lambda x: x['Batch size']))

    # Write to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_data:
            writer.writerow({key: (value if value is not None else '') for key, value in row.items()})

    print(f"Final sorted results saved to {output_file}")

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <log_folder> <output_file>")
        sys.exit(1)

    log_folder = sys.argv[1]
    output_file = sys.argv[2]

    data = process_logs(log_folder)
    save_to_csv(data, output_file)

