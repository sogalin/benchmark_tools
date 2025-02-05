import re
import os
import csv
import argparse
from collections import defaultdict

# Define regex patterns to match the required metrics
prefill_pattern = r"Prefill\. latency:\s*([\d\.]+)\s*s,\s*throughput:\s*([\d\.]+)\s*token/s"
decode_median_pattern = r"Decode\.  median latency:\s*([\d\.]+)\s*s,\s*median throughput:\s*([\d\.]+)\s*token/s"
total_pattern = r"Total\. latency:\s*([\d\.]+)\s*s,\s*throughput:\s*([\d\.]+)\s*token/s"

# Corrected filename pattern
filename_pattern = (
    r"(?P<company>[\w\.-]+)_(?P<gpu>[\w\.-]+)_(?P<model>[\w\.-]+)"
    r"_tp(?P<tp>\d+)_result_bs(?P<batch_size>\d+)_in(?P<input>\d+)_out(?P<output>\d+)"
)

# Read log files and extract relevant information (only keep the second occurrence)
def parse_log_file(file_path, file_name):
    file_info = re.match(filename_pattern, file_name)
    if not file_info:
        print(f"File name format incorrect: {file_name}")
        return None

    company_name = file_info.group('company')
    gpu_name = file_info.group('gpu')
    model_name = file_info.group('model')
    tp = int(file_info.group('tp'))
    batch_size = int(file_info.group('batch_size'))
    input_size = int(file_info.group('input'))
    output_size = int(file_info.group('output'))

    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

        # Extract all occurrences of the metrics
        prefill_matches = re.findall(prefill_pattern, log_content)
        decode_median_matches = re.findall(decode_median_pattern, log_content)
        total_matches = re.findall(total_pattern, log_content)

        # Extract only the second occurrence if available
        def get_second_occurrence(matches):
            return map(float, matches[1]) if len(matches) >= 2 else (None, None)

        prefill_latency, prefill_throughput = get_second_occurrence(prefill_matches)
        decode_latency, decode_throughput = get_second_occurrence(decode_median_matches)
        total_latency, total_throughput = get_second_occurrence(total_matches)

        if prefill_latency is not None and decode_latency is not None and total_latency is not None:
            results.append({
                'Company': company_name,
                'GPU': gpu_name,
                'Model': model_name,
                'TP': tp,
                'Batch size': batch_size,
                'Input size': input_size,
                'Output size': output_size,
                'Benchmark number': 2,  # Since we're capturing the second occurrence
                'Prefill latency (s)': prefill_latency,
                'Prefill throughput (token/s)': prefill_throughput,
                'Decode median latency (s)': decode_latency,
                'Decode median throughput (token/s)': decode_throughput,
                'Total latency (s)': total_latency,
                'Total throughput (token/s)': total_throughput,
            })
        else:
            print(f"Insufficient data in file: {file_name}")

    return results

# Parse all log files in a folder, only processing .txt and .log files
def parse_folder(folder_path):
    all_results = []
    for file_name in sorted(os.listdir(folder_path)):  # Ensure ordered processing
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(('.txt', '.log')):
            log_data = parse_log_file(file_path, file_name)
            if log_data:
                all_results.extend(log_data)

    return all_results

# Group the data by input-output size and sort by batch size
def group_and_sort_data(data):
    grouped_data = defaultdict(list)

    for entry in data:
        key = (entry['Input size'], entry['Output size'])
        grouped_data[key].append(entry)

    for key in grouped_data:
        grouped_data[key].sort(key=lambda x: x['Batch size'])

    sorted_data = []
    for key in grouped_data:
        sorted_data.extend(grouped_data[key])

    return sorted_data

# Save the extracted data to a CSV file
def save_to_csv(data, output_file):
    fieldnames = [
        'Company', 'GPU', 'Model', 'TP', 'Batch size', 'Input size', 'Output size', 'Benchmark number',
        'Prefill latency (s)', 'Prefill throughput (token/s)',
        'Decode median latency (s)', 'Decode median throughput (token/s)',
        'Total latency (s)', 'Total throughput (token/s)'
    ]

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({key: (value if value is not None else '') for key, value in row.items()})

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Parse log files and extract benchmark results to CSV.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the log files.')
    parser.add_argument('output_file', type=str, help='Path to save the output CSV file.')

    args = parser.parse_args()

    all_results = parse_folder(args.folder_path)
    sorted_data = group_and_sort_data(all_results)
    save_to_csv(sorted_data, args.output_file)

    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()

