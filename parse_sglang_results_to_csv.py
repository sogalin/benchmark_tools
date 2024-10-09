import re
import os
import csv
import argparse
from collections import defaultdict

# Define regex patterns to match the required metrics
prefill_pattern = r"Prefill\. latency:\s*([\d\.]+)\s*s,\s*throughput:\s*([\d\.]+)\s*token/s"
decode_median_pattern = r"Decode\.  median latency:\s*([\d\.]+)\s*s,\s*median throughput:\s*([\d\.]+)\s*token/s"
total_pattern = r"Total\. latency:\s*([\d\.]+)\s*s,\s*throughput:\s*([\d\.]+)\s*token/s"

# Update the filename pattern to capture company, GPU, model, TP, batch size, input, and output
filename_pattern = r"(?P<company>[\w\.-]+)_(?P<gpu>[\w\.-]+)_(?P<model>[\w\.-]+)_tp(?P<tp>\d+)_bs(?P<batch_size>\d+)_in(?P<input>\d+)_out(?P<output>\d+)"

# Read log files and extract relevant information (only keep the second occurrence)
def parse_log_file(file_path, file_name):
    # Use regex to extract information from the file name
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

    # Initialize a list to store the results
    results = []
    with open(file_path, 'r') as f:
        log_content = f.read()
        
        # Extract all Prefill latency
        prefill_matches = re.findall(prefill_pattern, log_content)
        # Extract all Decode median latency
        decode_median_matches = re.findall(decode_median_pattern, log_content)
        # Extract all Total latency
        total_matches = re.findall(total_pattern, log_content)

        # Ensure the number of matches are consistent and take only the second occurrence
        if len(prefill_matches) >= 2 and len(decode_median_matches) >= 2 and len(total_matches) >= 2:
            # Collect only the second occurrence
            results.append({
                'Company': company_name,
                'GPU': gpu_name,
                'Model': model_name,
                'TP': tp,
                'Batch size': batch_size,
                'Input size': input_size,
                'Output size': output_size,
                'Benchmark number': 2,  # Since we're capturing the second occurrence
                'Prefill latency (s)': float(prefill_matches[1][0]),
                'Prefill throughput (token/s)': float(prefill_matches[1][1]),
                'Decode median latency (s)': float(decode_median_matches[1][0]),
                'Decode median throughput (token/s)': float(decode_median_matches[1][1]),
                'Total latency (s)': float(total_matches[1][0]),
                'Total throughput (token/s)': float(total_matches[1][1]),
            })
        else:
            print(f"Insufficient data in file: {file_name}")
    
    return results

# Parse all log files in a folder, only processing .txt and .log files
def parse_folder(folder_path):
    all_results = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Only process .txt and .log files
        if os.path.isfile(file_path) and file_name.endswith(('.txt', '.log')):
            log_data = parse_log_file(file_path, file_name)
            if log_data:
                all_results.extend(log_data)
    
    return all_results

# Group the data by input-output size and sort by batch size
def group_and_sort_data(data):
    grouped_data = defaultdict(list)
    
    # Group data by (input_size, output_size)
    for entry in data:
        key = (entry['Input size'], entry['Output size'])
        grouped_data[key].append(entry)
    
    # Sort each group by batch size
    for key in grouped_data:
        grouped_data[key].sort(key=lambda x: x['Batch size'])
    
    # Flatten the grouped data into a single list for CSV export
    sorted_data = []
    for key in grouped_data:
        sorted_data.extend(grouped_data[key])
    
    return sorted_data

# Save the extracted data to a CSV file
def save_to_csv(data, output_file):
    # Define the column headers for the CSV file
    fieldnames = [
        'Company', 'GPU', 'Model', 'TP', 'Batch size', 'Input size', 'Output size', 'Benchmark number',
        'Prefill latency (s)', 'Prefill throughput (token/s)', 
        'Decode median latency (s)', 'Decode median throughput (token/s)', 
        'Total latency (s)', 'Total throughput (token/s)'
    ]
    
    # Open the CSV file and write data to it
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row
        writer.writerows(data)  # Write multiple rows of data

# Main function to handle command-line arguments
def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Parse log files and extract benchmark results to CSV.")
    
    # Add arguments for folder path and output file
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the log files.')
    parser.add_argument('output_file', type=str, help='Path to save the output CSV file.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Parse the log folder
    all_results = parse_folder(args.folder_path)
    
    # Group the data by input-output sizes and sort by batch size
    sorted_data = group_and_sort_data(all_results)
    
    # Save the sorted results to the CSV file
    save_to_csv(sorted_data, args.output_file)

    print(f"Results saved to {args.output_file}")

# Run the main function when this script is executed
if __name__ == '__main__':
    main()

