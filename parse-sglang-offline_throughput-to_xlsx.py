import os
import re
import pandas as pd
import argparse

def extract_log_data(log_file_path):
    """
    Extract benchmark result data from a log file.
    """
    with open(log_file_path, 'r') as file:
        content = file.read()
    
    # Define regex patterns for extracting the data
    patterns = {
        "Backend": r"Backend:\s+(.+)",
        "Successful requests": r"Successful requests:\s+(\d+)",
        "Benchmark duration (s)": r"Benchmark duration \(s\):\s+([\d.]+)",
        "Total input tokens": r"Total input tokens:\s+(\d+)",
        "Total generated tokens": r"Total generated tokens:\s+(\d+)",
        "Request throughput (req/s)": r"Request throughput \(req/s\):\s+([\d.]+)",
        "Input token throughput (tok/s)": r"Input token throughput \(tok/s\):\s+([\d.]+)",
        "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "Total token throughput (tok/s)": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    }
    
    # Extract data using regex patterns
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        data[key] = match.group(1) if match else None
    
    return data

def parse_filename(filename):
    """
    Parse the log file name to extract configuration parameters.
    Example file name:
    benchmark_Meta-Llama-3.1-405B-Instruct-FP8-KV_tp8_input128_output128_prompts1000_context4096_chunk4096_range0_20241129_033721.log
    """
    pattern = (
        r"benchmark_"
        r"(?P<Model_Name>[\w\-.]+)_"
        r"tp(?P<TP>\d+)_"
        r"input(?P<Input_Size>\d+)_"
        r"output(?P<Output_Size>\d+)_"
        r"prompts(?P<Prompts>\d+)_"
        r"context(?P<Context_Length>\d+)_"
        r"chunk(?P<Chunked_Prefill_Size>\d+)_"
        r"range(?P<Random_Range_Ratio>[0-1])_"
        r"\d{8}_\d{6}\.log"
    )
    match = re.match(pattern, filename)
    
    if match:
        return {key: int(value) if value.isdigit() else value for key, value in match.groupdict().items()}
    else:
        return {
            "Model_Name": None,
            "TP": None,
            "Input_Size": None,
            "Output_Size": None,
            "Prompts": None,
            "Context_Length": None,
            "Chunked_Prefill_Size": None,
            "Random_Range_Ratio": None,
        }

def process_logs_to_excel(logs_directory, output_file):
    """
    Process all log files in a directory and save the extracted data to an Excel file.
    """
    log_files = [f for f in os.listdir(logs_directory) if f.endswith(".log")]
    all_data = []

    for log_file in log_files:
        log_file_path = os.path.join(logs_directory, log_file)
        log_data = extract_log_data(log_file_path)
        file_metadata = parse_filename(log_file)
        log_data.update(file_metadata)  # Combine metadata with log data
        log_data["Log_File_Name"] = log_file  # Add the log file name for reference
        all_data.append(log_data)
    
    # Convert the data into a DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns to match the file name order, followed by the extracted log data
    column_order = [
        "Model_Name", "TP", "Input_Size", "Output_Size", "Prompts", 
        "Context_Length", "Chunked_Prefill_Size", "Random_Range_Ratio",
        "Backend", "Successful requests", "Benchmark duration (s)", 
        "Total input tokens", "Total generated tokens", "Request throughput (req/s)", 
        "Input token throughput (tok/s)", "Output token throughput (tok/s)", 
        "Total token throughput (tok/s)", "Log_File_Name"
    ]
    df = df[column_order]
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Data successfully saved to {output_file}")

def main():
    """
    Main function to parse arguments and process logs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parse log files from a directory, extract configuration and benchmark results, "
            "and export the results to an Excel file."
        ),
        epilog=(
            "Example usage:\n"
            "  python process_logs_to_excel.py --logs-dir ./logs --output-file results.xlsx\n\n"
            "The script scans all .log files in the specified directory, extracts key metrics and configuration, "
            "and saves them to an Excel file."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--logs-dir",
        type=str,
        required=True,
        help="The directory containing log files to process."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="The name of the output Excel file (e.g., results.xlsx)."
    )
    
    args = parser.parse_args()
    
    # Process the log files and save results to Excel
    process_logs_to_excel(args.logs_dir, args.output_file)

if __name__ == "__main__":
    main()

