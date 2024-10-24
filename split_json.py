import json
import argparse
import tarfile
import os

# Function to calculate the number of chunks based on the input file size
def calculate_default_chunks(input_filename):
    # Get the size of the input file in bytes
    file_size = os.path.getsize(input_filename)
    # Maximum size per chunk in bytes (2 GB)
    max_chunk_size = 2 * (1024 ** 3)  # 2 GB
    # Calculate the number of chunks needed
    num_chunks = max(1, (file_size // max_chunk_size) + (1 if file_size % max_chunk_size > 0 else 0))
    return num_chunks

# Function to split the trace events from a given JSON file
def split_trace_events(input_filename, num_chunks):
    # Read the JSON file
    with open(input_filename, 'r') as f:
        data = json.load(f)

    # Get the total number of traceEvents
    trace_events = data['traceEvents']
    total_events = len(trace_events)

    # Calculate the number of events per chunk
    chunk_size = total_events // num_chunks
    remainder = total_events % num_chunks

    # Split traceEvents into the specified number of chunks
    chunks = []
    start = 0

    for i in range(num_chunks):
        # If there is a remainder, the first few chunks get an extra event
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(trace_events[start:end])
        start = end

    # Save the chunks into temporary files
    output_files = []
    for i, chunk in enumerate(chunks):
        split_data = {'traceEvents': chunk}
        output_filename = f"{input_filename.split('.')[0]}_split_{i+1}.json"
        with open(output_filename, 'w') as outfile:
            json.dump(split_data, outfile, indent=4)
        output_files.append(output_filename)

    # Create a tar.gz archive of the output files
    tar_filename = f"{input_filename.split('.')[0]}_split.tar.gz"
    with tarfile.open(tar_filename, 'w:gz') as tar:
        for output_file in output_files:
            tar.add(output_file)
    # Remove the individual JSON files after compression
    for output_file in output_files:
        os.remove(output_file)

    print(f"Splitting completed. The traceEvents have been compressed into {tar_filename}.")

# Main execution block
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split traceEvents from a JSON file into a specified number of parts and compress the output.")
    parser.add_argument("input_filename", help="The JSON file containing traceEvents to be split.")
    parser.add_argument("--num-chunks", type=int, help="Number of chunks to split the traceEvents into.")

    # Parse arguments
    args = parser.parse_args()

    # Determine the default number of chunks based on the input file size if not specified
    if args.num_chunks is None:
        args.num_chunks = calculate_default_chunks(args.input_filename)

    # Call the split function with the provided input filename and number of chunks
    split_trace_events(args.input_filename, args.num_chunks)

