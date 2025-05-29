import re  
from collections import Counter  
import argparse  
import csv  
  
def count_shapes(log_file_path):  
    pattern = re.compile(r"x\.shape: torch\.Size\(\[(\d+),\s*(\d+)\]\)")  
    shape_counter = Counter()  
      
    with open(log_file_path, 'r') as file:  
        for line in file:  
            match = pattern.search(line)  
            if match:  
                shape = (int(match.group(1)), int(match.group(2)))  
                shape_counter[shape] += 1  
  
    return shape_counter  
  
def write_to_csv(shape_counts, output_file):  
    # Sort shapes based on the X-dimension  
    sorted_shapes = sorted(shape_counts.items(), key=lambda item: item[0][0])  
      
    with open(output_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)  
        # Write header  
        writer.writerow(['X-Dimension', 'Count'])  
        # Write sorted data  
        for shape, count in sorted_shapes:  
            writer.writerow([shape[0], count])  
  
def main():  
    # Set up argument parser  
    parser = argparse.ArgumentParser(description='Count shape occurrences in log file.')  
    parser.add_argument('log_file', type=str,   
                        help='Path to the log file to be processed.')  
    parser.add_argument('output_file', type=str,   
                        help='Path to the output CSV file.')  
  
    # Parse arguments  
    args = parser.parse_args()  
  
    # Count the shapes  
    shape_counts = count_shapes(args.log_file)  
  
    # Write the results to a CSV file  
    write_to_csv(shape_counts, args.output_file)  
  
if __name__ == '__main__':  
    main()  

