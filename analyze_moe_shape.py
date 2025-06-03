import argparse  
import csv  
from collections import defaultdict  
  
def count_batch_sizes(log_file_path):  
    prefill_counter = defaultdict(int)  
    decode_counter = defaultdict(int)  
      
    with open(log_file_path, 'r') as file:  
        for line in file:  
            line = line.strip()  
            if 'prefill - batch_size =' in line:  
                parts = line.split('prefill - batch_size = ')[1].split(',')  
                token_length = int(parts[1].strip())  # 使用 token length 來表示 total batch size  
                prefill_counter[token_length] += 1  
            elif 'decode - batch_size =' in line:  
                parts = line.split('decode - batch_size = ')[1].split(',')  
                batch_size = int(parts[0].strip())  
                total_batch_size = batch_size * 1  # 對應 total batch size  
                decode_counter[total_batch_size] += 1  
  
    return prefill_counter, decode_counter  
  
def write_to_csv(prefill_counts, decode_counts, output_file):  
    with open(output_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)  
        # Write header  
        writer.writerow(['Category', 'Total Batch Size', 'Count'])  
  
        # Write prefill data  
        for total_batch_size, count in sorted(prefill_counts.items()):  
            writer.writerow(['Prefill', total_batch_size, count])  
  
        # Write decode data  
        for total_batch_size, count in sorted(decode_counts.items()):  
            writer.writerow(['Decode', total_batch_size, count])  
  
def main():  
    parser = argparse.ArgumentParser(description='Count batch size occurrences in log file.')  
    parser.add_argument('log_file', type=str,   
                        help='Path to the log file to be processed.')  
    parser.add_argument('output_file', type=str,   
                        help='Path to the output CSV file.')  
  
    args = parser.parse_args()  
  
    prefill_counts, decode_counts = count_batch_sizes(args.log_file)  
  
    write_to_csv(prefill_counts, decode_counts, args.output_file)  
  
if __name__ == '__main__':  
    main()  

