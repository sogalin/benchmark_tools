import sys
import os
import re
import sqlite3
import json
import argparse
import tarfile
from tqdm import tqdm

def rpd_to_trace_events(rpd_filename, start_time=None, end_time=None):
    """
    Parses .rpd file and extracts trace events in a format suitable for JSON output.
    """
    connection = sqlite3.connect(rpd_filename)
    trace_data = {"traceEvents": []}

    rangeStringApi = ""
    rangeStringOp = ""
    rangeStringMonitor = ""
    
    min_time = connection.execute("SELECT MIN(start) FROM rocpd_api;").fetchall()[0][0]
    max_time = connection.execute("SELECT MAX(end) FROM rocpd_api;").fetchall()[0][0]
    if min_time is None:
        raise Exception("Trace file is empty.")

    if start_time is not None:
        rangeStringApi = f"WHERE rocpd_api.start >= {start_time * 1000}"
        rangeStringOp = f"WHERE rocpd_op.start >= {start_time * 1000}"
        rangeStringMonitor = f"WHERE start >= {start_time * 1000}"
    if end_time is not None:
        rangeStringApi += f" AND rocpd_api.start <= {end_time * 1000}"
        rangeStringOp += f" AND rocpd_op.start <= {end_time * 1000}"
        rangeStringMonitor += f" AND start <= {end_time * 1000}"

    # Process GPU IDs
    for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):
        trace_data["traceEvents"].append({
            "name": "process_name", "ph": "M", "pid": row[0], "args": {"name": f"GPU{row[0]}"}
        })
        trace_data["traceEvents"].append({
            "name": "process_sort_index", "ph": "M", "pid": row[0], "args": {"sort_index": row[0] + 1000000}
        })

    # Process threads in rocpd_api
    for row in connection.execute("SELECT DISTINCT pid, tid FROM rocpd_api"):
        trace_data["traceEvents"].extend([
            {"name": "thread_name", "ph": "M", "pid": row[0], "tid": row[1], "args": {"name": f"Hip {row[1]}"}},
            {"name": "thread_sort_index", "ph": "M", "pid": row[0], "tid": row[1], "args": {"sort_index": row[1] * 2}}
        ])

    # Add rocpd_op data entries
    for row in connection.execute(
        f"SELECT A.string as optype, B.string as description, gpuId, queueId, rocpd_op.start/1000, (rocpd_op.end - rocpd_op.start) / 1000 "
        f"FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.opType_id INNER JOIN rocpd_string B ON B.id = rocpd_op.description_id {rangeStringOp}"
    ):
        name = row[0] if not row[1] else row[1]
        trace_data["traceEvents"].append({
            "pid": row[2], "tid": row[3], "name": name, "ts": row[4], "dur": row[5], "ph": "X", "args": {"desc": row[0]}
        })

    connection.close()
    return trace_data

def calculate_default_chunks(input_filename):
    file_size = os.path.getsize(input_filename)
    max_chunk_size = 2 * (1024 ** 3)  # 2 GB
    num_chunks = max(1, (file_size // max_chunk_size) + (1 if file_size % max_chunk_size > 0 else 0))
    return num_chunks

def split_trace_events(trace_data, input_filename, num_chunks):
    trace_events = trace_data['traceEvents']
    headers = [event for event in trace_events if 'ts' not in event]
    trace_events = [event for event in trace_events if 'ts' in event]
    trace_events.sort(key=lambda x: x['ts'])

    total_events = len(trace_events)
    chunk_size = total_events // num_chunks
    remainder = total_events % num_chunks
    chunks = []
    start = 0

    for i in tqdm(range(num_chunks), desc="Splitting traceEvents", unit="chunk"):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(trace_events[start:end])
        start = end

    output_files = []
    for i, chunk in enumerate(tqdm(chunks, desc="Writing JSON files", unit="file")):
        output_filename = f"{input_filename.split('.')[0]}_split_{i+1}.json"
        with open(output_filename, 'w') as outfile:
            outfile.write('{"traceEvents":[\n')
            for header in headers:
                outfile.write(json.dumps(header, separators=(',', ':')) + ',\n')
            for j, event in enumerate(chunk):
                json_line = json.dumps(event, separators=(',', ':'))
                if j < len(chunk) - 1:
                    json_line += ','
                outfile.write(json_line + '\n')
            outfile.write(']}')
        output_files.append(output_filename)

    tar_filename = f"{input_filename.split('.')[0]}_split.tar.gz"
    with tarfile.open(tar_filename, 'w:gz') as tar:
        for output_file in tqdm(output_files, desc="Creating tar.gz", unit="file"):
            tar.add(output_file)
    for output_file in output_files:
        os.remove(output_file)

    print(f"Splitting completed. The traceEvents have been compressed into {tar_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .rpd to traceEvents, split into chunks, and compress output.")
    parser.add_argument("input_rpd", help="The .rpd file containing traceEvents to be split.")
    parser.add_argument("--num-chunks", type=int, help="Number of chunks to split the traceEvents into.")
    parser.add_argument("--start", type=str, help="Start time in us or as a percentage (e.g., 50%)")
    parser.add_argument("--end", type=str, help="End time in us or as a percentage (e.g., 50%)")
    args = parser.parse_args()

    min_time, max_time = None, None
    if args.start:
        start_time = int(args.start.strip('%')) / 100 * (max_time - min_time) + min_time if "%" in args.start else int(args.start)
    else:
        start_time = None

    if args.end:
        end_time = int(args.end.strip('%')) / 100 * (max_time - min_time) + min_time if "%" in args.end else int(args.end)
    else:
        end_time = None

    trace_data = rpd_to_trace_events(args.input_rpd, start_time=start_time, end_time=end_time)
    
    if args.num_chunks is None:
        args.num_chunks = calculate_default_chunks(args.input_rpd)
    
    split_trace_events(trace_data, args.input_rpd, args.num_chunks)

