import sqlite3
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def analyze_gpu_data(gpu_id, database_path):
    """Analyze kernel data for a given gpuId and return relevant statistics."""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT SUM(duration) 
        FROM kernel 
        WHERE gpuId = ?
    ''', (gpu_id,))
    total_duration = cursor.fetchone()[0] or 0

    cursor.execute('''
        SELECT kernelName, 
               SUM(duration) AS total_duration, 
               COUNT(*) AS call_count,
               AVG(duration) AS avg_duration
        FROM kernel 
        WHERE gpuId = ?
        GROUP BY kernelName
        ORDER BY total_duration DESC
    ''', (gpu_id,))
    rows = cursor.fetchall()
    
    conn.close()

    if rows:
        df = pd.DataFrame(rows, columns=['Kernel Name', 'Total Duration', 'Call Count', 'Average Duration'])
        df['Percentage of Total Time'] = (df['Total Duration'] / total_duration * 100).round(2)
        return (gpu_id, df, total_duration)
    else:
        return (gpu_id, None, 0)

def main(database_path, output_file, top_n=10, threshold=5):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(analyze_gpu_data, gpu_id, database_path): gpu_id for gpu_id in range(8)}
        results = []
        total_durations = {}

        with tqdm(total=len(futures), desc="Processing gpuIds") as pbar:
            for future in as_completed(futures):
                gpu_id = futures[future]
                result = future.result()
                results.append(result)
                total_durations[gpu_id] = result[2]
                pbar.update(1)

    results.sort(key=lambda x: x[0])

    with pd.ExcelWriter(output_file) as writer:
        summary_data = []

        for gpu_id, df, total_duration in results:
            if df is not None:
                sheet_name = f'gpuId_{gpu_id}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"{sheet_name} data written to Excel")
                summary_data.append({'gpuId': gpu_id, 'Total Duration': total_duration})
            else:
                print(f"gpuId {gpu_id}: No data available")

        summary_df = pd.DataFrame(summary_data)
        min_duration = summary_df['Total Duration'].min()
        min_gpu_id = summary_df.loc[summary_df['Total Duration'] == min_duration, 'gpuId'].values[0]
        print(f"Minimum gpuId: {min_gpu_id} with duration: {min_duration}")

        summary_df['Problematic'] = summary_df['Total Duration'].apply(lambda x: 'Yes' if (x - min_duration) / min_duration * 100 > threshold else 'No')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        start_row = len(summary_df) + 2
        problematic_gpu_ids = summary_df[summary_df['Problematic'] == 'Yes']['gpuId'].tolist()
        if problematic_gpu_ids:
            print(f"Reference gpuId: {min_gpu_id} with duration: {min_duration}")

            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT kernelName, 
                       SUM(duration) AS total_duration 
                FROM kernel 
                WHERE gpuId = ?
                GROUP BY kernelName
            ''', (min_gpu_id,))
            min_kernel_data = cursor.fetchall()
            conn.close()

            if min_kernel_data:
                min_kernel_df = pd.DataFrame(min_kernel_data, columns=['Kernel Name', 'Min Total Duration'])
                comparison_data = []

                for gpu_id in problematic_gpu_ids:
                    conn = sqlite3.connect(database_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT kernelName, 
                               SUM(duration) AS total_duration 
                        FROM kernel 
                        WHERE gpuId = ?
                        GROUP BY kernelName
                    ''', (gpu_id,))
                    kernel_data = cursor.fetchall()
                    cursor.close()

                    kernel_df = pd.DataFrame(kernel_data, columns=['Kernel Name', 'Problematic Total Duration'])
                    comparison = min_kernel_df.merge(kernel_df, on='Kernel Name', how='outer').fillna(0)
                    comparison['Difference'] = comparison['Problematic Total Duration'] - comparison['Min Total Duration']
                    comparison['gpuId'] = gpu_id
                    comparison_data.append(comparison)

                all_comparisons = pd.concat(comparison_data)
                top_diff_kernels = all_comparisons.sort_values(by='Difference', ascending=False).head(5)
                top_diff_kernels.to_excel(writer, sheet_name='Summary', index=False, startrow=start_row)
                print(f"Top differences written to {output_file}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze kernel statistics for each gpuId from an SQLite database and export to an Excel file.")
    parser.add_argument("database_path", type=str, help="Path to the SQLite database file containing kernel data.")
    parser.add_argument("output_file", type=str, help="Path for the output Excel file. Must have an .xlsx extension.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top kernels to compare, default is 10.")
    parser.add_argument("--threshold", type=float, default=5, help="Percentage threshold to flag problematic kernel durations, default is 5.")
    args = parser.parse_args()

    # Check if output file extension is .xlsx; if not, prompt the user to correct it
    if not args.output_file.endswith('.xlsx'):
        print("Warning: Only .xlsx format is supported for output.")
        correct_extension = input(f"Do you want to change the output file extension to .xlsx? (y/n): ").strip().lower()
        if correct_extension == 'y':
            args.output_file = args.output_file.rsplit('.', 1)[0] + '.xlsx'
            print(f"Output file extension corrected to: {args.output_file}")
        else:
            print("Error: Unsupported file extension. Please provide an output file with the .xlsx extension.")
            exit(1)

    main(args.database_path, args.output_file, args.top_n, args.threshold)

