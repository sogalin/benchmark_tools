import sqlite3
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def analyze_gpu_data(gpu_id, database_path):
    """Function to analyze kernel data for a given gpuId."""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Calculate total duration for this gpuId
    cursor.execute('''
        SELECT SUM(duration) 
        FROM kernel 
        WHERE gpuId = ?
    ''', (gpu_id,))
    total_duration = cursor.fetchone()[0]
    total_duration = total_duration if total_duration is not None else 0

    # Fetch kernel stats for this gpuId
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
    
    # Create DataFrame if there are rows
    if rows:
        df = pd.DataFrame(rows, columns=['Kernel Name', 'Total Duration', 'Call Count', 'Average Duration'])
        df['Percentage of Total Time'] = (df['Total Duration'] / total_duration * 100).round(2)
        return (gpu_id, df, total_duration)
    else:
        return (gpu_id, None, 0)

def main(database_path, output_file, top_n=10, threshold=5):
    # Run analysis in parallel for gpuIds 0 through 7 with a progress bar
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(analyze_gpu_data, gpu_id, database_path): gpu_id for gpu_id in range(8)}
        results = []
        total_durations = {}
        
        with tqdm(total=len(futures), desc="Processing gpuIds") as pbar:
            for future in as_completed(futures):
                gpu_id = futures[future]
                result = future.result()
                results.append(result)
                total_durations[gpu_id] = result[2]  # Store total duration for each gpuId
                pbar.update(1)

    # Sort results by gpu_id to ensure sheets are in order
    results.sort(key=lambda x: x[0])

    # Create an Excel file with a sheet for each gpuId
    with pd.ExcelWriter(output_file) as writer:
        summary_data = []

        for gpu_id, df, total_duration in results:
            if df is not None:
                sheet_name = f'gpuId_{gpu_id}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"{sheet_name} data written to Excel")

                # Append total durations for summary
                summary_data.append({'gpuId': gpu_id, 'Total Duration': total_duration})

            else:
                print(f"gpuId {gpu_id}: No data available")

        # Write summary table for total durations
        summary_df = pd.DataFrame(summary_data)
        
        # Determine the minimum total duration across all gpuIds
        min_duration = summary_df['Total Duration'].min()
        min_gpu_id = summary_df.loc[summary_df['Total Duration'] == min_duration, 'gpuId'].values[0]
        print(f"Minimum gpuId: {min_gpu_id} with duration: {min_duration}")

        # Compare each gpuId to the minimum duration
        summary_df['Problematic'] = summary_df['Total Duration'].apply(lambda x: 'Yes' if (x - min_duration) / min_duration * 100 > threshold else 'No')
        
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Insert empty rows for spacing
        start_row = len(summary_df) + 2

        # Second table: Analyze problematic gpuIds
        problematic_gpu_ids = summary_df[summary_df['Problematic'] == 'Yes']['gpuId'].tolist()
        if problematic_gpu_ids:
            print(f"Reference gpuId: {min_gpu_id} with duration: {min_duration}")

            # Fetch kernel stats for the min gpuId
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

            # Check if min gpuId has kernel data
            if not min_kernel_data:
                print(f"No kernel data found for min gpuId {min_gpu_id}.")
            else:
                print(f"Kernel data for min gpuId {min_gpu_id}: {min_kernel_data}")

            min_kernel_df = pd.DataFrame(min_kernel_data, columns=['Kernel Name', 'Min Total Duration'])

            # Create a DataFrame to hold all problematic comparisons
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
                comparison['gpuId'] = gpu_id  # Add gpuId for identification
                comparison_data.append(comparison)

            # Combine all comparison data
            all_comparisons = pd.concat(comparison_data)

            # Get top 5 kernels with the most significant differences
            top_diff_kernels = all_comparisons.sort_values(by='Difference', ascending=False).head(5)

            # Write the comparison results
            top_diff_kernels.to_excel(writer, sheet_name='Summary', index=False, startrow=start_row)
            print(f"Top differences written to {output_file}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze kernel statistics for each gpuId from an SQLite database and export to Excel")
    parser.add_argument("database_path", type=str, help="Path to the SQLite database file")
    parser.add_argument("output_file", type=str, help="Output Excel file path")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top kernels to compare")
    parser.add_argument("--threshold", type=float, default=5, help="Percentage threshold for difference in kernel durations")
    args = parser.parse_args()
    
    main(args.database_path, args.output_file, args.top_n, args.threshold)

