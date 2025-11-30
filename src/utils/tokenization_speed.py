#!/usr/bin/env python3
"""
Estimates tokenization time for peoples_speech datasets based on a known benchmark.
Benchmark provided: 12.53 hours to process 1,084.69 hours of audio (clean_sa).
"""

def main():
    # --- 1. CONFIGURATION ---
    
    # Benchmark Data (from your specific run)
    benchmark_folder = "clean_sa"
    benchmark_audio_hours = 1084.69
    benchmark_process_time_hours = 12.53
    
    # Dataset Statistics (from your previous output)
    dataset_stats = {
        "clean": 5987.71,
        "clean_sa": 1084.69,
        "dirty": 21791.63,
        "dirty_sa": 2223.17,
        "microset": 1.32,
        "test": 59.80,
        "validation": 33.16,
    }

    # --- 2. CALCULATE SPEED ---
    
    # How many hours of processing time per 1 hour of audio?
    # Ratio = 12.53 / 1084.69 ≈ 0.01155
    processing_ratio = benchmark_process_time_hours / benchmark_audio_hours
    
    # How many hours of audio can we process in 1 hour? (Real-time factor)
    # Speed = 1084.69 / 12.53 ≈ 86.5x real-time
    processing_speed_x = benchmark_audio_hours / benchmark_process_time_hours

    print(f"\n--- BENCHMARK ---")
    print(f"Reference: {benchmark_folder}")
    print(f"Speed: {processing_speed_x:.2f}x real-time")
    print(f"Rate: Processing 1,000 hours of audio takes approx {1000 * processing_ratio:.2f} hours.\n")

    # --- 3. GENERATE ESTIMATES ---

    header = f"{'FOLDER':<12} | {'AUDIO (h)':<12} | {'EST. HOURS':<12} | {'EST. DAYS':<10}"
    sep = "-" * len(header)
    
    print(header)
    print(sep)
    
    total_est_hours = 0
    total_audio = 0

    for folder, hours in dataset_stats.items():
        # Estimate time
        est_hours = hours * processing_ratio
        est_days = est_hours / 24
        
        # Totals
        total_est_hours += est_hours
        total_audio += hours
        
        print(f"{folder:<12} | {hours:<12,.2f} | {est_hours:<12,.2f} | {est_days:<10,.2f}")

    # --- 4. TOTALS ---
    total_est_days = total_est_hours / 24
    
    print(sep)
    print(f"{'TOTAL':<12} | {total_audio:<12,.2f} | {total_est_hours:<12,.2f} | {total_est_days:<10,.2f}")
    print(sep)
    print(f"\nNote: Estimates assume linear scaling based on {benchmark_folder} performance.")

if __name__ == "__main__":
    main()