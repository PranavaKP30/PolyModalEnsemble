#!/usr/bin/env python3
"""
Master script to run all preprocessing pipelines with detailed progress tracking
"""

import sys
import time
import datetime
import threading
from pathlib import Path

def log_with_timestamp(message, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    sys.stdout.flush()  # Force immediate output

def progress_monitor(stop_event, dataset_name):
    """Background thread to show progress dots"""
    while not stop_event.is_set():
        print(".", end="", flush=True)
        time.sleep(5)  # Print dot every 5 seconds

def run_amazon_reviews_preprocessing():
    """Run Amazon Reviews preprocessing with detailed progress"""
    print("\n" + "="*60)
    print("🛒 AMAZON REVIEWS PREPROCESSING")
    print("="*60)
    
    start_time = time.time()
    log_with_timestamp("Starting Amazon Reviews preprocessing...")
    
    try:
        log_with_timestamp("Importing AmazonReviewsPreProcessor...")
        from AmazonReviewsPreProcess import AmazonReviewsPreProcessor
        
        log_with_timestamp("Initializing preprocessor...")
        preprocessor = AmazonReviewsPreProcessor(
            data_dir="../Data/AmazonReviews",
            output_dir="../../ProcessedData/AmazonReviews"
        )
        
        log_with_timestamp("Starting data processing (FULL DATASET CONVERSION)...")
        log_with_timestamp("Processing ALL reviews + metadata...")
        
        # Start progress monitor
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=progress_monitor, args=(stop_event, "AmazonReviews"))
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            results = preprocessor.process()
            stop_event.set()
            progress_thread.join(timeout=1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print()  # New line after progress dots
            log_with_timestamp(f"Amazon Reviews preprocessing completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print("\n✅ Amazon Reviews preprocessing completed successfully!")
            print("Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            stop_event.set()
            progress_thread.join(timeout=1)
            raise e
        
    except Exception as e:
        print()  # New line after progress dots
        log_with_timestamp(f"Amazon Reviews preprocessing failed: {e}", "ERROR")
        return False

def run_coco_captions_preprocessing():
    """Run COCO Captions preprocessing with detailed progress"""
    print("\n" + "="*60)
    print("🖼️ COCO CAPTIONS PREPROCESSING")
    print("="*60)
    
    start_time = time.time()
    log_with_timestamp("Starting COCO Captions preprocessing...")
    
    try:
        log_with_timestamp("Importing CocoCaptionsPreProcessor...")
        from CocoCaptionsPreProcess import CocoCaptionsPreProcessor
        
        log_with_timestamp("Initializing preprocessor...")
        preprocessor = CocoCaptionsPreProcessor(
            data_dir="../Data/CocoCaptions",
            output_dir="../../ProcessedData/CocoCaptions"
        )
        
        log_with_timestamp("Starting data processing (FULL DATASET CONVERSION)...")
        log_with_timestamp("Processing ALL captions + images...")
        
        # Start progress monitor
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=progress_monitor, args=(stop_event, "CocoCaptions"))
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            results = preprocessor.process()
            stop_event.set()
            progress_thread.join(timeout=1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print()  # New line after progress dots
            log_with_timestamp(f"COCO Captions preprocessing completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print("\n✅ COCO Captions preprocessing completed successfully!")
            print("Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            stop_event.set()
            progress_thread.join(timeout=1)
            raise e
        
    except Exception as e:
        print()  # New line after progress dots
        log_with_timestamp(f"COCO Captions preprocessing failed: {e}", "ERROR")
        return False

def run_yelp_preprocessing():
    """Run Yelp preprocessing with detailed progress"""
    print("\n" + "="*60)
    print("🍕 YELP OPEN DATASET PREPROCESSING")
    print("="*60)
    
    start_time = time.time()
    log_with_timestamp("Starting Yelp preprocessing...")
    
    try:
        log_with_timestamp("Importing YelpPreProcessor...")
        from YelpPreProcess import YelpPreProcessor
        
        log_with_timestamp("Initializing preprocessor...")
        preprocessor = YelpPreProcessor(
            data_dir="../Data/YelpOpen",
            output_dir="../../ProcessedData/YelpOpen"
        )
        
        log_with_timestamp("Starting data processing (FULL DATASET CONVERSION)...")
        log_with_timestamp("Processing ALL businesses + reviews...")
        
        # Start progress monitor
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=progress_monitor, args=(stop_event, "Yelp"))
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            results = preprocessor.process()
            stop_event.set()
            progress_thread.join(timeout=1)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print()  # New line after progress dots
            log_with_timestamp(f"Yelp preprocessing completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print("\n✅ Yelp preprocessing completed successfully!")
            print("Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            stop_event.set()
            progress_thread.join(timeout=1)
            raise e
        
    except Exception as e:
        print()  # New line after progress dots
        log_with_timestamp(f"Yelp preprocessing failed: {e}", "ERROR")
        return False

def main():
    """Main function to run all preprocessing pipelines"""
    print("🚀 STARTING ALL PREPROCESSING PIPELINES")
    print("="*60)
    
    overall_start_time = time.time()
    log_with_timestamp("Initializing preprocessing pipeline...")
    
    # Track success/failure
    results = {}
    
    # Run Amazon Reviews preprocessing
    log_with_timestamp("=== STARTING AMAZON REVIEWS ===")
    results['amazon_reviews'] = run_amazon_reviews_preprocessing()
    
    # Run COCO Captions preprocessing
    log_with_timestamp("=== STARTING COCO CAPTIONS ===")
    results['coco_captions'] = run_coco_captions_preprocessing()
    
    # Run Yelp preprocessing
    log_with_timestamp("=== STARTING YELP ===")
    results['yelp'] = run_yelp_preprocessing()
    
    # Summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print("\n" + "="*60)
    print("📊 PREPROCESSING SUMMARY")
    print("="*60)
    
    successful = sum(results.values())
    total = len(results)
    
    log_with_timestamp(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    log_with_timestamp(f"Successful datasets: {successful}/{total}")
    
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        log_with_timestamp(f"{dataset.replace('_', ' ').title()}: {status}")
    
    if successful == total:
        print("\n🎉 ALL PREPROCESSING PIPELINES COMPLETED SUCCESSFULLY!")
        print("📁 Processed data is ready in the ProcessedData/ directory")
        print("🔧 Preprocessing components saved for future use")
        log_with_timestamp("All preprocessing completed successfully!")
        return 0
    else:
        print(f"\n⚠️ {total - successful} preprocessing pipeline(s) failed")
        print("Check the error messages above for details")
        log_with_timestamp(f"{total - successful} preprocessing pipeline(s) failed", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())
