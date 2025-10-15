#!/usr/bin/env python3
"""
Script to re-encode all sgi2v videos using imageio for better compatibility
"""

import imageio.v2 as imageio
import numpy as np
import os

def fix_video(input_path, output_path):
    """Re-encode video using imageio with libx264 codec"""
    try:
        # Read the problematic video
        reader = imageio.get_reader(input_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 30)
        
        print(f"Input video info: {meta}")
        print(f"FPS: {fps}")
        
        # Create output writer with libx264 codec
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        
        frame_count = 0
        
        # Process each frame
        for frame in reader:
            # Ensure frame is in correct format
            if frame.ndim == 2:
                # Convert grayscale to RGB
                frame = np.stack([frame] * 3, axis=-1)
            
            writer.append_data(frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")
        
        # Clean up
        reader.close()
        writer.close()
        
        print(f"Successfully re-encoded: {output_path}")
        print(f"Total frames: {frame_count}")
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

def fix_sgi2v_in_case(case_dir):
    """Fix sgi2v video in a specific case directory"""
    input_file = os.path.join(case_dir, "sgi2v_1.mp4" if "case_1" in case_dir else 
                              "sgi2v_2.mp4" if "case_2" in case_dir else "sgi2v_3.mp4")
    output_file = input_file.replace(".mp4", "_fixed.mp4")
    backup_file = input_file.replace(".mp4", "_backup.mp4")
    
    print(f"\n{'='*50}")
    print(f"Processing: {case_dir}")
    print(f"Re-encoding: {input_file}")
    print(f"Output: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Re-encode the video
    success = fix_video(input_file, output_file)
    
    if success:
        # Backup original file
        print(f"Backing up original to: {backup_file}")
        os.rename(input_file, backup_file)
        
        # Replace with fixed version
        print(f"Replacing with fixed version...")
        os.rename(output_file, input_file)
        
        print("✓ Video fixed successfully!")
        return True
    else:
        print("✗ Failed to fix video")
        return False

def main():
    base_dir = "/Users/zchen/Desktop/Study/STANCE/STANCE_page/assets/comparison_video"
    
    cases = ["case_1", "case_2", "case_3"]
    
    print("Starting to fix all sgi2v videos...")
    
    for case in cases:
        case_dir = os.path.join(base_dir, case)
        if os.path.exists(case_dir):
            fix_sgi2v_in_case(case_dir)
        else:
            print(f"Warning: {case_dir} not found")
    
    print(f"\n{'='*50}")
    print("All sgi2v videos processing completed!")

if __name__ == "__main__":
    main() 