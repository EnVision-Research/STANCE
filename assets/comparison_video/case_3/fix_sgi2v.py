#!/usr/bin/env python3
"""
Script to re-encode sgi2v_3.mp4 using imageio for better compatibility
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

def main():
    input_file = "/Users/zchen/Desktop/Study/STANCE/STANCE_page/assets/comparison_video/case_3/sgi2v_3.mp4"
    output_file = "/Users/zchen/Desktop/Study/STANCE/STANCE_page/assets/comparison_video/case_3/sgi2v_3_fixed.mp4"
    backup_file = "/Users/zchen/Desktop/Study/STANCE/STANCE_page/assets/comparison_video/case_3/sgi2v_3_backup.mp4"
    
    print(f"Re-encoding: {input_file}")
    print(f"Output: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
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
    else:
        print("✗ Failed to fix video")

if __name__ == "__main__":
    main() 