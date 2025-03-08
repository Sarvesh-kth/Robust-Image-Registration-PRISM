import cv2
import os
import argparse
import shutil

def clear_directory(directory_path):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Remove all files and subdirectories in the directory
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isdir(item_path):
                    # Remove subdirectories and their contents
                    shutil.rmtree(item_path)
                else:
                    # Remove files
                    os.remove(item_path)

            print(f'Directory {directory_path} cleared successfully.')
        else:
            print(f'Directory {directory_path} does not exist.')
    except Exception as e:
        print(f'Error clearing directory: {str(e)}')
def extract_frames(video_file, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames
    w = 0
    while True and w < 25:
        ret, frame = cap.read()

        # Break the loop when we reach the end of the video
        if not ret:
            break

        # Define the output file name (e.g., frame_001.jpg)


        # Write the frame to the output file
        if frame_count % 1 == 0:
            w+=1
            frame_filename = os.path.join(output_directory, f'frame_{w:04d}.jpg')
            cv2.imwrite(frame_filename, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'Frames extracted: {w}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video file.')
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('output_directory', type=str, help='Path to the output directory')
    args = parser.parse_args()
    clear_directory(args.output_directory)
    extract_frames(args.video_file, args.output_directory)
