# Install PyTube (run this cell first)
!pip install pytube

# Import necessary libraries
from pytube import YouTube
from google.colab import files
import os

def download_youtube_video(url, output_path='/content'):
    try:
        # Create a YouTube object
        yt = YouTube(url)
        
        # Get the highest resolution progressive stream (usually MP4)
        video = yt.streams.get_highest_resolution()
        
        # Download the video
        print(f"Downloading: {yt.title}")
        output_file = video.download(output_path)
        print("Download completed!")
        
        return output_file
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your desired video URL
output_file = download_youtube_video(video_url)

if output_file and os.path.exists(output_file):
    print(f"Video saved as: {output_file}")
    files.download(output_file)  # This will prompt a download in Colab
else:
    print("Failed to download the video.")
