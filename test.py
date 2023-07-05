from TikTokApi import TikTokApi

# Initialize TikTokApi
api = TikTokApi()

# TikTok video URL
video_url = "https://www.tiktok.com/@tiktok/video/1234567890"

# Extract video ID from the URL
video_id = video_url.split("/video/")[1]

# Get TikTok video data
video_data = api.get_tiktok_by_id(video_id)

# Get video URL
video_download_url = video_data['itemInfo']['itemStruct']['video']['downloadAddr']

# Download the video
response = api.download_video(video_download_url, output_file='video.mp4')

# Check if the video was successfully downloaded
if response['status'] == 'success':
    print("Video downloaded successfully!")
else:
    print("Failed to download the video.")
