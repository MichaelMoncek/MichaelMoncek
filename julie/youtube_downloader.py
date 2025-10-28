import yt_dlp

def download_as_mp3(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',  # Highest quality
        }],
        'quiet': False,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = "https://youtu.be/QaXgEXZtmIc?si=eI8U3LmWlo-VfyLk" # input("Enter the YouTube video URL: ")
    download_as_mp3(video_url)
