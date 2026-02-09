import os 
import sys
import yt_dlp
from faster_whisper import WhisperModel

# AI Disclaimer: Parts of this code were generated using AI.

# Uses yt_dlp and ffmpeg to download audio from a YouTube video and converts it to WAV format.
# Returns the path to the .wav file.
def download_audio_from_youtube(url, output_filename="downloaded_audio.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename.replace(".wav", "") + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': os.getcwd()
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    base_name = output_filename.replace(".wav", "")
    for file in os.listdir():
        if file.startswith(base_name) and file.endswith(".wav"):
            return file

    raise FileNotFoundError("Audio download or conversion failed.")

# Uses the faster_whisper libary to transcribe the audio input to plain text
def transcribe_file(wav_file, model_size="base.en", output_transcript="transcript.txt"):
    model = WhisperModel(model_size, device="auto", compute_type="int8")

    segments, info = model.transcribe(wav_file)  #The model transcribes the video in segments
    transcript_text = ""
    for seg in segments:
        transcript_text += seg.text.strip() + "\n"

    # Save transcript to file
    with open(output_transcript, "w") as file:
        file.write(transcript_text)
    
def main():

    # Arguments passed in when running the file
    if len(sys.argv) >= 2:
        input_arg = sys.argv[1]  #holds youtube link
        model_size = sys.argv[2] if len(sys.argv) == 3 else "base.en"  # Specify a model, otherwise default to base model
        output_transcript = sys.argv[3] if len(sys.argv) >= 4 else "transcript.txt"  # Option to specify a custom output file

        if input_arg.startswith("https://"):
            wav_file = download_audio_from_youtube(input_arg)
        else:
            wav_file = input_arg

        transcribe_file(wav_file, model_size, output_transcript)

    else:
        print("Usage: python transcribe_draft.py <wav_file or YouTube_URL> [<model_size>] [<output_transcript>]")

if __name__ == "__main__":
    main()
