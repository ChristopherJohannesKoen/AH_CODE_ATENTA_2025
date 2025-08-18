import whisper
import os
import torch
import math
import numpy as np
import ffmpeg
from tqdm import tqdm
from pyannote.audio import Pipeline

# Function to load audio using FFmpeg with error handling
def load_audio_ffmpeg(audio_file, sample_rate=16000):
    try:
        # Use ffmpeg to load the audio file
        out, err = (
            ffmpeg.input(audio_file, threads=0)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=f"{sample_rate}")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")  # Safely decode stderr
        raise e  # Raise the exception to stop the program if there's an error

# Function for speaker diarization using pyannote
def diarize_audio(audio_file):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="HUGGINGFACE_TOKEN_REDACTED")
        diarization_result = pipeline(audio_file)
        return diarization_result
    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        raise e

# Function to transcribe audio with progress bar and speaker labels
def transcribe_audio_with_diarization(audio_file, output_file):
    # Load the Whisper model and move it to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small").to(device)

    # Perform speaker diarization
    diarization_result = diarize_audio(audio_file)

    # Load the audio data using FFmpeg
    audio = load_audio_ffmpeg(audio_file)

    # Whisper processes at 16kHz sample rate
    sample_rate = 16000

    # Initialize a list to store the transcription results
    full_transcription = []

    # Process each speaker segment with Whisper and update the progress bar
    for segment in tqdm(diarization_result.itertracks(yield_label=True), desc="Transcribing with speakers"):
        start = int(segment[0].start * sample_rate)
        end = int(segment[0].end * sample_rate)
        speaker_label = segment[2]  # Speaker label (e.g., Speaker 1, Speaker 2)

        # Extract the chunk of audio data for this speaker segment
        audio_chunk = audio[start:end]

        # Process the chunk with Whisper
        audio_chunk = whisper.pad_or_trim(np.array(audio_chunk))
        mel = whisper.log_mel_spectrogram(audio_chunk).to(device)

        # Decode the transcription
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # Append the text with speaker labels to the full transcription
        full_transcription.append(f"{speaker_label}: {result.text}")

    # Combine all transcribed chunks with speaker labels into the final transcription
    transcription_text = '\n'.join(full_transcription)

    # Save the transcription to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    print(f"Transcription with speaker labels saved to: {output_file}")

# Function to convert mp3 to wav using FFmpeg if needed
def convert_mp3_to_wav(input_file, output_file):
    try:
        ffmpeg.input(input_file).output(output_file).run()
        print(f"Converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error during conversion: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        raise e

# Main logic to decide whether to use mp3 or wav
def main(audio_file, output_file):
    # Check if the file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found.")
        return

    if audio_file.endswith('.mp3'):
        # If it's an mp3 file, convert it to wav first
        wav_file = audio_file.replace(".mp3", ".wav")
        convert_mp3_to_wav(audio_file, wav_file)
        transcribe_audio_with_diarization(wav_file, output_file)
    else:
        # If it's already a wav file, just transcribe
        transcribe_audio_with_diarization(audio_file, output_file)

# Replace 'your_audio_file.mp3' with the path to your audio file
# Specify the output text file where the transcription will be saved
main("resources\audio\consultation_x1_combined_dialogue.mp3", "transcription_output-M1.txt")
