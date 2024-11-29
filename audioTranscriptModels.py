"""
    Read .wav into transcript made by google model
"""
# import speech_recognition as sr
# audio_file = "output.wav"
# r = sr.Recognizer()

# try:

#     with sr.AudioFile(audio_file) as source:
#         # Get the total duration of the audio file in seconds
#         duration = int(source.DURATION)  # Total duration of the audio in seconds
        
#         print(f"Processing audio file: {audio_file}, Duration: {duration} seconds")

#         # Loop through the audio file in 60-second chunks
#         for start_time in range(0, duration, 60):
#             print(f"Processing from {start_time} to {min(start_time + 60, duration)} seconds...")

#             # Record the next 60 seconds of audio
#             audio = r.record(source, offset=start_time, duration=60)
            
#             try:
#                 # Transcribe the audio chunk
#                 command = r.recognize_google(audio)
#                 print(f"Chunk [{start_time} - {min(start_time + 60, duration)}]: {command}")

#                 # Save the transcription to a file
#                 with open("GoogleTranscript.txt", "a") as file:
#                     file.write(f"[{start_time} - {min(start_time + 60, duration)}]: {command}\n")
            
#             except sr.UnknownValueError:
#                 print(f"Chunk [{start_time} - {start_time + 60}]: Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Chunk [{start_time} - {start_time + 60}]: Request failed: {e}")

# except FileNotFoundError:
#     print(f"Audio file '{audio_file}' not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")
import whisper

model = whisper.load_model("medium")

print("Loaded Model")

result = model.transcribe("output.wav")

with open("OpenAITranscript.txt", "a") as file:
    file.write(result["text"])

print(result["text"])