import whisper
from transformers import pipeline
from deepface import DeepFace
import warnings
import re
import time
import wave
import cv2

warnings.filterwarnings("ignore")

# Load Whisper model
model = whisper.load_model("base")

# Transcribe audio file
audio_path = "output.wav"  # Change this to your actual file
start_time = time.time()
result = model.transcribe(audio_path)
end_time = time.time()
transcription = result["text"]

# Calculate Speech Rate (WPM)
audio_duration = end_time - start_time  # Approximate duration in seconds
total_words = len(transcription.split())
wpm = (total_words / audio_duration) * 60

# Classify Speech Rate
if wpm < 100:
    speech_rate = "Slow (Possible nervousness or careful speech)"
elif 100 <= wpm <= 160:
    speech_rate = "Normal (Confident, natural pace)"
else:
    speech_rate = "Fast (Possible anxiety or excitement)"

# Emotion Detection using Hugging Face model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
emotion_result = emotion_classifier(transcription)

detected_text_emotion = emotion_result[0][0]['label']
confidence_score = emotion_result[0][0]['score']

# Confidence Score Analysis (Detect Filler Words)
filler_words = re.findall(r"\b(uh+|umm+|I\.{2,})\b", transcription, re.IGNORECASE)
filler_count = len(filler_words)

# Determine Confidence Level
if filler_count > 5:
    confidence_level = "Low (Many hesitations detected)"
elif 2 <= filler_count <= 5:
    confidence_level = "Medium (Some hesitations)"
else:
    confidence_level = "High (Clear and confident speech)"

# Facial Emotion Detection using DeepFace
image_path = "frame.jpg"  # Change this to the actual captured frame path
try:
    facial_emotion = DeepFace.analyze(img_path=image_path, actions=['emotion'])
    detected_facial_emotion = facial_emotion[0]['dominant_emotion']
except Exception as e:
    detected_facial_emotion = "Error detecting emotion"

# Output results
print("📝 Transcribed Text:", transcription)
print("🎭 Text-Based Emotion:", detected_text_emotion, f"(Confidence: {confidence_score:.2f})")
print("📸 Facial Emotion Detected:", detected_facial_emotion)
print("⚡ Speech Rate:", f"{wpm:.2f} WPM - {speech_rate}")
print("💬 Confidence Level:", confidence_level)