import cv2
import pyaudio
import wave
import threading

# Video Capture Setup
video_filename = "output.mp4"
video_capture = cv2.VideoCapture(0)  # 0 for default webcam
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

# Audio Capture Setup
audio_filename = "output.wav"
chunk = 1024  # Audio chunk size
format = pyaudio.paInt16
channels = 1
rate = 44100  # Sample rate
audio = pyaudio.PyAudio()
stream = audio.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk)
frames = []

# Flag to control recording
recording = True
frame_count = 0
captured_frame = False  # To ensure we only save one frame

# Function to record audio
def record_audio():
    global recording
    while recording:
        data = stream.read(chunk)
        frames.append(data)

# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

print("🎥 Recording video & audio... Press 'q' to stop.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    video_writer.write(frame)
    cv2.imshow("Recording", frame)
    
    # Capture a frame at the 50th frame for emotion analysis
    frame_count += 1
    if frame_count == 50 and not captured_frame:
        cv2.imwrite("frame.jpg", frame)  # Save frame
        print("📸 Frame captured for emotion analysis!")
        captured_frame = True  # Prevent multiple captures

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop recording
recording = False
audio_thread.join()

# Save Audio File
stream.stop_stream()
stream.close()
audio.terminate()
with wave.open(audio_filename, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))

# Release Video Capture
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

print("\n✅ Recording saved as output.mp4 and output.wav")
