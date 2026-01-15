import wave
import struct

def create_silent_wav(filename, duration=5.0):
    sample_rate = 44100
    n_frames = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        # Write silence
        data = struct.pack('<h', 0) * 2 * n_frames
        f.writeframes(data)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_silent_wav("test_audio.wav")
