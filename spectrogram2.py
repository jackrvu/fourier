import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal
import threading
import queue
import time

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Real-time Audio Spectrogram")

# App title and description
st.title("Real-time Audio Spectrogram")
st.markdown("""
This application captures audio from your microphone and displays the spectrogram in real-time.
""")

# Audio parameters
SAMPLE_RATE = 44100  # Sample rate in Hz
BLOCK_SIZE = 1024    # Number of samples per block
WINDOW_SIZE = 2048   # FFT window size
OVERLAP = 512        # Overlap between windows
DURATION = 5         # Duration of spectrogram window in seconds

# Initialize audio data queue
audio_queue = queue.Queue()

# Function to get available audio devices
def get_audio_devices():
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
    return input_devices

# Get available input devices
input_devices = get_audio_devices()
device_names = [f"{name} (ID: {idx})" for idx, name in input_devices]

# UI for device selection
selected_device = st.selectbox("Select Audio Input Device:", device_names, index=0)
device_id = int(selected_device.split("(ID: ")[1].split(")")[0])

# Function to capture audio
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

# Create a figure for the spectrogram
fig, ax = plt.subplots(figsize=(12, 6))
spectrogram_placeholder = st.empty()

# Fixed settings - no UI controls
colormap = 'viridis'
freq_min = 0
freq_max = 5000
log_scale = True
normalize = True

# Buffer for spectrogram
spectrogram_buffer = []
max_frames = int(DURATION * SAMPLE_RATE / OVERLAP)

# Flag to track if stream is started
stream_started = False

def update_spectrogram():
    global spectrogram_buffer
    
    while True:
        if not stream_started:
            time.sleep(0.1)
            continue
            
        try:
            # Get audio data from queue
            audio_data = audio_queue.get(timeout=1)
            
            # Convert to mono if needed
            if audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            else:
                audio_data = audio_data.flatten()
            
            # Apply FFT
            # Use scipy.signal.spectrogram which handles windowing and overlapping
            f, t, Sxx = signal.spectrogram(
                audio_data, 
                fs=SAMPLE_RATE,
                window='hann',
                nperseg=WINDOW_SIZE,
                noverlap=OVERLAP,
                nfft=WINDOW_SIZE,
                detrend=False,
                scaling='spectrum'
            )
            
            # Convert to dB
            Sxx = 10 * np.log10(Sxx + 1e-10)
            
            # Append to buffer
            spectrogram_buffer.append(Sxx)
            
            # Limit buffer size
            if len(spectrogram_buffer) > max_frames:
                spectrogram_buffer = spectrogram_buffer[-max_frames:]
            
            # Only update plot every few frames to avoid excessive computation
            if audio_queue.qsize() < 2:
                # Clear the plot
                ax.clear()
                
                # Combine all frames in buffer
                if len(spectrogram_buffer) > 0:
                    combined_spec = np.hstack(spectrogram_buffer)
                    
                    # Frequency range selection
                    freq_indices = np.logical_and(f >= freq_min, f <= freq_max)
                    f_plot = f[freq_indices]
                    combined_spec_plot = combined_spec[freq_indices, :]
                    
                    # Normalize if selected
                    if normalize:
                        vmin = np.min(combined_spec_plot)
                        vmax = np.max(combined_spec_plot)
                    else:
                        vmin = -80
                        vmax = 0
                    
                    # Set y-axis scale
                    if log_scale:
                        # Avoid log(0)
                        f_plot = np.maximum(f_plot, 1)
                        ax.set_yscale('log')
                    
                    # Plot the spectrogram
                    img = ax.pcolormesh(
                        np.arange(combined_spec_plot.shape[1]) * OVERLAP / SAMPLE_RATE,
                        f_plot,
                        combined_spec_plot,
                        shading='gouraud',
                        cmap=colormap,
                        vmin=vmin,
                        vmax=vmax
                    )
                    
                    # Add colorbar
                    plt.colorbar(img, ax=ax, label='Power/Frequency (dB/Hz)')
                    
                    # Set labels and title
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_title('Real-time Audio Spectrogram')
                    
                    # Update the plot in Streamlit
                    spectrogram_placeholder.pyplot(fig)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in spectrogram update: {e}")

# Display info message while starting
start_message = st.info("Starting audio stream...")

# Start audio stream automatically
try:
    # Clear buffer
    spectrogram_buffer = []
    
    # Create a new stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=device_id,
        channels=1,
        callback=audio_callback
    )
    
    # Start stream
    stream.start()
    stream_started = True
    
    # Start the spectrogram update thread
    update_thread = threading.Thread(target=update_spectrogram)
    update_thread.daemon = True
    update_thread.start()
    
    # Update message
    start_message.success("Audio stream is active. Spectrogram should appear below.")
except Exception as e:
    start_message.error(f"Error starting audio stream: {e}")

# Display technical information
with st.expander("Technical Information"):
    st.markdown("""
    ### How it works:
    1. **Audio Capture**: Raw audio data is captured from your microphone in real-time using the SoundDevice library.
    2. **Fast Fourier Transform (FFT)**: The audio data is transformed from the time domain to the frequency domain using FFT.
    3. **Spectrogram Generation**: The frequency data over time is visualized as a spectrogram.
    
    ### Parameters:
    - **Sample Rate**: {0} Hz
    - **FFT Window Size**: {1} samples
    - **Window Function**: Hann window
    - **Overlap**: {2} samples
    - **Colormap**: Viridis
    - **Frequency Range**: 0-5000 Hz (logarithmic scale)
    """.format(SAMPLE_RATE, WINDOW_SIZE, OVERLAP))

# Handle app closing
def on_close():
    if stream_started:
        stream.stop()
        stream.close()

# Register cleanup handler
import atexit
atexit.register(on_close)