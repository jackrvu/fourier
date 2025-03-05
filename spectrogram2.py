import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sounddevice as sd
from scipy import signal
import queue
import time

# Configure page
st.set_page_config(layout="wide", page_title="Real-time Audio Spectrogram")
st.title("Real-time Audio Spectrogram")

# Audio parameters
SAMPLE_RATE = 44100  # Sample rate in Hz
BLOCK_SIZE = 1024    # Number of samples per block
WINDOW_SIZE = 2048   # FFT window size
OVERLAP = 512        # Overlap between windows
DURATION = 5         # Duration of display window in seconds

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

# Select first device by default
selected_device = st.selectbox("Select Audio Input Device:", device_names, index=0)
device_id = int(selected_device.split("(ID: ")[1].split(")")[0])

# Create a status message
status_message = st.empty()

# Create a placeholder for the spectrogram
spectrogram_placeholder = st.empty()

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        status_message.warning(f"Audio status: {status}")
    audio_queue.put(indata.copy())

# Buffer for recent audio data
buffer = np.zeros(DURATION * SAMPLE_RATE)
buffer_idx = 0

# Create spectrogram plot with no duplicate colorbars
def create_spectrogram(data):
    # Calculate spectrogram
    f, t, Sxx = signal.spectrogram(
        data,
        fs=SAMPLE_RATE,
        window='hann',
        nperseg=WINDOW_SIZE,
        noverlap=OVERLAP,
        nfft=WINDOW_SIZE,
        detrend=False,
        scaling='spectrum'
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Select frequency range (0-5000 Hz)
    freq_mask = (f >= 0) & (f <= 5000)
    f_plot = f[freq_mask]
    Sxx_plot = Sxx_db[freq_mask, :]
    
    # Create a new figure each time but with no duplicate elements
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Create the mesh
    mesh = ax.pcolormesh(t, f_plot, Sxx_plot, shading='nearest', cmap='viridis')
    
    # Set colorbar range
    vmin = max(-110, np.min(Sxx_plot))
    vmax = min(-90, np.max(Sxx_plot))
    if vmax - vmin < 5:
        vmin = vmax - 20  # Ensure minimum contrast
    mesh.set_clim(vmin, vmax)
    
    # Create a colorbar
    cbar = fig.colorbar(mesh)
    cbar.set_label('Power/Frequency (dB/Hz)')
    
    # Configure axes
    ax.set_yscale('log')
    ax.set_ylim(20, 5000)  # Start from 20Hz to avoid log(0) issues
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Real-time Audio Spectrogram')
    
    # Ensure proper layout
    fig.tight_layout()
    
    return fig

# Start audio stream
status_message.info("Starting audio stream...")
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=device_id,
        channels=1,
        callback=audio_callback
    )
    stream.start()
    status_message.success("Audio stream started. Spectrogram will update automatically.")
    
    # Main loop
    last_update_time = time.time()
    update_interval = 0.1  # Update every 100ms
    
    while True:
        try:
            # Get audio data from queue
            audio_data = audio_queue.get(timeout=0.1)
            
            # Convert to mono if needed
            if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            else:
                audio_data = audio_data.flatten()
            
            # Add to circular buffer
            n_samples = len(audio_data)
            if buffer_idx + n_samples <= len(buffer):
                buffer[buffer_idx:buffer_idx+n_samples] = audio_data
                buffer_idx += n_samples
            else:
                # Wrap around
                first_part = len(buffer) - buffer_idx
                buffer[buffer_idx:] = audio_data[:first_part]
                buffer[:n_samples-first_part] = audio_data[first_part:]
                buffer_idx = n_samples - first_part
            
            # Only update visualization periodically to reduce CPU usage
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                # Create and display the spectrogram
                fig = create_spectrogram(buffer)
                spectrogram_placeholder.pyplot(fig)
                last_update_time = current_time
                
        except queue.Empty:
            pass
            
        except Exception as e:
            status_message.error(f"Error: {e}")
            time.sleep(0.5)
            
except Exception as e:
    status_message.error(f"Error starting audio stream: {e}")
finally:
    if 'stream' in locals() and stream.active:
        stream.stop()
        stream.close()

# Display technical information
with st.expander("Technical Information"):
    st.markdown("""
    ### How it works:
    1. **Audio Capture**: Raw audio data is captured from your microphone in real-time.
    2. **Fast Fourier Transform (FFT)**: The audio data is transformed from the time domain to the frequency domain.
    3. **Spectrogram Generation**: The frequency data over time is visualized as a heatmap.
    
    ### Parameters:
    - **Sample Rate**: {0} Hz
    - **FFT Window Size**: {1} samples
    - **Window Function**: Hann window
    - **Overlap**: {2} samples
    - **Colormap**: Viridis
    - **Frequency Range**: 0-5000 Hz (logarithmic scale)
    """.format(SAMPLE_RATE, WINDOW_SIZE, OVERLAP))