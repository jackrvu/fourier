import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
import time

st.title("Real-time Audio Spectrogram")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100         # Sample rate (Hz)
CHUNK = 1024         # Number of samples per buffer
WINDOW = np.hamming(CHUNK)  # Hamming window to reduce spectral leakage

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=CHUNK
)

# Set up the matplotlib figure and axis for the spectrogram
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Configure the plot
freqs = np.fft.rfftfreq(CHUNK, 1/RATE)  # Frequency bins
time_bins = np.arange(0, 100)             # Number of time bins to display
spectrogram_data = np.zeros((len(freqs), len(time_bins)))  # Initialize spectrogram data

spectrogram = ax.imshow(
    spectrogram_data,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    extent=[0, len(time_bins), 0, RATE/2],  # [xmin, xmax, ymin, ymax]
    vmin=0,
    vmax=5  # Adjust for better color scaling if needed
)

cbar = fig.colorbar(spectrogram)
cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)

ax.set_xlabel('Time')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Real-time Audio Spectrogram')
ax.set_yscale('log')      # Logarithmic scale for frequency
ax.set_ylim(20, RATE/2)     # Lower limit set to 20Hz (approximate human hearing threshold)

# Create a Streamlit placeholder for the plot
plot_placeholder = st.empty()

# Main update loop – runs for a large number of iterations as an example.
# To stop the app, simply click the "Stop" button in the Streamlit interface or interrupt the app.
try:
    for _ in range(10000):
        # Read audio data
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        count = len(audio_data) // 2
        fmt = f"{count}h"
        audio_data = np.array(struct.unpack(fmt, audio_data)) / 32768.0  # Normalize to [-1, 1]
        
        # Apply window function and compute FFT
        windowed_data = audio_data * WINDOW
        fft_data = np.fft.rfft(windowed_data)
        magnitude = np.abs(fft_data)
        
        # Convert magnitude to dB scale
        magnitude = 20 * np.log10(magnitude + 1e-10)  # Avoid log(0)
        
        # Roll the spectrogram data array and update the last column with the new data
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = magnitude
        
        # Update the image data in the plot
        spectrogram.set_array(spectrogram_data)
        
        # Update the Streamlit placeholder with the new figure
        plot_placeholder.pyplot(fig)
        
        # Small pause to control update rate
        time.sleep(0.03)

except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    # Clean up audio resources when done
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.write("Audio resources cleaned up.")
