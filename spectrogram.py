import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import struct
import time

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate (Hz)
CHUNK = 1024  # Number of samples per buffer
WINDOW = np.hamming(CHUNK)  # Window function to reduce spectral leakage

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

# Set up the figure and axis for the spectrogram
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Configure the plot
freqs = np.fft.rfftfreq(CHUNK, 1/RATE)  # Frequency bins
time_bins = np.arange(0, 100)  # Number of time bins to display
spectrogram_data = np.zeros((len(freqs), len(time_bins)))  # Initialize spectrogram data

# Plot the spectrogram with a color map
spectrogram = ax.imshow(
    spectrogram_data,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    extent=[0, len(time_bins), 0, RATE/2],  # [xmin, xmax, ymin, ymax]
    vmin=0,
    vmax=5  # Adjust this for better color scaling
)

# Add a colorbar
cbar = fig.colorbar(spectrogram)
cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Real-time Audio Spectrogram')
ax.set_yscale('log')  # Logarithmic scale for frequency
ax.set_ylim(20, RATE/2)  # Set lower limit to 20Hz (human hearing threshold)

# Function to update the plot with new audio data
def update_plot(frame):
    global spectrogram_data
    
    # Read audio data
    audio_data = stream.read(CHUNK, exception_on_overflow=False)
    
    # Convert audio data to numpy array
    count = len(audio_data) // 2
    format = f"{count}h"
    audio_data = np.array(struct.unpack(format, audio_data)) / 32768.0  # Normalize to [-1.0, 1.0]
    
    # Apply window function
    windowed_data = audio_data * WINDOW
    
    # Compute FFT and get magnitudes
    fft_data = np.fft.rfft(windowed_data)
    magnitude = np.abs(fft_data)
    
    # Convert to dB scale (log scale for better visualization)
    magnitude = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
    
    # Roll the spectrogram data array and update the last column
    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    spectrogram_data[:, -1] = magnitude
    
    # Update the plot
    spectrogram.set_array(spectrogram_data)
    
    return spectrogram,

# Create animation
ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

# Display the plot
plt.tight_layout()
plt.show()

# Clean up
def cleanup():
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio resources cleaned up.")

# Set up cleanup on exit
try:
    plt.show()
finally:
    cleanup()