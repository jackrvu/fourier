import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# 1) Define a Parametric Curve
#    Here we use a "heart" shape, but feel free to replace x(t), y(t) with any shape you like.
# -----------------------
N = 500  # number of sample points
t_vals = np.linspace(0, 2*np.pi, N, endpoint=False)

# Heart shape (classic parametric form)
x_vals = 16 * np.sin(t_vals)**3
y_vals = 13 * np.cos(t_vals) - 5 * np.cos(2*t_vals) - 2 * np.cos(3*t_vals) - np.cos(4*t_vals)

# Combine into a single complex signal z(t) = x(t) + i*y(t)
z_vals = x_vals + 1j*y_vals

# -----------------------
# 2) Compute the Discrete Fourier Transform (DFT)
#    We'll use numpy's built-in FFT (which is actually a DFT for discrete signals).
# -----------------------
Z = np.fft.fft(z_vals)   # FFT of z(t)
# For convenience, we shift the zero-frequency component to the center.
# But for epicycles, we can simply keep the standard ordering.
# Alternatively, we could do: Z_shifted = np.fft.fftshift(Z)

# Frequencies:
# freq[k] corresponds to k in [0, N-1], but we typically interpret as
# negative frequencies after N/2. We'll keep the standard ordering for now.
N_half = N // 2  # used to handle negative/positive frequencies

# -----------------------
# 3) Prepare for Animation
# -----------------------
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ax.set_aspect("equal", "datalim")
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_title("Fourier Epicycle Drawing")

# Line objects for:
#   - The epicycle path as it is being drawn
#   - The final shape (optional, so you can compare)
epicycle_path, = ax.plot([], [], 'r-', lw=2)  # dynamic path in red
final_shape, = ax.plot([], [], 'k:', alpha=0.3)  # the original shape for reference (dotted line)

# We'll also keep track of center points for circles
circle_centers = []
circle_lines = []
max_freq_to_show = 50  # how many harmonics (positive + negative) to show in circles

# Create circle artists (we can store them for update)
for _ in range(2*max_freq_to_show + 1):
    center_plot, = ax.plot([], [], 'bo', ms=3)   # small dot for center
    circle_plot, = ax.plot([], [], 'b-', lw=1, alpha=0.5)  # circle perimeter
    circle_centers.append(center_plot)
    circle_lines.append(circle_plot)

# We'll store the drawn (x, y) path for each animation frame
drawn_x = []
drawn_y = []

# -----------------------
# 4) Epicycle Summation Function
# -----------------------
def epicycle_sum(t, Z, N, max_k):
    """
    Sum the first `max_k` positive and negative frequency components of Z
    at a normalized time t in [0,1).
    Returns the complex point z(t).
    """
    # The DFT element Z[k] corresponds to frequency k (0 <= k < N).
    # If k > N/2, that is effectively a negative frequency in the usual sense.
    # We'll handle indices carefully.

    # normalizing factor for the inverse DFT is 1/N
    z_t = 0+0j
    for k in range(-max_k, max_k+1):
        # Convert k in [-max_k, ..., 0, ..., max_k] to the appropriate index in Z
        # Python's FFT indexing: 
        #   freq index k in [0..N-1]
        #   actual freq = 2π*(k/N) in continuous sense
        # We'll define a helper function to wrap k into correct index:
        idx = k % N  
        # The time-domain factor e^{i*(2πk t)}:
        z_t += (1.0/N) * Z[idx] * np.exp(1j * 2 * np.pi * k * t)
    return z_t

# -----------------------
# 5) Animation Update Function
# -----------------------
def update(frame):
    """
    Update function for FuncAnimation.
    frame ranges from 0 to N-1, corresponding to discrete times t = frame/N
    """
    # Clear the dynamic path each time (we'll re-plot the entire path so far)
    global drawn_x, drawn_y
    
    # time as a fraction in [0, 1)
    t = frame / N
    
    # We'll build epicycles "from largest circle to smallest" or vice versa.
    # The order doesn't strictly matter, but it's nice to do from k=0 up or down.
    
    current_center = 0+0j
    # We'll store the centers so we can draw circles
    # Then we'll move to the next center for the next frequency.
    
    # Sort frequencies by absolute frequency (k=0, ±1, ±2, ...)
    freq_indices = range(-max_freq_to_show, max_freq_to_show + 1)
    
    # For plotting circles, we’ll keep track of the circle center and radius
    for i, k in enumerate(freq_indices):
        idx = k % N
        # amplitude = magnitude of (1/N)*Z[idx]
        amplitude = abs((1.0/N)*Z[idx])
        # angle at time t = arg(Z[idx]) + 2πk t
        phase = np.angle(Z[idx]) + 2*np.pi*k*t
        
        # The circle center is the current_center
        # Next point is where the vector points
        circle_centers[i].set_data([current_center.real], [current_center.imag])
        
        # Circle perimeter: param in [0..2π]
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = current_center.real + amplitude*np.cos(theta)
        circle_y = current_center.imag + amplitude*np.sin(theta)
        circle_lines[i].set_data(circle_x, circle_y)
        
        # The tip of this vector
        tip = current_center + amplitude * np.exp(1j*phase)
        
        # Update the center to the tip for the next frequency
        current_center = tip
    
    # 'current_center' is now the sum of all used frequencies at time t
    drawn_x.append(current_center.real)
    drawn_y.append(current_center.imag)
    
    epicycle_path.set_data(drawn_x, drawn_y)
    
    # Plot the original shape (static reference)
    final_shape.set_data(x_vals, y_vals)
    
    return [epicycle_path, final_shape] + circle_centers + circle_lines

# -----------------------
# 6) Create the Animation
# -----------------------
anim = FuncAnimation(fig, update, frames=N, interval=20, blit=True)

plt.show()
