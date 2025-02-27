import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# 1) Define the PDE(t) shape
# ----------------------------
def PDE(t):
    """
    Returns a complex number x + i*y corresponding to a point
    in the plane that sketches out 'P', 'D', and 'E' as t goes
    from 0 to 3.
    """
    # We'll break it into intervals: 
    #  P : t in [0,1)
    #  D : t in [1,2)
    #  E : t in [2,3)
    
    # To keep things simple, we won't wrap t around 
    # (i.e., we assume 0 <= t < 3).
    
    # P: vertical stroke [0, 0.5], loop [0.5, 1.0]
    if 0 <= t < 1.0:
        u = t  # local parameter in [0,1)
        if u < 0.5:
            # Vertical stroke: from (0,0) to (0,2)
            # u=0 -> (0,0), u=0.5 -> (0,2)
            x = 0
            y = 4 * u
        else:
            # Loop on top part
            # Break it again into two segments:
            #   upper arc:  [0.5 -> 0.75]
            #   lower arc:  [0.75 -> 1.0]
            u2 = (u - 0.5) / 0.5  # Now in [0,1]
            if u2 < 0.5:
                # upper arc: from (0,2) to roughly (1,1.5)
                v = u2 / 0.5  # in [0,1]
                x = 2 * u2       # 0 -> 1 as u2 goes 0->0.5
                y = 2 - u2       # 2 -> 1.5
            else:
                # lower arc: from (1,1.5) to (1,1)
                v = (u2 - 0.5) / 0.5  # in [0,1]
                x = 1.0 + (1 - v)*0   # stays 1
                y = 1.5 - (1 - v)     # 1.5 -> 1
        return x + 1j*y

    # D: vertical stroke [1,1.5], half-circle [1.5,2]
    elif 1.0 <= t < 2.0:
        u = t - 1.0  # local parameter in [0,1)
        if u < 0.5:
            # Vertical stroke from (1.5,0) to (1.5,2)
            x = 1.5
            y = 4 * u  # 0 -> 2
        else:
            # Half-circle from (1.5,2) back to (1.5,0)
            # Let angle go from pi/2 down to -pi/2
            u2 = (u - 0.5) / 0.5  # in [0,1]
            theta = np.pi/2 - np.pi * u2  # pi/2 -> -pi/2
            # Center at (1.5,1), radius=1
            x = 1.5 + np.cos(theta)
            y = 1.0 + np.sin(theta)
        return x + 1j*y

    # E: vertical stroke [2,2.5], top bar [2.5,2.625],
    #    middle bar [2.625,2.75], bottom bar [2.75,3.0]
    else:  # 2.0 <= t < 3.0
        u = t - 2.0  # local parameter in [0,1)
        if u < 0.5:
            # Vertical stroke from (3,0) to (3,2)
            x = 3.0
            y = 4 * u  # 0 -> 2
        elif u < 0.625:
            # Top horizontal bar from (3,2) to (4,2)
            u2 = (u - 0.5) / 0.125  # in [0,1]
            x = 3.0 + u2
            y = 2.0
        elif u < 0.75:
            # Middle bar from (3,1) to (3.5,1)
            u2 = (u - 0.625) / 0.125  # in [0,1]
            x = 3.0 + 0.5*u2
            y = 1.0
        else:
            # Bottom bar from (3,0) to (4,0)
            u2 = (u - 0.75) / 0.25  # in [0,1]
            x = 3.0 + u2
            y = 0.0
        return x + 1j*y

# ---------------------------------
# 2) Sample the PDE function
# ---------------------------------
N = 500  # Number of sample points
t_vals = np.linspace(0, 3, N, endpoint=False)  # from 0 to 3
f_vals = np.array([PDE(t) for t in t_vals], dtype=complex)

# ------------------------------------------------
# 3) Compute the Discrete Fourier Transform (DFT)
#    (Using NumPy's FFT with normalization)
# ------------------------------------------------
c = np.fft.fft(f_vals) / N  # Array of complex Fourier coefficients
# For epicycles, we'll use c[k] for k in [0..N-1].
# Frequencies are effectively k cycles per full rotation in our animation.

# --------------------------------------------
# 4) Set up the epicycle animation in Matplotlib
# --------------------------------------------

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlim(-1, 5)   # Adjust as needed to see all letters
ax.set_ylim(-1, 3.5)

# We'll draw:
#  - The chain of epicycles (circles) showing each frequency
#  - A trace (dot) for the final sum that should outline PDE

# A small circle to draw each epicycle
max_freq_to_show = 50  # Show the first +/-some frequencies (optional)
# If you want all frequencies, set max_freq_to_show = N//2 or so.
# For simplicity here, we'll just do the positive side frequencies up to max_freq_to_show.

circles = []
lines = []
for k in range(max_freq_to_show):
    circle, = ax.plot([], [], 'gray', lw=1)  # circle outline
    circles.append(circle)
    line, = ax.plot([], [], 'r-', lw=1)      # radius line
    lines.append(line)

# We'll also place a dot at the final tip
tip_dot, = ax.plot([], [], 'bo', markersize=4)
path_line, = ax.plot([], [], 'b--', lw=1, alpha=0.5)  # Optional: path traced so far

path_x = []
path_y = []

def init():
    for circle in circles:
        circle.set_data([], [])
    for line in lines:
        line.set_data([], [])
    tip_dot.set_data([], [])
    path_line.set_data([], [])
    return circles + lines + [tip_dot, path_line]

def animate(frame):
    """
    For frame in [0..N-1], compute the partial sum and
    update each epicycle circle and line.
    """
    # We'll interpret 'frame' as the time index:
    # angle_k = 2*pi*k*(frame/N).
    # Summation: sum_{k=0..N-1} c[k] * exp(i*angle_k)
    
    # But we'll only *draw* the first few frequencies for clarity (max_freq_to_show),
    # though the tip is computed from all frequencies.
    
    # Center of the current epicycle chain
    current_center = 0+0j
    for k in range(max_freq_to_show):
        angle_k = 2.0 * np.pi * k * (frame / N)
        # Next vector
        vector_k = c[k] * np.exp(1j * angle_k)
        
        # The circle to draw has center = current_center and radius = |c[k]|
        # We'll parameterize the circle in 100 points for plotting
        radius = np.abs(c[k])
        th = np.linspace(0, 2*np.pi, 100)
        circle_x = current_center.real + radius*np.cos(th)
        circle_y = current_center.imag + radius*np.sin(th)
        
        circles[k].set_data(circle_x, circle_y)
        
        # The line from center to the tip of this epicycle
        line_x = [current_center.real, (current_center + vector_k).real]
        line_y = [current_center.imag, (current_center + vector_k).imag]
        lines[k].set_data(line_x, line_y)
        
        # Update the center for the next frequency
        current_center += vector_k
    
    # Now compute the *full* tip using all frequencies (for best accuracy).
    # If we only used up to max_freq_to_show, the shape might look incomplete.
    # Summation for all k in [0..N-1].
    full_sum = 0+0j
    for k in range(N):
        angle_k = 2.0 * np.pi * k * (frame / N)
        full_sum += c[k] * np.exp(1j * angle_k)
    
    tip_dot.set_data([full_sum.real], [full_sum.imag])

    # Store the path for trailing
    path_x.append(full_sum.real)
    path_y.append(full_sum.imag)
    path_line.set_data(path_x, path_y)
    
    return circles + lines + [tip_dot, path_line]

anim = FuncAnimation(
    fig, animate, frames=N, 
    init_func=init, blit=True, interval=20, repeat=True
)

plt.title("Fourier Epicycles Spelling 'PDE'")
plt.show()
