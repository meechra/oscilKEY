import streamlit as st
import numpy as np
import plotly.graph_objects as go
import hashlib
import io
import os
from datetime import datetime
import scipy.io.wavfile as wav  # Using SciPy as a replacement for PySoundFile

# ------------------ Utility Functions ------------------

def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a space-separated binary string back to text."""
    byte_list = binary_str.split()
    try:
        byte_array = bytearray(int(b, 2) for b in byte_list)
        text = byte_array.decode(encoding)
    except Exception as e:
        text = "Decoding error: " + str(e)
    return text

def derive_initial_conditions(passphrase):
    """
    Derive initial conditions for the chaotic system from the passphrase.
    Uses SHA‑256 to generate high entropy initial conditions.
    """
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()  # 64 hex characters
    x0 = int(hash_digest[0:21], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    y0 = int(hash_digest[21:42], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    z0 = int(hash_digest[42:64], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    return x0, y0, z0

# ------------------ Chaotic System Functions (Rossler) ------------------

def rossler_derivatives(state, a, b, c):
    """Compute the derivatives for the Rossler attractor."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(state, dt, a, b, c):
    """Perform a single RK4 integration step for the Rossler system."""
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence_rossler_rk4(n, dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Generate a sequence of chaotic x-values using the Rossler attractor via RK4.
    The sequence is normalized to [0, 1].
    """
    state = np.array([x0, y0, z0], dtype=float)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------ Decryption Function ------------------

def decrypt_waveform_to_binary(waveform, sample_rate, tone_duration, gap_duration,
                               base_freq, freq_range, chaos_mod_range,
                               dt, a, b, c, passphrase):
    """
    Decrypt the provided audio waveform (encrypted via oscilLOCK) by:
      1. Segmenting the waveform into tone portions (based on tone/gap durations).
      2. Estimating the tone frequency (via FFT) for each segment.
      3. Reconstructing the chaotic sequence from the passphrase and parameters.
      4. Removing the chaotic modulation and inverting the mapping to recover bytes.
      
    Returns a space-separated binary string.
    """
    tone_samples = int(sample_rate * tone_duration)
    gap_samples = int(sample_rate * gap_duration)
    segment_length = tone_samples + gap_samples
    total_samples = len(waveform)
    n_segments = total_samples // segment_length

    # Derive chaotic initial conditions from the passphrase and regenerate chaotic sequence
    x0, y0, z0 = derive_initial_conditions(passphrase)
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(n_segments, dt=dt, a=a, b=b, c=c,
                                                             x0=x0, y0=y0, z0=z0)
    
    binary_list = []
    for i in range(n_segments):
        start = i * segment_length
        end = start + tone_samples
        tone_segment = waveform[start:end]
        
        # Estimate frequency using FFT with a Hann window for improved accuracy.
        N_tone = len(tone_segment)
        window = np.hanning(N_tone)
        fft_result = np.fft.rfft(tone_segment * window)
        fft_magnitude = np.abs(fft_result)
        peak_index = np.argmax(fft_magnitude)
        freq_resolution = sample_rate / N_tone
        observed_freq = peak_index * freq_resolution
        
        # Remove the chaotic offset
        chaotic_offset = chaotic_sequence[i] * chaos_mod_range
        plain_freq = observed_freq - chaotic_offset
        
        # Invert the mapping:
        #   plain_freq = base_freq + (byte_value/255)*freq_range
        byte_value = (plain_freq - base_freq) / freq_range * 255
        byte_value = int(np.rint(byte_value))
        byte_value = max(0, min(255, byte_value))
        
        # Convert byte to an 8-bit binary string.
        byte_binary = format(byte_value, '08b')
        binary_list.append(byte_binary)
    
    binary_string = " ".join(binary_list)
    return binary_string

# ------------------ Visualization ------------------

def create_waveform_figure(waveform, sample_rate, title="Waveform"):
    """Generate a Plotly figure of the audio waveform."""
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    fig = go.Figure(data=go.Scatter(x=time_vector, y=waveform, mode='lines'))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig

# ------------------ Streamlit Interface for oscilKEY ------------------

def main():
    st.set_page_config(page_title="oscilKEY - Decryption", layout="wide")
    st.title("oscilKEY: Audio Waveform Decryption")
    st.markdown("""
    **oscilKEY** decrypts an encrypted WAV audio waveform (produced by oscilLOCK) to recover the original text.
    
    Provide the encrypted WAV file along with the same encryption parameters and passphrase.
    """)
    
    st.sidebar.header("Decryption Settings")
    
    # File upload: Encrypted audio file (WAV only)
    uploaded_file = st.sidebar.file_uploader("Upload Encrypted Audio (WAV)", type=['wav'])
    passphrase = st.sidebar.text_input("Enter Passphrase:", type="password", value="DefaultPassphrase")
    
    st.sidebar.markdown("### Audio Parameters")
    tone_duration = st.sidebar.slider("Tone Duration (sec)", 0.1, 0.5, 0.2)
    gap_duration = st.sidebar.slider("Gap Duration (sec)", 0.01, 0.1, 0.05)
    base_freq = st.sidebar.number_input("Base Frequency (Hz)", 100, 1000, 300)
    freq_range = st.sidebar.number_input("Frequency Range (Hz)", 100, 2000, 700)
    chaos_mod_range = st.sidebar.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
    
    st.sidebar.markdown("### Chaotic Parameters")
    dt = st.sidebar.slider("dt", 0.001, 0.05, 0.01, step=0.001)
    a = st.sidebar.slider("a", 0.1, 1.0, 0.2, step=0.1)
    b = st.sidebar.slider("b", 0.1, 1.0, 0.2, step=0.1)
    c = st.sidebar.slider("c", 1.0, 10.0, 5.7, step=0.1)
    
    sample_rate = st.sidebar.number_input("Sample Rate (Hz)", 8000, 96000, 44100)
    
    if uploaded_file and passphrase:
        try:
            # Read the uploaded WAV file using SciPy
            sample_rate_file, waveform = wav.read(uploaded_file)
            # Check sample rate consistency
            if sample_rate_file != sample_rate:
                st.warning(f"File sample rate ({sample_rate_file} Hz) differs from selected rate ({sample_rate} Hz). Using file's sample rate.")
                sample_rate = sample_rate_file
            # Convert to float in case the file is in integer format (assuming 16-bit PCM)
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32767.0
        except Exception as e:
            st.error(f"Error reading audio file: {e}")
            return
        
        st.subheader("Uploaded Encrypted Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        # Display the encrypted waveform
        fig_wave = create_waveform_figure(waveform, sample_rate, title="Encrypted Audio Waveform")
        st.plotly_chart(fig_wave, use_container_width=True)
        
        if st.button("Decrypt Audio"):
            with st.spinner("Decrypting..."):
                binary_output = decrypt_waveform_to_binary(
                    waveform, sample_rate, tone_duration, gap_duration,
                    base_freq, freq_range, chaos_mod_range,
                    dt, a, b, c, passphrase
                )
                recovered_text = binary_to_text(binary_output)
            
            st.subheader("Decryption Output")
            st.markdown("**Recovered Binary:**")
            st.code(binary_output)
            st.markdown("**Recovered Text:**")
            st.write(recovered_text)

if __name__ == "__main__":
    main()
