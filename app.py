import os
# Set the SNDFILE_LIB path before importing soundfile.
os.environ["SNDFILE_LIB"] = "/usr/lib/x86_64-linux-gnu/libsndfile.so.1"

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import soundfile as sf
import hashlib
from datetime import datetime

# ------------------ Module 1: Data Preprocessing ------------------
def text_to_binary(text, encoding='utf-8'):
    """Convert text to its binary representation."""
    byte_array = text.encode(encoding)
    binary_str = ' '.join(format(byte, '08b') for byte in byte_array)
    return binary_str

def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a binary string back to text."""
    byte_list = binary_str.split()
    byte_array = bytearray(int(b, 2) for b in byte_list)
    text = byte_array.decode(encoding)
    return text

# ------------------ Module 2: Grouped Binary-to-Waveform (Plain Mapping) ------------------
def grouped_binary_to_waveform_plain(binary_str, sample_rate=44100, tone_duration=0.2, gap_duration=0.05,
                                     base_freq=300, freq_range=700):
    """
    Convert the binary string (grouped into 8-bit chunks) into an audio waveform 
    without chaotic modulation.
    """
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, '0')
    bytes_list = [binary_clean[i:i+8] for i in range(0, len(binary_clean), 8)]
    
    waveform = np.array([], dtype=np.float32)
    for byte_str in bytes_list:
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t_tone)
        gap_samples = int(sample_rate * gap_duration)
        gap = np.zeros(gap_samples, dtype=np.float32)
        waveform = np.concatenate((waveform, tone, gap))
        
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------ Module 3: Chaotic Function Integration using RK4 for Rossler Attractor ------------------
def rossler_derivatives(state, a, b, c):
    """Compute the derivatives for the Rossler attractor given state = [x, y, z]."""
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
    Generate a sequence of chaotic x-values using the Rossler attractor solved by RK4.
    The sequence is normalized to the range [0, 1].
    """
    state = np.array([x0, y0, z0], dtype=float)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------ Module 4: Grouped Binary-to-Waveform with Chaotic Modulation (Encryption) ------------------
def grouped_binary_to_waveform_chaotic(binary_str, sample_rate=44100, tone_duration=0.2, gap_duration=0.05,
                                       base_freq=300, freq_range=700, chaos_mod_range=100,
                                       dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Convert the binary string into an audio waveform with chaotic modulation.
    Each 8-bit chunk is mapped to a tone whose frequency is shifted by a chaotic offset.
    The chaotic sequence is generated using RK4 integration of the Rossler attractor.
    """
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, '0')
    bytes_list = [binary_clean[i:i+8] for i in range(0, len(binary_clean), 8)]
    
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(len(bytes_list), dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
    
    waveform = np.array([], dtype=np.float32)
    for i, byte_str in enumerate(bytes_list):
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        chaotic_offset = chaotic_sequence[i] * chaos_mod_range
        modulated_freq = freq + chaotic_offset
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * modulated_freq * t_tone)
        gap_samples = int(sample_rate * gap_duration)
        gap = np.zeros(gap_samples, dtype=np.float32)
        waveform = np.concatenate((waveform, tone, gap))
    
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------ Audio Synthesis and Storage Module ------------------
def synthesize_and_store_audio(waveform, sample_rate=44100, filename_prefix="oscilLOCK_audio", file_format="WAV"):
    """
    Convert a waveform (NumPy array) into an audio file and save it locally.
    Returns the full file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{file_format.lower()}"
    storage_dir = "audio_storage"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    full_path = os.path.join(storage_dir, filename)
    sf.write(full_path, waveform, sample_rate, format=file_format)
    return full_path

# ------------------ Visualization Functions ------------------
def create_waveform_figure(waveform, sample_rate, title="Waveform", zoom_range=None):
    """Return a Plotly figure of the waveform."""
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    fig = go.Figure(data=go.Scatter(x=time_vector, y=waveform, mode='lines', name='Waveform'))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude", hovermode='x')
    if zoom_range:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])
    return fig

def create_fft_figure(waveform_plain, waveform_chaotic, sample_rate):
    """Return a Plotly figure comparing the FFTs of the plain and chaotic waveforms."""
    N = len(waveform_plain)
    fft_plain = np.abs(np.fft.rfft(waveform_plain))
    fft_chaotic = np.abs(np.fft.rfft(waveform_chaotic))
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=fft_plain, mode='lines', name='Encoded Audio FFT'))
    fig.add_trace(go.Scatter(x=freqs, y=fft_chaotic, mode='lines', name='Encrypted Audio FFT'))
    fig.update_layout(title="FFT Comparison", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    fig.update_xaxes(range=[0, 1500])
    return fig

def create_difference_figure(waveform_plain, waveform_chaotic, sample_rate, zoom_range=(0, 0.005)):
    """Return a Plotly figure of the difference waveform (encrypted - encoded) zoomed in."""
    diff_waveform = waveform_chaotic - waveform_plain
    fig = create_waveform_figure(diff_waveform, sample_rate, title="Difference Waveform (Zoomed-In)", zoom_range=zoom_range)
    return fig

def create_chaotic_phase_plot(binary_str, dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Generate and return a phase plot (scatter plot of x[i+1] vs x[i]) of the chaotic sequence
    produced by the Rossler attractor using RK4 integration.
    """
    binary_clean = binary_str.replace(" ", "")
    n_bytes = len(binary_clean) // 8
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(n_bytes, dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
    x_vals = chaotic_sequence[:-1]
    y_vals = chaotic_sequence[1:]
    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=dict(color='red', size=8)))
    fig.update_layout(title="Chaotic Phase Plot", xaxis_title="x[i]", yaxis_title="x[i+1]")
    return fig

def convert_waveform_to_audio_bytes(waveform, sample_rate, file_format="WAV"):
    """Convert a numpy waveform to audio bytes for playback in the specified format."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format=file_format)
    return buf.getvalue()

# ------------------ Additional Visualization Functions for Key Generation ------------------
def get_audio_feature_values(waveform, num_samples=128):
    """
    Extract and quantize amplitude samples from the audio waveform.
    Returns a numpy array of 8-bit quantized values.
    """
    indices = np.linspace(0, len(waveform)-1, num_samples, dtype=int)
    samples = waveform[indices]
    quantized = np.uint8(255 * ((samples - samples.min()) / (samples.max() - samples.min() + 1e-6)))
    return quantized

def plot_chaotic_sequence(chaotic_sequence):
    """Plot the raw chaotic sequence as a line graph."""
    fig = go.Figure(data=go.Scatter(x=list(range(len(chaotic_sequence))), y=chaotic_sequence, mode='lines', name='Chaotic Sequence'))
    fig.update_layout(title="Raw Chaotic Sequence", xaxis_title="Index", yaxis_title="Value")
    return fig

def plot_quantized_chaotic_sequence(chaotic_sequence):
    """Plot the quantized chaotic sequence (8-bit values)."""
    chaotic_array = np.array(chaotic_sequence)
    quantized = np.uint8(255 * chaotic_array)
    fig = go.Figure(data=go.Scatter(x=list(range(len(quantized))), y=quantized, mode='lines+markers', name='Quantized Chaotic Sequence'))
    fig.update_layout(title="Quantized Chaotic Sequence", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

def plot_audio_features(audio_features):
    """Plot the quantized audio feature values."""
    fig = go.Figure(data=go.Scatter(x=list(range(len(audio_features))), y=audio_features, mode='lines+markers', name='Audio Features'))
    fig.update_layout(title="Audio Features (Quantized)", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

def plot_combined_features(quantized_chaotic, audio_features):
    """Display quantized chaotic sequence and audio features side by side."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Quantized Chaotic Sequence", "Audio Features"))
    fig.add_trace(go.Scatter(x=list(range(len(quantized_chaotic))), y=quantized_chaotic, mode='lines+markers', name='Quantized Chaotic'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(audio_features))), y=audio_features, mode='lines+markers', name='Audio Features'), row=1, col=2)
    fig.update_layout(title="Combined Features Comparison")
    return fig

# ------------------ Key Generation Functions ------------------
def derive_initial_conditions(passphrase):
    """
    Derive initial conditions for the chaotic system from the passphrase.
    Hash the passphrase with SHA-256 and split the hash into three parts for x0, y0, and z0.
    """
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()  # 64 hex characters
    x0 = int(hash_digest[0:21], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    y0 = int(hash_digest[21:42], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    z0 = int(hash_digest[42:64], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    return x0, y0, z0

def sample_audio_features(waveform, num_samples=128):
    """
    Extract a set number of amplitude samples from the audio waveform,
    quantize them to 8 bits, and return as a byte string.
    """
    indices = np.linspace(0, len(waveform)-1, num_samples, dtype=int)
    samples = waveform[indices]
    quantized = np.uint8(255 * ((samples - samples.min()) / (samples.max()-samples.min() + 1e-6)))
    return quantized.tobytes()

def generate_chaotic_key(passphrase, waveform, chaotic_params, num_chaotic_samples=128):
    """
    Generate a cryptographic key using both a passphrase and audio features.
    
    Steps:
      1. Derive initial conditions from the passphrase.
      2. Generate a chaotic sequence using RK4 with these conditions.
      3. Quantize the chaotic sequence to 8-bit values.
      4. Extract audio features from the waveform.
      5. Concatenate the two byte strings and hash them to produce a key.
    """
    x0, y0, z0 = derive_initial_conditions(passphrase)
    dt, a, b, c = chaotic_params
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(num_chaotic_samples, dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
    chaotic_array = np.array(chaotic_sequence)
    chaotic_quantized = np.uint8(255 * chaotic_array)
    chaotic_bytes = chaotic_quantized.tobytes()
    audio_bytes = sample_audio_features(waveform, num_samples=num_chaotic_samples)
    combined = chaotic_bytes + audio_bytes
    key = hashlib.sha256(combined).hexdigest()
    return key, chaotic_sequence

# ------------------ Streamlit Interface ------------------
def main():
    st.set_page_config(page_title="oscilLOCK", layout="wide")
    
    # Sidebar for input & customization
    st.sidebar.title("CONTROL PANEL")
    with st.sidebar.form(key="input_form"):
        user_text = st.text_input("Enter text to encrypt:", "Hello, oscilLOCK!", max_chars=500)
        passphrase = st.text_input("Enter passphrase for key generation:", "DefaultPassphrase", type="password")
        
        st.markdown("### Audio Parameters")
        tone_duration = st.slider("Tone Duration (sec)", 0.1, 0.5, 0.2)
        gap_duration = st.slider("Gap Duration (sec)", 0.01, 0.1, 0.05)
        base_freq = st.number_input("Base Frequency (Hz)", 100, 1000, 300)
        freq_range = st.number_input("Frequency Range (Hz)", 100, 2000, 700)
        chaos_mod_range = st.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
        
        st.markdown("### Chaotic Key Generation")
        num_chaotic_samples = st.slider("Number of Chaotic Samples", 64, 1024, 128, step=64)
        
        with st.expander("Advanced Chaotic Parameters"):
            dt = st.slider("dt", 0.001, 0.05, 0.01, step=0.001)
            a = st.slider("a", 0.1, 1.0, 0.2, step=0.1)
            b = st.slider("b", 0.1, 1.0, 0.2, step=0.1)
            c = st.slider("c", 1.0, 10.0, 5.7, step=0.1)
            # Initial conditions (x0, y0, z0) are derived solely from the passphrase.
        
        submit_button = st.form_submit_button(label="Enter")
    
    if submit_button and user_text:
        with st.spinner("Processing..."):
            # Module 1: Data Preprocessing
            binary_output = text_to_binary(user_text)
            recovered_text = binary_to_text(binary_output)
            sample_rate = 44100
            
            # Module 2: Generate Encoded Audio Waveform (Plain Mapping)
            waveform_encoded, _ = grouped_binary_to_waveform_plain(
                binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
                base_freq=base_freq, freq_range=freq_range
            )
            
            # Derive initial conditions from passphrase for chaotic integration
            derived_x0, derived_y0, derived_z0 = derive_initial_conditions(passphrase)
            
            # Module 3/4: Generate Encrypted Audio Waveform with Chaotic Modulation
            waveform_encrypted, _ = grouped_binary_to_waveform_chaotic(
                binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
                base_freq=base_freq, freq_range=freq_range, chaos_mod_range=chaos_mod_range,
                dt=dt, a=a, b=b, c=c, x0=derived_x0, y0=derived_y0, z0=derived_z0
            )
            
            # Key Generation: Use chaotic_params (dt, a, b, c) and user-selected num_chaotic_samples
            chaotic_params = (dt, a, b, c)
            derived_key, chaotic_sequence = generate_chaotic_key(passphrase, waveform_encoded, chaotic_params, num_chaotic_samples)
            
            # Extract audio features for key generation visualization
            audio_features = get_audio_feature_values(waveform_encoded, num_samples=num_chaotic_samples)
            
            # Prepare encrypted audio bytes for download (format will be selected in Storage tab)
        
        # Create tabs for the pipeline (5 tabs)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Preprocessing & Encoding", 
            "Encryption Module", 
            "Comparison", 
            "Key Generation", 
            "Storage"
        ])
        
        with tab1:
            st.header("Preprocessing & Encoding")
            st.markdown("**Binary Representation:**")
            st.code(binary_output)
            st.markdown(f"**Recovered Text:** {recovered_text}")
            st.markdown("**Encoded Audio:**")
            audio_encoded = convert_waveform_to_audio_bytes(waveform_encoded, sample_rate, file_format="WAV")
            st.audio(audio_encoded, format='audio/wav', start_time=0)
        
        with tab2:
            st.header("Encryption Module")
            st.markdown("**Encrypted Audio:**")
            encrypted_audio_bytes = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate, file_format="WAV")
            st.audio(encrypted_audio_bytes, format='audio/wav', start_time=0)
        
        with tab3:
            st.header("Comparison")
            zoom_range = (0, 0.005)
            st.subheader("Difference Waveform (Encrypted - Encoded)")
            fig_diff = create_difference_figure(waveform_encoded, waveform_encrypted, sample_rate, zoom_range=zoom_range)
            st.plotly_chart(fig_diff, use_container_width=True)
            
            st.subheader("FFT Comparison")
            fig_fft = create_fft_figure(waveform_encoded, waveform_encrypted, sample_rate)
            st.plotly_chart(fig_fft, use_container_width=True)
            
            st.subheader("Chaotic Dynamics Visualization")
            fig_phase = create_chaotic_phase_plot(binary_output, dt=dt, a=a, b=b, c=c, x0=derived_x0, y0=derived_y0, z0=derived_z0)
            st.plotly_chart(fig_phase, use_container_width=True)
        
        with tab4:
            st.header("Key Generation")
            st.markdown("**Derived Cryptographic Key:**")
            st.code(derived_key)
            st.markdown("### Visualizing the Key Generation Process")
            fig_chaotic_raw = plot_chaotic_sequence(chaotic_sequence)
            st.plotly_chart(fig_chaotic_raw, use_container_width=True)
            fig_chaotic_quant = plot_quantized_chaotic_sequence(chaotic_sequence)
            st.plotly_chart(fig_chaotic_quant, use_container_width=True)
            fig_audio_features = plot_audio_features(audio_features)
            st.plotly_chart(fig_audio_features, use_container_width=True)
            fig_combined = plot_combined_features(np.uint8(255 * np.array(chaotic_sequence)), audio_features)
            st.plotly_chart(fig_combined, use_container_width=True)
            st.markdown("The key is generated by concatenating the quantized chaotic sequence (derived from your passphrase) with audio features extracted from the encoded waveform, then hashing the result with SHA‑256.")
        
        with tab5:
            st.header("Storage")
            st.markdown("Select a format to download the encrypted audio file:")
            file_format = st.selectbox("Download Format", options=["WAV", "FLAC", "OGG"], index=0)
            if file_format.upper() == "WAV":
                mime_type = "audio/wav"
            elif file_format.upper() == "FLAC":
                mime_type = "audio/flac"
            elif file_format.upper() == "OGG":
                mime_type = "audio/ogg"
            else:
                mime_type = "audio/wav"
            download_audio = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate, file_format=file_format.upper())
            st.download_button(label=f"Download Encrypted Audio ({file_format.upper()})",
                               data=download_audio,
                               file_name=f"encrypted_audio.{file_format.lower()}",
                               mime=mime_type)

if __name__ == "__main__":
    main()
