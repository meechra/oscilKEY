import streamlit as st
import numpy as np
import soundfile as sf
import io
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------ Common Functions (from Encryption code) ------------------

def derive_initial_conditions(passphrase):
    """
    Derive initial conditions from the passphrase.
    """
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()
    x0 = int(hash_digest[0:21], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    y0 = int(hash_digest[21:42], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    z0 = int(hash_digest[42:64], 16) / float(0xFFFFFFFFFFFFFFFFFFFFF)
    return x0, y0, z0

def rossler_derivatives(state, a, b, c):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(state, dt, a, b, c):
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence_rossler_rk4(n, dt, a, b, c, x0, y0, z0):
    """
    Generate a normalized chaotic sequence (of x-values) using RK4 integration.
    """
    state = np.array([x0, y0, z0], dtype=float)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------ Frequency Estimation ------------------
def estimate_frequency(segment, sample_rate):
    """
    Estimate the dominant frequency of the given audio segment via FFT peak detection.
    Returns the frequency in Hz.
    """
    # Apply a Hamming window
    window = np.hamming(len(segment))
    segment_win = segment * window
    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(segment_win))
    freqs = np.fft.rfftfreq(len(segment), d=1/sample_rate)
    # Find peak index
    peak_index = np.argmax(fft_vals)
    dominant_freq = freqs[peak_index]
    return dominant_freq

# ------------------ Segmentation ------------------
def segment_audio(waveform, sample_rate, tone_duration, gap_duration):
    """
    Given the encrypted waveform, break it into segments corresponding to each tone.
    Assumes that each tone is tone_duration seconds followed by gap_duration seconds.
    Returns a list of tone segments.
    """
    tone_samples = int(sample_rate * tone_duration)
    gap_samples = int(sample_rate * gap_duration)
    total_samples_per_chunk = tone_samples + gap_samples
    num_chunks = len(waveform) // total_samples_per_chunk
    segments = []
    for i in range(num_chunks):
        start = i * total_samples_per_chunk
        tone_segment = waveform[start : start + tone_samples]
        segments.append(tone_segment)
    return segments

# ------------------ Frequency-to-Byte Conversion ------------------
def frequency_to_byte(frequency, base_freq, freq_range):
    """
    Given a base frequency mapping: frequency = base_freq + (byte_val/255)*freq_range,
    invert the mapping to recover byte_val.
    """
    # Ensure frequency is within range
    normalized = (frequency - base_freq) / freq_range
    byte_val = int(round(normalized * 255))
    # Clamp between 0 and 255
    byte_val = max(0, min(255, byte_val))
    return byte_val

# ------------------ Binary-to-Text Conversion ------------------
def binary_string_to_text(binary_str):
    """
    Convert a string of concatenated 8-bit groups (without spaces) back into text.
    """
    # Ensure the string is a multiple of 8
    bytes_list = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    text = "".join([chr(int(b, 2)) for b in bytes_list])
    return text

# ------------------ Decryption Process ------------------
def decrypt_audio(waveform, sample_rate, tone_duration, gap_duration,
                  base_freq, freq_range, chaos_mod_range,
                  dt, a, b, c, passphrase, num_chaotic_samples):
    """
    Decrypt the given encrypted audio waveform:
      1. Segment the waveform into tone segments.
      2. For each tone, estimate the dominant frequency.
      3. Regenerate the chaotic sequence using RK4 from the passphrase.
      4. For each tone, subtract the chaotic offset to recover the base frequency.
      5. Invert the mapping to compute the original byte value.
      6. Reassemble bytes into a binary string and convert to text.
    
    Returns the recovered text.
    """
    # Step 1: Segment the audio
    tone_segments = segment_audio(waveform, sample_rate, tone_duration, gap_duration)
    num_segments = len(tone_segments)
    
    # Derive initial conditions from the passphrase (as in encryption)
    x0, y0, z0 = derive_initial_conditions(passphrase)
    
    # Step 2: Regenerate chaotic sequence for the number of segments
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(num_segments, dt, a, b, c, x0, y0, z0)
    
    # Step 3: Process each tone segment
    binary_str = ""
    for i, segment in enumerate(tone_segments):
        # Estimate the dominant frequency from the segment
        est_freq = estimate_frequency(segment, sample_rate)
        # Remove the chaotic offset: offset = chaotic_value * chaos_mod_range
        offset = chaotic_sequence[i] * chaos_mod_range
        recovered_freq = est_freq - offset
        # Invert frequency mapping to get byte value:
        byte_val = frequency_to_byte(recovered_freq, base_freq, freq_range)
        binary_str += format(byte_val, '08b')
    # Step 4: Convert the binary string to text
    recovered_text = binary_string_to_text(binary_str)
    return recovered_text

# ------------------ Streamlit Interface for Decryption ------------------
def main():
    st.set_page_config(page_title="oscilKEY", layout="wide")
    st.title("oscilKEY: Decryption Module")
    st.markdown("Upload the encrypted audio file and enter the decryption parameters.")
    
    # Upload encrypted audio file
    uploaded_file = st.file_uploader("Upload Encrypted Audio (WAV, FLAC, OGG)", type=["wav", "flac", "ogg"])
    
    # Decryption parameters (these should match those used during encryption)
    passphrase = st.text_input("Enter decryption passphrase:", type="password")
    tone_duration = st.number_input("Tone Duration (sec)", 0.1, 1.0, 0.2, step=0.05)
    gap_duration = st.number_input("Gap Duration (sec)", 0.01, 0.5, 0.05, step=0.01)
    base_freq = st.number_input("Base Frequency (Hz)", 100, 1000, 300)
    freq_range = st.number_input("Frequency Range (Hz)", 100, 2000, 700)
    chaos_mod_range = st.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
    dt = st.number_input("dt", 0.001, 0.1, 0.01, step=0.001)
    a = st.number_input("a", 0.1, 1.0, 0.2, step=0.1)
    b = st.number_input("b", 0.1, 1.0, 0.2, step=0.1)
    c = st.number_input("c", 1.0, 10.0, 5.7, step=0.1)
    num_chaotic_samples = st.number_input("Number of Chaotic Samples", 64, 1024, 128, step=64)
    
    if uploaded_file and passphrase:
        # Read the uploaded audio file
        audio_bytes = uploaded_file.read()
        waveform, file_sample_rate = sf.read(io.BytesIO(audio_bytes))
        # For simplicity, assume the sample_rate used in encryption matches file_sample_rate
        st.audio(audio_bytes, format='audio/wav', start_time=0)
        
        # Decrypt the audio to recover text
        recovered_text = decrypt_audio(
            waveform, file_sample_rate, tone_duration, gap_duration,
            base_freq, freq_range, chaos_mod_range,
            dt, a, b, c, passphrase, num_chaotic_samples
        )
        
        st.header("Decrypted Text")
        st.write(recovered_text)

if __name__ == "__main__":
    main()
