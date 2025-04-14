import streamlit as st
import numpy as np
import hashlib
import io
import os
from datetime import datetime
import scipy.io.wavfile as wav  # to read WAV files

# ------------------------------------------------------------------
# Hard‑Coded Parameters (matching the oscilLOCK encryption settings)
# ------------------------------------------------------------------
TONE_DURATION   = 0.11        # seconds
GAP_DURATION    = 0.02        # seconds
BASE_FREQ       = 500         # Hz
FREQ_RANGE      = 1000        # Hz
CHAOS_MOD_RANGE = 349.39      # Hz

# Rossler attractor parameters (from grid search)
DT      = 0.005251616433272467  # seconds
A_PARAM = 0.12477067210511437
B_PARAM = 0.2852679643352883
C_PARAM = 6.801715623942842
BURN_IN = 900  # steps

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------
def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a space‑separated binary string back to text."""
    byte_list = binary_str.split()
    try:
        byte_array = bytearray(int(b, 2) for b in byte_list)
        text = byte_array.decode(encoding)
    except Exception as e:
        text = "Decoding error: " + str(e)
    return text

def derive_initial_conditions(passphrase):
    """Derive initial conditions from SHA‑256(passphrase)."""
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()
    norm_const = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(hash_digest[0:21], 16) / norm_const
    y0 = int(hash_digest[21:42], 16) / norm_const
    z0 = int(hash_digest[42:64], 16) / norm_const
    return x0, y0, z0

# ------------------------------------------------------------------
# Chaotic System (Rossler Attractor) Functions
# ------------------------------------------------------------------
def rossler_derivatives(state, a, b, c):
    """Compute the derivatives for the Rossler attractor."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(state, dt, a, b, c):
    """Perform a single RK4 integration step."""
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence_rossler_rk4(n, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM, 
                                          x0=0.1, y0=0.0, z0=0.0, burn_in=BURN_IN):
    """
    Generate a sequence of chaotic x-values using the Rossler attractor via RK4.
    The sequence is normalized to [0, 1].
    """
    state = np.array([x0, y0, z0], dtype=float)
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------------------------------------------------------
# Helper Function for Inverse XOR Chaining
# ------------------------------------------------------------------
def inverse_xor_chain(chained_bytes, iv):
    """
    Invert the XOR chaining to recover the original plaintext bytes.
    
    :param chained_bytes: List of XOR chained 8-bit integers.
    :param iv: Initialization vector used during encryption.
    :return: List of original plaintext byte values.
    """
    recovered_bytes = []
    prev = iv
    for chained_byte in chained_bytes:
        original_byte = chained_byte ^ prev
        recovered_bytes.append(original_byte)
        prev = chained_byte
    return recovered_bytes

# ------------------------------------------------------------------
# Decryption Function
# ------------------------------------------------------------------
def decrypt_waveform_to_binary(waveform, sample_rate, tone_duration, gap_duration,
                               base_freq, freq_range, chaos_mod_range,
                               dt, a, b, c, passphrase, debug=False):
    """
    Decrypt the provided audio waveform (encrypted via oscilLOCK) to recover a binary string.
    Steps:
      1. Segment the waveform by tone and gap durations.
      2. Estimate the frequency of each tone segment using FFT and parabolic interpolation.
      3. Regenerate the chaotic sequence from the passphrase.
      4. Remove the chaotic modulation to recover the original byte values.
    Returns a space‑separated binary string.
    """
    tone_samples = int(sample_rate * tone_duration)
    gap_samples  = int(sample_rate * gap_duration)
    segment_length = tone_samples + gap_samples
    total_samples = len(waveform)
    n_segments = total_samples // segment_length

    # Regenerate chaotic sequence using derived initial conditions
    x0, y0, z0 = derive_initial_conditions(passphrase)
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(n_segments, dt=dt, a=a, b=b, c=c,
                                                             x0=x0, y0=y0, z0=z0)
    
    binary_list = []
    for i in range(n_segments):
        start = i * segment_length
        end = start + tone_samples
        tone_segment = waveform[start:end]
        
        # Apply a Hann window to reduce spectral leakage
        N = len(tone_segment)
        window = np.hanning(N)
        windowed = tone_segment * window

        # Zero-pad for higher resolution FFT; increased padding factor (×8)
        n_fft = int(2**np.ceil(np.log2(N)) * 8)
        fft_res = np.fft.rfft(windowed, n=n_fft)
        fft_mag = np.abs(fft_res)

        peak_index = np.argmax(fft_mag)
        if 0 < peak_index < len(fft_mag) - 1:
            alpha = fft_mag[peak_index - 1]
            beta  = fft_mag[peak_index]
            gamma = fft_mag[peak_index + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        else:
            p = 0
        
        peak_index_adj = peak_index + p
        freq_res = sample_rate / n_fft
        observed_freq = peak_index_adj * freq_res
        
        chaotic_offset = chaotic_sequence[i] * chaos_mod_range
        plain_freq = observed_freq - chaotic_offset
        
        # Invert mapping: plain_freq = base_freq + (byte_value/255)*freq_range
        byte_val = (plain_freq - base_freq) / freq_range * 255
        byte_val = int(np.rint(byte_val))
        byte_val = max(0, min(255, byte_val))
        
        if debug:
            st.write(f"Segment {i}: observed_freq={observed_freq:.2f} Hz, "
                     f"chaotic_offset={chaotic_offset:.2f}, plain_freq={plain_freq:.2f} Hz, "
                     f"byte_val={byte_val}")
            
        binary_byte = format(byte_val, '08b')
        binary_list.append(binary_byte)
    
    return " ".join(binary_list)

# ------------------------------------------------------------------
# Streamlit UI (Decryption Only; Including XOR Chaining Toggle & Debug Mode)
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="oscilKEY - Decryption", layout="wide")
    st.title("oscilKEY: Audio Waveform Decryption")
    st.markdown("This tool decrypts an encrypted WAV audio (produced by oscilLOCK) to recover the original message.")
    
    st.sidebar.header("Decryption Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Encrypted Audio (WAV)", type=["wav"])
    passphrase = st.sidebar.text_input("Enter Passphrase:", type="password", value="DefaultPassphrase")
    xor_toggle = st.sidebar.checkbox("Enable XOR Chaining", value=True)
    debug_toggle = st.sidebar.checkbox("Enable Debug Mode (Show internal estimates)", value=False)
    enter_button = st.sidebar.button("Enter")
    
    if uploaded_file and passphrase and enter_button:
        try:
            sample_rate_file, waveform = wav.read(uploaded_file)
            # Convert int16 to float32 if necessary (assuming 16-bit PCM)
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32767.0
        except Exception as e:
            st.error(f"Error reading audio file: {e}")
            return
        
        # Run decryption using hard-coded parameters
        binary_output = decrypt_waveform_to_binary(
            waveform, sample_rate_file,
            tone_duration=TONE_DURATION, gap_duration=GAP_DURATION,
            base_freq=BASE_FREQ, freq_range=FREQ_RANGE, chaos_mod_range=CHAOS_MOD_RANGE,
            dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM, passphrase=passphrase,
            debug=debug_toggle
        )
        
        # If XOR chaining was used during encryption, undo the chaining
        if xor_toggle:
            # Convert recovered binary string to list of integers
            int_bytes = [int(b, 2) for b in binary_output.split()]
            # Derive the IV from the passphrase (same derivation as encryption)
            iv = int(hashlib.sha256(passphrase.encode()).hexdigest()[:2], 16)
            original_int_bytes = inverse_xor_chain(int_bytes, iv)
            # Reconstruct the binary string from the recovered integers
            binary_output = " ".join(format(b, '08b') for b in original_int_bytes)
        
        recovered_text = binary_to_text(binary_output)
        
        st.subheader("Decryption Output")
        st.markdown("**Recovered Binary:**")
        st.code(binary_output)
        st.markdown("**Recovered Text:**")
        st.write(recovered_text)

if __name__ == "__main__":
    main()
