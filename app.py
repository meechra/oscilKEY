import streamlit as st
import numpy as np
import plotly.graph_objects as go
import hashlib
import io
import os
from datetime import datetime
import scipy.io.wavfile as wav  # to read WAV files

# ------------------------------------------------------------------
# Hard‑Coded Parameters (must match the encryptor)
# ------------------------------------------------------------------
TONE_DURATION    = 0.11        # sec
GAP_DURATION     = 0.02        # sec
BASE_FREQ        = 500         # Hz
FREQ_RANGE       = 1000        # Hz
CHAOS_MOD_RANGE  = 349.39      # Hz
DT               = 0.005251616433272467
A_PARAM          = 0.12477067210511437
B_PARAM          = 0.2852679643352883
C_PARAM          = 6.801715623942842
BURN_IN          = 900         # steps

# ------------------------------------------------------------------
# Utility / Conversion
# ------------------------------------------------------------------
def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a space‑separated binary string back to text."""
    byte_list = binary_str.split()
    try:
        byte_array = bytearray(int(b,2) for b in byte_list)
        return byte_array.decode(encoding)
    except Exception as e:
        return f"Decoding error: {e}"

def derive_initial_conditions(passphrase):
    """Derive x0,y0,z0 from SHA‑256(passphrase)."""
    digest = hashlib.sha256(passphrase.encode()).hexdigest()
    norm = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(digest[0:21],16)/norm
    y0 = int(digest[21:42],16)/norm
    z0 = int(digest[42:64],16)/norm
    return x0, y0, z0

# ------------------------------------------------------------------
# Rossler Attractor + RK4
# ------------------------------------------------------------------
def rossler_derivatives(state, a, b, c):
    x,y,z = state
    return np.array([-y - z, x + a*y, b + z*(x - c)])

def rk4_step(state, dt, a, b, c):
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2*k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2*k2, a, b, c)
    k4 = rossler_derivatives(state + dt*k3, a, b, c)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence_rossler_rk4(n, dt, a, b, c, x0, y0, z0, burn_in):
    """Generate normalized x[i] sequence after burn‑in."""
    state = np.array([x0,y0,z0],dtype=float)
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    seq = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        seq.append(state[0])
    arr = np.array(seq)
    return ((arr - arr.min())/(arr.max()-arr.min())).tolist()

# ------------------------------------------------------------------
# Decryption: Estimate freq, remove chaos, map back to bits
# ------------------------------------------------------------------
def decrypt_waveform_to_binary(waveform, sample_rate, passphrase):
    tone_samples = int(sample_rate * TONE_DURATION)
    gap_samples  = int(sample_rate * GAP_DURATION)
    seg_len      = tone_samples + gap_samples
    n_segments   = len(waveform) // seg_len

    # regenerate chaotic offsets
    x0,y0,z0 = derive_initial_conditions(passphrase)
    chaos_seq = generate_chaotic_sequence_rossler_rk4(
        n_segments, DT, A_PARAM, B_PARAM, C_PARAM, x0, y0, z0, BURN_IN
    )

    binary_list = []
    for i in range(n_segments):
        start = i*seg_len
        tone = waveform[start:start+tone_samples]

        # window + zero‑pad for fine FFT
        N = len(tone)
        windowed = tone * np.hanning(N)
        n_fft = int(2**np.ceil(np.log2(N))*4)
        mag = np.abs(np.fft.rfft(windowed, n=n_fft))

        # peak + parabolic interp
        idx = np.argmax(mag)
        if 0<idx<len(mag)-1:
            α,β,γ = mag[idx-1],mag[idx],mag[idx+1]
            p = 0.5*(α-γ)/(α - 2*β + γ)
        else:
            p = 0
        freq_res = sample_rate / n_fft
        observed = (idx + p) * freq_res

        # remove chaotic offset
        plain_freq = observed - chaos_seq[i]*CHAOS_MOD_RANGE

        # invert mapping to byte
        val = (plain_freq - BASE_FREQ)/FREQ_RANGE * 255
        b = int(np.rint(val))
        b = max(0,min(255,b))
        binary_list.append(f"{b:08b}")

    return " ".join(binary_list)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
def create_waveform_figure(wf, sr):
    t = np.linspace(0, len(wf)/sr, len(wf), endpoint=False)
    fig = go.Figure(go.Scatter(x=t, y=wf, mode='lines'))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="oscilKEY - Decryptor", layout="wide")
    st.title("oscilKEY: Audio‑Based Decryptor")

    st.sidebar.header("Upload & Passphrase")
    uploaded = st.sidebar.file_uploader("Encrypted WAV", type="wav")
    pwd       = st.sidebar.text_input("Passphrase", type="password")
    decrypt   = st.sidebar.button("Decrypt Audio")

    if uploaded and pwd and decrypt:
        try:
            sr_file, data = wav.read(uploaded)
            if data.dtype == np.int16:
                data = data.astype(np.float32)/32767.0
            sr = sr_file
        except Exception as e:
            st.error(f"Failed to read WAV: {e}")
            return

        st.subheader("Encrypted Waveform")
        st.audio(uploaded, format="audio/wav")
        st.plotly_chart(create_waveform_figure(data, sr), use_container_width=True)

        with st.spinner("Decrypting…"):
            binary_str = decrypt_waveform_to_binary(data, sr, pwd)
            text = binary_to_text(binary_str)

        st.subheader("Recovered Binary")
        st.code(binary_str)
        st.subheader("Recovered Text")
        st.write(text)

if __name__ == "__main__":
    main()
