import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import obspy
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Seismic AI Analytics Pro - Advanced",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Styling
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ADVANCED SEISMIC PROCESSING ENGINE
# ==========================================
class AdvancedSeismicEngine:
    def __init__(self, sampling_rate=250.0):
        self.fs = sampling_rate

    def apply_zero_phase_bandpass(self, data, lowcut, highcut, order=4):
        """Zero-Phase Butterworth Filter"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def compute_snr(self, raw_data, filtered_data):
        """Signal-to-Noise Ratio (dB)"""
        noise = raw_data - filtered_data
        p_signal = np.sum(filtered_data ** 2)
        p_noise = np.sum(noise ** 2)
        if p_noise == 0: return 100.0
        snr = 10 * np.log10(p_signal / (p_noise + 1e-10))
        return round(float(snr), 2)

    def sta_lta_detector(self, data, sta_sec=0.5, lta_sec=10.0, on_thresh=2.5, off_thresh=1.2):
        """High-Precision STA/LTA Event Picker"""
        n_sta = max(1, int(sta_sec * self.fs))
        n_lta = max(2, int(lta_sec * self.fs))
        
        cft = classic_sta_lta(data, n_sta, n_lta)
        triggers = trigger_onset(cft, on_thresh, off_thresh)
        
        event_picks = []
        for trg in triggers:
            onset_idx, offset_idx = trg[0], trg[1]
            duration_sec = (offset_idx - onset_idx) / self.fs
            if duration_sec >= 0.1:
                event_picks.append({
                    "onset_index": int(onset_idx),
                    "onset_time_sec": round(float(onset_idx / self.fs), 3),
                    "duration_sec": round(float(duration_sec), 3),
                    "peak_cft": round(float(np.max(cft[onset_idx:offset_idx])), 2)
                })
        return cft, event_picks

    def get_spectrogram(self, data):
        """Computes Short-Time Fourier Transform (Spectrogram)"""
        f, t, Sxx = spectrogram(data, fs=self.fs, nperseg=128, noverlap=100)
        return f, t, 10 * np.log10(Sxx + 1e-10)

# ==========================================
# SYNTHETIC DATA GENERATOR
# ==========================================
def generate_sample_seismic_data(fs=250.0, duration=10.0):
    n_pts = int(fs * duration)
    t = np.linspace(0, duration, n_pts)
    
    # Ambient Random Noise + Monochromatic Industrial Hum
    noise = np.random.normal(0, 0.5, n_pts) + 0.4 * np.sin(2 * np.pi * 50 * t)
    
    # P-wave Arrival at t = 2.0s
    p_wave = np.zeros(n_pts)
    i1, i1_e = int(2.0 * fs), int(4.5 * fs)
    p_wave[i1:i1_e] = np.sin(2 * np.pi * 14 * t[:i1_e-i1]) * np.exp(-1.8 * t[:i1_e-i1]) * 4.0
    
    # S-wave Arrival at t = 5.2s
    s_wave = np.zeros(n_pts)
    i2, i2_e = int(5.2 * fs), int(8.5 * fs)
    s_wave[i2:i2_e] = np.sin(2 * np.pi * 7 * t[:i2_e-i2]) * np.exp(-1.0 * t[:i2_e-i2]) * 6.5
    
    return t, noise + p_wave + s_wave

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("⚡ Seismic AI Analytics Pro - Enterprise Edition")
st.caption("Advanced 2D/1D Processing, Zero-Phase Bandpass, Spectrogram & STA/LTA Picker")

engine = AdvancedSeismicEngine(sampling_rate=250.0)

# SIDEBAR CONTROLS
st.sidebar.header("🕹️ Processing Controls")
data_source = st.sidebar.radio("Data Source:", ["Synthetic Demo Signal", "Upload (.mseed, .sac, .csv)"])

fs = 250.0
t_vec, raw_data = None, None

if data_source == "Synthetic Demo Signal":
    t_vec, raw_data = generate_sample_seismic_data(fs=fs, duration=10.0)
else:
    uploaded_file = st.sidebar.file_uploader("Upload File", type=["mseed", "sac", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                raw_data = df.iloc[:, 0].values
                t_vec = np.arange(len(raw_data)) / fs
            else:
                st_obj = obspy.read(io.BytesIO(uploaded_file.read()))
                tr = st_obj[0]
                fs = tr.stats.sampling_rate
                engine.fs = fs
                raw_data = tr.data
                t_vec = tr.times()
        except Exception as e:
            st.error(f"Error loading file: {e}")

if raw_data is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Bandpass Filter")
    low_cut = st.sidebar.slider("Low Cutoff (Hz)", 0.5, 20.0, 4.0, 0.5)
    high_cut = st.sidebar.slider("High Cutoff (Hz)", 15.0, 100.0, 35.0, 1.0)
    filter_order = st.sidebar.slider("Filter Order", 2, 8, 4)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Event Detection (STA/LTA)")
    sta_win = st.sidebar.slider("STA Window (sec)", 0.1, 2.0, 0.3, 0.1)
    lta_win = st.sidebar.slider("LTA Window (sec)", 2.0, 15.0, 6.0, 0.5)
    trigger_on = st.sidebar.slider("Trigger On Threshold", 1.2, 6.0, 2.2, 0.1)
    trigger_off = st.sidebar.slider("Trigger Off Threshold", 0.5, 3.0, 1.1, 0.1)

    # EXECUTE PROCESSING
    filtered_data = engine.apply_zero_phase_bandpass(raw_data, low_cut, high_cut, order=filter_order)
    snr_val = engine.compute_snr(raw_data, filtered_data)
    cft, events = engine.sta_lta_detector(filtered_data, sta_win, lta_win, trigger_on, trigger_off)

    # TOP METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sampling Frequency", f"{fs} Hz")
    m2.metric("Signal SNR", f"{snr_val} dB")
    m3.metric("Events Detected", f"{len(events)} Picks")
    m4.metric("Engine Health", "Optimal (Zero Phase)", delta_color="normal")

    # TABS FOR ADVANCED VIEWS
    tab1, tab2, tab3 = st.tabs(["📉 Time Series & Detection", "🌈 Time-Frequency Spectrogram", "📋 Event Logs"])

    with tab1:
        # Time Series Trace
        fig_trace = go.Figure()
        fig_trace.add_trace(go.Scatter(x=t_vec, y=raw_data, mode='lines', name='Raw Signal (Noisy)', line=dict(color='#64748b', width=1)))
        fig_trace.add_trace(go.Scatter(x=t_vec, y=filtered_data, mode='lines', name='Filtered Signal', line=dict(color='#38bdf8', width=1.5)))

        for evt in events:
            fig_trace.add_vline(
                x=evt["onset_time_sec"], 
                line_dash="dash", 
                line_color="#ef4444", 
                annotation_text=f" Pick {evt['onset_time_sec']}s",
                annotation_position="top left"
            )

        fig_trace.update_layout(
            title="Time-Domain Trace (Raw vs Zero-Phase Filtered)",
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=360, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_trace, use_container_width=True)

        # STA/LTA Ratio
        fig_cft = go.Figure()
        fig_cft.add_trace(go.Scatter(x=t_vec, y=cft, mode='lines', name='STA/LTA Ratio', line=dict(color='#facc15', width=1.2)))
        fig_cft.add_hline(y=trigger_on, line_dash="dot", line_color="#ef4444", annotation_text="Trigger ON")

        fig_cft.update_layout(
            title="STA/LTA Trigger Function",
            xaxis_title="Time (s)", yaxis_title="Ratio",
            template="plotly_dark", height=240, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_cft, use_container_width=True)

    with tab2:
        # Spectrogram Analysis
        freqs, times, Sxx_db = engine.get_spectrogram(filtered_data)
        fig_spec = go.Figure(data=go.Heatmap(
            z=Sxx_db, x=times, y=freqs, colorscale='Viridis'
        ))
        fig_spec.update_layout(
            title="Short-Time Fourier Transform (STFT) Spectrogram",
            xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
            template="plotly_dark", height=450
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    with tab3:
        if len(events) > 0:
            df_ev = pd.DataFrame(events)
            st.dataframe(df_ev, use_container_width=True)
            
            # CSV Download Button
            csv_data = df_ev.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Pick Report (CSV)", csv_data, "seismic_picks.csv", "text/csv")
        else:
            st.info("No events detected. Lower the Trigger ON slider to detect lower-amplitude events.")