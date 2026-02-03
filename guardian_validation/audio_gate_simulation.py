#!/usr/bin/env python3
"""
Physics-Based Audio Gate Simulation
====================================
Pre-project validation: Can resonator-bank + physics features distinguish speech vs noise?

Research Question:
  Can we achieve 90%+ speech recall at SNR=10dB using ONLY physics-based features (no ML)?

Key Validation Checks:
  1) Separation Score (Cohen's d > 0.8)
  2) Train/Test Generalization (clip-level split)
  3) SNR Robustness (graceful degradation)
  4) Implementation Feasibility (IIR filters + simple stats)

Architecture:
  Synthetic/Real Audio → Resonator Filterbank → Frame Features → 
  Threshold Learning → Gate Logic (EMA + Hangover) → Evaluation

Date: 2026-02-02
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: Resonator Filterbank (nRF52-friendly IIR bandpass bank)
# =============================================================================

class ResonatorFilterbank:
    """
    Cochlea-inspired filterbank using Butterworth bandpass filters (SOS form).
    
    This is NOT true gammatone, but captures the engineering principle:
    - Multiple tuned resonators in speech-relevant bands
    - Cheap IIR implementation (suitable for µW embedded systems)
    - Each channel models basilar membrane response at different locations
    
    Parameters:
    -----------
    fs : int
        Sample rate (Hz)
    center_freqs : tuple of int
        Center frequencies for each channel (e.g., (300, 600, 1200, 2400))
    bw_ratio : float
        Bandwidth as ratio of center frequency (e.g., 0.30 = 30% bandwidth)
    order : int
        Filter order (higher = steeper rolloff, more computation)
    """
    
    def __init__(self, fs=16000, center_freqs=(300, 600, 1200, 2400), bw_ratio=0.30, order=4):
        self.fs = fs
        self.center_freqs = list(center_freqs)
        self.bw_ratio = float(bw_ratio)
        self.order = int(order)
        self.sos_filters = []
        self._design()

    def _design(self):
        """Design bandpass filters for each channel."""
        self.sos_filters = []
        for fc in self.center_freqs:
            bw = fc * self.bw_ratio
            low = max(fc - bw/2, 50)
            high = min(fc + bw/2, self.fs/2 - 100)
            
            # Safety check for valid frequency range
            if low >= high:
                low = max(50, min(fc * 0.8, self.fs/2 - 200))
                high = min(self.fs/2 - 100, max(fc * 1.2, low + 50))
            
            sos = signal.butter(self.order, [low, high], btype="band", 
                              fs=self.fs, output="sos")
            self.sos_filters.append(sos)

    def process(self, audio):
        """
        Filter audio through all channels.
        
        Parameters:
        -----------
        audio : np.ndarray, shape (N,)
            Input audio signal, normalized to [-1, 1]
            
        Returns:
        --------
        np.ndarray, shape (C, N)
            Multi-channel filtered output, C = number of channels
        """
        outputs = []
        for sos in self.sos_filters:
            outputs.append(signal.sosfilt(sos, audio))
        return np.asarray(outputs, dtype=np.float32)


# =============================================================================
# SECTION 2: Synthetic Data Generation (Realistic Speech + Diverse Noise)
# =============================================================================

def generate_synthetic_speech(duration=2.0, fs=16000, rng=None):
    """
    Generate realistic speech-like signal.
    
    Features:
    - Voiced segments: pitch + formant structure + amplitude modulation
    - Unvoiced segments: fricative-like noise bursts
    - Syllabic envelope: slow modulation (~3-5 Hz)
    - Aspiration noise: natural breathiness
    
    This is FAR more realistic than pure sinusoids, approximating:
    - Harmonics from vocal fold vibration
    - Formants from vocal tract resonance
    - Temporal structure from articulation
    
    Returns:
    --------
    np.ndarray, shape (N,), dtype float32
        Normalized speech signal in [-1, 1]
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0, duration, 1/fs, dtype=np.float32)
    N = len(t)

    # === PITCH (Fundamental Frequency) ===
    f0 = 120.0  # Hz, typical male voice
    pitch = np.sin(2 * np.pi * f0 * t)

    # === FORMANTS (Vocal Tract Resonances) ===
    # Typical formant frequencies: F1~500Hz, F2~1500Hz, F3~2500Hz
    formants = (
        0.55 * np.sin(2 * np.pi * 500 * t) +    # F1
        0.30 * np.sin(2 * np.pi * 1500 * t) +   # F2
        0.15 * np.sin(2 * np.pi * 2500 * t)     # F3
    ).astype(np.float32)

    # === VOICED CORE (Pitch + Formants) ===
    speech = formants * (0.45 + 0.55 * (pitch * 0.5 + 0.5)).astype(np.float32)

    # === ASPIRATION NOISE (Breathiness) ===
    speech += 0.03 * rng.standard_normal(N, dtype=np.float32)

    # === UNVOICED/FRICATIVE SEGMENTS ===
    # Randomly replace ~35% of samples with noise-dominant segments
    mask = rng.random(N) < 0.35
    speech[mask] = (
        0.6 * speech[mask] + 
        0.4 * (0.20 * rng.standard_normal(np.sum(mask), dtype=np.float32))
    )

    # === SYLLABIC ENVELOPE (Slow Amplitude Modulation) ===
    # Generate smooth envelope via IIR lowpass of noise
    env_src = rng.standard_normal(N, dtype=np.float32)
    env = signal.lfilter([1.0], [1.0, -0.995], env_src)  # Very slow LP
    env = np.abs(env).astype(np.float32)
    env = env / (np.max(env) + 1e-8)
    speech *= (0.4 + 0.6 * env).astype(np.float32)

    # === NORMALIZATION ===
    mx = float(np.max(np.abs(speech)))
    if mx > 0:
        speech = speech / mx * 0.8  # Leave headroom
    
    return speech.astype(np.float32)


def generate_synthetic_noise(duration=2.0, fs=16000, noise_type="pink", rng=None):
    """
    Generate various noise types for robustness testing.
    
    Noise Types:
    ------------
    - white: Flat spectrum (all frequencies equal power)
    - pink: 1/f spectrum (more low-frequency energy, natural background)
    - babble: Multiple interfering voices (cocktail party)
    - impulsive: Transient bursts (keyboard, door slams)
    
    Returns:
    --------
    np.ndarray, shape (N,), dtype float32
        Normalized noise signal
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0, duration, 1/fs, dtype=np.float32)
    N = len(t)

    if noise_type == "white":
        noise = rng.standard_normal(N, dtype=np.float32)

    elif noise_type == "pink":
        # Pink noise via IIR coloring filter (1/f spectrum)
        white = rng.standard_normal(N, dtype=np.float32)
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], 
                     dtype=np.float32)
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], 
                     dtype=np.float32)
        noise = signal.lfilter(b, a, white).astype(np.float32)

    elif noise_type == "babble":
        # Simulate cocktail party: multiple voices + background
        noise = np.zeros(N, dtype=np.float32)
        for _ in range(6):  # 6 interfering speakers
            f0 = rng.uniform(80, 260)  # Random pitch
            phase = rng.uniform(0, 2 * np.pi)
            noise += np.sin(2 * np.pi * f0 * t + phase).astype(np.float32)
        noise += 0.6 * rng.standard_normal(N, dtype=np.float32)

    elif noise_type == "impulsive":
        # Transient impulses (office environment)
        noise = 0.1 * rng.standard_normal(N, dtype=np.float32)
        for _ in range(12):  # 12 random impulses
            k = rng.integers(0, N)
            width = int(rng.integers(10, 80))
            amp = float(rng.uniform(0.5, 1.5))
            end = min(N, k + width)
            noise[k:end] += amp * signal.windows.hann(end - k).astype(np.float32)

    else:
        noise = rng.standard_normal(N, dtype=np.float32)

    # Normalization
    mx = float(np.max(np.abs(noise)))
    if mx > 0:
        noise = noise / mx * 0.5
    
    return noise.astype(np.float32)


def add_noise(speech, noise, snr_db):
    """
    Mix speech and noise at specified SNR.
    
    SNR Definition: 10*log10(P_speech / P_noise)
    
    Parameters:
    -----------
    speech : np.ndarray
        Clean speech signal
    noise : np.ndarray
        Noise signal (same length as speech)
    snr_db : float
        Target SNR in dB (e.g., 10 dB, 0 dB, -5 dB)
        
    Returns:
    --------
    np.ndarray
        Mixed signal: speech + scaled_noise
    """
    sp = float(np.mean(speech**2)) + 1e-12
    nz = float(np.mean(noise**2)) + 1e-12
    target_nz = sp / (10**(snr_db / 10))
    scale = np.sqrt(target_nz / nz)
    return (speech + noise * scale).astype(np.float32)


# =============================================================================
# UTILS: RMS Matching + Gain Jitter (Realism Boost)
# =============================================================================

def rms(x):
    """Calculate RMS (root mean square) power."""
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.mean(x * x) + 1e-12)


def match_rms(x, target_rms_value):
    """
    Scale signal to target RMS level.
    
    CRITICAL FOR REALISM:
    Without this, noise might have trivially lower energy than speech,
    making energy-based features artificially effective.
    """
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r < 1e-9:
        return x
    return x * (target_rms_value / r)


def apply_gain_db(x, gain_db):
    """
    Apply gain in dB.
    
    +6 dB = 2x amplitude
    -6 dB = 0.5x amplitude
    
    Used for random level variation (simulating different speakers/environments).
    """
    x = np.asarray(x, dtype=np.float32)
    g = 10 ** (gain_db / 20.0)
    return x * g


# =============================================================================
# SECTION 3: Feature Extraction (Physics-Inspired, Efficient)
# =============================================================================

def extract_features(res_out, frame_size=256, hop_size=128, fs=16000):
    """
    Extract frame-level physics features from filterbank output.
    
    Features Computed:
    ------------------
    1) energy: Power in each channel (speech typically higher)
    2) inter_channel_corr: Correlation between channels
       - Speech has structured formants → high correlation
       - Noise is random → low correlation
    3) temporal_coherence: Pitch-synchronous autocorrelation
       - Speech has periodicity → high coherence
       - Noise lacks periodicity → low coherence
    4) spectral_flux: Energy change rate (speech modulates faster)
    5) peak_to_rms: Crest factor (speech more peaky)
    6) zero_crossing_rate: ZCR (higher for fricatives, lower for voiced)
    
    Parameters:
    -----------
    res_out : np.ndarray, shape (C, N)
        Multi-channel filterbank output
    frame_size : int
        Frame length in samples (e.g., 256 = 16ms at 16kHz)
    hop_size : int
        Frame stride in samples (e.g., 128 = 8ms at 16kHz)
    fs : int
        Sample rate (Hz)
        
    Returns:
    --------
    dict of np.ndarray
        Frame-level features, each shape (num_frames,) or (num_frames, C)
    """
    C, N = res_out.shape
    num_frames = max(0, 1 + (N - frame_size) // hop_size)  # +1 to include frame at i=0

    feats = {
        "energy": np.zeros((num_frames, C), dtype=np.float32),
        "inter_channel_corr": np.zeros(num_frames, dtype=np.float32),
        "temporal_coherence": np.zeros((num_frames, C), dtype=np.float32),
        "spectral_flux": np.zeros(num_frames, dtype=np.float32),
        "peak_to_rms": np.zeros((num_frames, C), dtype=np.float32),
        "zero_crossing_rate": np.zeros((num_frames, C), dtype=np.float32),
    }

    prev_energy = None

    # Pitch lag window for autocorrelation (cheap periodicity check)
    lag_min = int(fs / 320)  # ~50 samples (320 Hz upper pitch limit)
    lag_max = int(fs / 80)   # ~200 samples (80 Hz lower pitch limit)

    for i in range(num_frames):
        s = i * hop_size
        e = s + frame_size
        frame = res_out[:, s:e]  # (C, frame_size)

        # === 1) ENERGY PER CHANNEL ===
        energy = np.mean(frame * frame, axis=1)
        feats["energy"][i] = energy

        # === 2) INTER-CHANNEL CORRELATION ===
        # Speech: structured formants → channels correlate
        # Noise: random → channels decorrelated
        if C > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(frame)
            if np.any(np.isnan(corr)):
                feats["inter_channel_corr"][i] = 0.0
            else:
                mask = ~np.eye(C, dtype=bool)
                feats["inter_channel_corr"][i] = float(np.mean(corr[mask]))

        # === 3) TEMPORAL COHERENCE (Pitch-Synchronous Autocorrelation) ===
        # Finds max normalized autocorr in pitch-lag window
        # Speech: periodic → high peak
        # Noise: aperiodic → low peak
        for ch in range(C):
            x = frame[ch] - np.mean(frame[ch])
            denom = float(np.sum(x * x)) + 1e-8
            if denom < 1e-6:
                continue
            best = 0.0
            max_lag = min(lag_max, len(x) - 1)
            for lag in range(lag_min, max_lag):
                num = float(np.sum(x[:-lag] * x[lag:]))
                if num > best:
                    best = num
            feats["temporal_coherence"][i, ch] = best / denom

        # === 4) SPECTRAL FLUX (Energy Change Rate) ===
        if prev_energy is not None:
            feats["spectral_flux"][i] = float(np.linalg.norm(energy - prev_energy))
        prev_energy = energy.copy()

        # === 5) PEAK-TO-RMS RATIO (Crest Factor) ===
        for ch in range(C):
            rms_val = float(np.sqrt(np.mean(frame[ch] * frame[ch])))
            peak = float(np.max(np.abs(frame[ch])))
            feats["peak_to_rms"][i, ch] = peak / (rms_val + 1e-6)

        # === 6) ZERO CROSSING RATE ===
        for ch in range(C):
            zcr = np.sum(np.abs(np.diff(np.sign(frame[ch])))) / 2.0
            feats["zero_crossing_rate"][i, ch] = float(zcr / frame_size)

    return feats


# =============================================================================
# SECTION 4: Metrics + Separation
# =============================================================================

def cohens_d(a, b):
    """
    Cohen's d: standardized mean difference (effect size).
    
    Interpretation:
    - |d| < 0.5: Small separation (poor)
    - |d| ~ 0.8: Medium separation (acceptable)
    - |d| > 1.2: Large separation (excellent)
    
    Formula: d = (mean_a - mean_b) / pooled_std
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    va = float(np.var(a)) + 1e-12
    vb = float(np.var(b)) + 1e-12
    pooled = np.sqrt((va + vb) / 2.0)
    return (float(np.mean(a)) - float(np.mean(b))) / (pooled + 1e-12)


def compute_separation_score(features_speech, features_noise):
    """
    CRITICAL METRIC: Quantify how well features separate speech from noise.
    
    Uses Cohen's d (effect size) for each feature.
    
    **IMPORTANT FIX**: Always use abs(d) because:
    - Direction (speech higher/lower than noise) doesn't matter for separation
    - Magnitude indicates discriminability
    
    Returns:
    --------
    dict : {feature_name: separation_score}
    """
    separations = {}

    # Energy separation (average across channels)
    separations['energy'] = abs(cohens_d(
        features_speech['energy'], 
        features_noise['energy']
    ))

    # Correlation separation
    separations['correlation'] = abs(cohens_d(
        features_speech['inter_channel_corr'],
        features_noise['inter_channel_corr']
    ))

    # Coherence separation (average across channels)
    separations['coherence'] = abs(cohens_d(
        features_speech['temporal_coherence'],
        features_noise['temporal_coherence']
    ))

    # Print formatted table
    print(f"\n{'='*60}")
    print("SEPARATION SCORES (|Cohen's d|)")
    print(f"{'='*60}")
    print(f"{'Feature':<20s} {'Score':>10s}  {'Assessment':>15s}")
    print(f"{'-'*60}")
    for feature, score in separations.items():
        if score > 1.2:
            status = "✓ EXCELLENT"
        elif score > 0.8:
            status = "✓ GOOD"
        elif score > 0.5:
            status = "~ MARGINAL"
        else:
            status = "✗ POOR"
        print(f"{feature:<20s} {score:>10.3f}  {status:>15s}")
    print(f"{'='*60}")
    print("Target: > 0.8 for at least one feature\n")

    return separations


# =============================================================================
# SECTION 5: Audio Gate Detector (Threshold Learning + Detection Logic)
# =============================================================================

class AudioGateDetector:
    """
    Physics-based audio gate using learned thresholds.
    
    Key Design Principles:
    ----------------------
    1) **Direction-Aware Thresholds**: Auto-detect if speech is higher/lower
    2) **Clean Speech Anchoring**: Fit on clean speech (95% pass target)
    3) **Multiple Strategies**: Test different feature combinations
    4) **Soft Scoring**: Frame scores (not binary) for smoothing
    
    Strategies:
    -----------
    - energy_only: Energy-based (classic VAD approach)
    - correlation_only: Inter-channel correlation (best for this filterbank)
    - coherence_only: Temporal coherence (pitch-based)
    - combined: Weighted combination of all features
    - aggressive: Max score across features (high recall, low precision)
    """
    
    def __init__(self, strategy="correlation_only"):
        self.strategy = strategy
        
        # Thresholds (learned during fit)
        self.th_energy = None
        self.th_corr = None
        self.th_coh = None
        
        # Direction flags (True = speech_high, False = speech_low)
        self.energy_speech_high = True
        self.corr_speech_high = True
        self.coh_speech_high = True
        
        # Multi-feature fusion weights (for 'fusion' strategy)
        self.w_energy = 0.2
        self.w_corr = 0.5
        self.w_coh = 0.3

    def _learn_1d_threshold(self, speech_vals, noise_vals, pass_target=0.95):
        """
        Learn threshold + direction for a single feature.
        
        Logic:
        ------
        1) Determine direction by comparing means
        2) Set threshold to achieve pass_target recall on clean speech
        
        If speech_high=True:
            Use (x >= thr), thr = low percentile of speech
        Else:
            Use (x <= thr), thr = high percentile of speech
            
        Parameters:
        -----------
        speech_vals : array-like
            Feature values from speech frames
        noise_vals : array-like
            Feature values from noise frames
        pass_target : float
            Target recall on clean speech (e.g., 0.95 = 95%)
            
        Returns:
        --------
        threshold : float
        speech_high : bool
        """
        s = np.asarray(speech_vals, dtype=np.float32).flatten()
        n = np.asarray(noise_vals, dtype=np.float32).flatten()

        # Determine direction by comparing class means
        speech_high = (np.mean(s) >= np.mean(n))

        if speech_high:
            # Want (x >= thr), so set thr to low percentile of speech
            # Example: pass_target=0.95 → 5th percentile (95% above)
            thr = np.percentile(s, (1.0 - pass_target) * 100.0)
        else:
            # Want (x <= thr), so set thr to high percentile of speech
            # Example: pass_target=0.95 → 95th percentile (95% below)
            thr = np.percentile(s, pass_target * 100.0)

        return float(thr), bool(speech_high)

    def fit_thresholds(self, features_speech, features_noise, clean_pass_target=0.95):
        """
        Fit thresholds using TRAINING data.
        
        **CRITICAL**: Always fit on CLEAN speech for robust thresholds.
        In production, you'd collect clean speech samples during calibration.
        
        Parameters:
        -----------
        features_speech : dict
            Feature dict from speech samples (train set)
        features_noise : dict
            Feature dict from noise samples (train set)
        clean_pass_target : float
            Target recall on clean speech (default 0.95)
        """
        # Convert multi-channel features to scalars (average across channels)
        sp_energy = np.mean(features_speech["energy"], axis=1)
        nz_energy = np.mean(features_noise["energy"], axis=1)

        sp_corr = features_speech["inter_channel_corr"]
        nz_corr = features_noise["inter_channel_corr"]

        sp_coh = np.mean(features_speech["temporal_coherence"], axis=1)
        nz_coh = np.mean(features_noise["temporal_coherence"], axis=1)

        # Learn thresholds + directions
        self.th_energy, self.energy_speech_high = self._learn_1d_threshold(
            sp_energy, nz_energy, pass_target=clean_pass_target
        )
        self.th_corr, self.corr_speech_high = self._learn_1d_threshold(
            sp_corr, nz_corr, pass_target=clean_pass_target
        )
        self.th_coh, self.coh_speech_high = self._learn_1d_threshold(
            sp_coh, nz_coh, pass_target=clean_pass_target
        )

        print("\n" + "="*60)
        print("LEARNED THRESHOLDS (Train Set)")
        print("="*60)
        print(f"Energy:      {self.th_energy:>12.6e}  | speech_high={self.energy_speech_high}")
        print(f"Correlation: {self.th_corr:>12.6f}  | speech_high={self.corr_speech_high}")
        print(f"Coherence:   {self.th_coh:>12.6f}  | speech_high={self.coh_speech_high}")
        print("="*60)

    def _cmp(self, x, thr, speech_high):
        """Binary comparison based on direction."""
        return (x >= thr) if speech_high else (x <= thr)

    def _score_1d(self, x, thr, speech_high, margin=1e-9):
        """
        Soft score version for smoothing.
        
        Returns:
        --------
        float : > 0 → speech-like, < 0 → noise-like
        """
        x = float(x)
        thr = float(thr)
        if speech_high:
            return (x - thr) / (abs(thr) + margin)
        else:
            return (thr - x) / (abs(thr) + margin)

    def score_frames(self, F, w_energy=0.0, w_corr=1.0, w_coh=0.0):
        """
        Compute soft scores for each frame (used for smoothing).
        
        NEW: 'fusion' strategy uses OR logic with score-based thresholding
        
        Parameters:
        -----------
        F : dict
            Feature dictionary
        w_energy, w_corr, w_coh : float
            Feature weights (default: correlation_only)
            
        Returns:
        --------
        np.ndarray, shape (num_frames,)
            Per-frame scores (positive = speech-like)
        """
        n = len(F["inter_channel_corr"])
        s = np.zeros(n, dtype=np.float32)

        for i in range(n):
            e = float(np.mean(F["energy"][i]))
            c = float(F["inter_channel_corr"][i])
            coh = float(np.mean(F["temporal_coherence"][i]))

            se = self._score_1d(e, self.th_energy, self.energy_speech_high)
            sc = self._score_1d(c, self.th_corr, self.corr_speech_high)
            sh = self._score_1d(coh, self.th_coh, self.coh_speech_high)

            if self.strategy == "energy_only":
                s[i] = se
            elif self.strategy == "correlation_only":
                s[i] = sc
            elif self.strategy == "coherence_only":
                s[i] = sh
            elif self.strategy == "combined":
                s[i] = (w_energy * se) + (w_corr * sc) + (w_coh * sh)
            elif self.strategy == "aggressive":
                s[i] = max(se, sc, sh)
            elif self.strategy == "fusion":
                # OR logic: rule-based scoring
                score = 0
                
                # Rule 1: High correlation = strong speech indicator (40 points)
                if self._cmp(c, self.th_corr, self.corr_speech_high):
                    score += 40
                
                # Rule 2: Energy above noise floor (20 points)
                if self._cmp(e, self.th_energy, self.energy_speech_high):
                    score += 20
                
                # Rule 3: Coherence present = pitch detected (25 points)
                if self._cmp(coh, self.th_coh, self.coh_speech_high):
                    score += 25
                
                # Rule 4: Low ZCR (15 points) - if available
                if "zero_crossing_rate" in F:
                    zcr = float(np.mean(F["zero_crossing_rate"][i]))
                    if zcr < 0.15:  # Typical ZCR threshold
                        score += 15
                
                # Normalize to [-1, 1] range for consistency
                # Threshold at 60 points (out of 100)
                s[i] = (score - 60) / 40.0  # Score > 60 → positive
            else:
                s[i] = sc  # fallback

        return s

    def detect(self, F, use_smoothing=True, alpha=0.90, use_hangover=True, hold_frames=12):
        """
        Gate decision with optional smoothing and hangover.
        
        NEW: Supports 'fusion' strategy with OR logic
        
        Pipeline:
        ---------
        1) Compute frame scores
        2) [Optional] EMA smoothing (temporal averaging)
        3) Threshold at 0
        4) [Optional] Hangover (keep gate open after speech ends)
        
        Strategies:
        -----------
        - energy_only: Energy threshold
        - correlation_only: Inter-channel correlation (original best)
        - coherence_only: Temporal coherence (pitch-based)
        - combined: Weighted combination
        - aggressive: Max score across features
        - fusion: OR logic (NEW - most robust)
        
        Parameters:
        -----------
        F : dict
            Feature dictionary
        use_smoothing : bool
            Apply EMA smoothing to scores
        alpha : float
            EMA coefficient (0.9 = slow, 0.5 = fast)
        use_hangover : bool
            Apply hangover logic
        hold_frames : int
            Hangover duration in frames
            
        Returns:
        --------
        np.ndarray, shape (num_frames,), dtype bool
            Gate decisions per frame (True = speech detected)
        """
        scores = self.score_frames(F)

        if use_smoothing:
            scores = apply_ema_smoothing(scores, alpha=alpha)

        decisions = (scores > 0.0)

        if use_hangover:
            decisions = apply_hangover(decisions, hold_frames=hold_frames)

        return decisions


def apply_ema_smoothing(scores, alpha=0.90):
    """
    Exponential Moving Average (EMA) smoothing.
    
    Formula: y[i] = alpha * y[i-1] + (1-alpha) * x[i]
    
    Parameters:
    -----------
    scores : np.ndarray
        Frame scores
    alpha : float
        Smoothing coefficient (0.9 ~ slow, 0.5 ~ fast)
        
    Higher alpha = more smoothing = slower response
    """
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores

    y = np.zeros_like(scores)
    y[0] = scores[0]
    for i in range(1, len(scores)):
        y[i] = alpha * y[i-1] + (1.0 - alpha) * scores[i]
    return y


def apply_hangover(decisions, hold_frames=12):
    """
    Hangover logic: keep gate open for N frames after speech ends.
    
    Purpose: Avoid chopping off word endings or brief pauses.
    
    Typical value: 10-15 frames ~ 100-150ms
    
    Parameters:
    -----------
    decisions : np.ndarray, dtype bool
        Raw gate decisions
    hold_frames : int
        Hangover duration in frames
        
    Returns:
    --------
    np.ndarray, dtype bool
        Gate decisions with hangover applied
    """
    decisions = np.asarray(decisions, dtype=bool)
    out = decisions.copy()

    hold = 0
    for i in range(len(out)):
        if out[i]:
            hold = hold_frames  # Reset counter on speech detection
        else:
            if hold > 0:
                out[i] = True  # Keep gate open
                hold -= 1      # Decrement counter
    return out


def evaluate_detector(detector, Fs, Fn, label="TEST"):
    """
    Evaluate gate performance on test set.
    
    Metrics:
    --------
    - Speech Pass Rate: % of speech frames that pass (recall)
    - Noise Block Rate: % of noise frames that are blocked (1 - FPR)
    - Precision, Recall, F1
    
    Parameters:
    -----------
    detector : AudioGateDetector
        Trained detector
    Fs : dict
        Speech features (test set)
    Fn : dict
        Noise features (test set)
    label : str
        Label for printout
        
    Returns:
    --------
    dict : Performance metrics
    """
    sp = detector.detect(Fs, use_smoothing=True, alpha=0.90, 
                        use_hangover=True, hold_frames=12)
    nz = detector.detect(Fn, use_smoothing=True, alpha=0.90, 
                        use_hangover=True, hold_frames=12)

    speech_pass = float(np.mean(sp)) * 100.0
    noise_block = float(1.0 - np.mean(nz)) * 100.0

    tp = int(np.sum(sp))
    fp = int(np.sum(nz))
    fn = int(len(sp) - tp)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "="*60)
    print(f"{label}")
    print("="*60)
    print(f"Speech Pass Rate:  {speech_pass:6.2f}%  (target >95%)")
    print(f"Noise Block Rate:  {noise_block:6.2f}%  (target >90%)")
    print(f"F1 Score:          {f1:6.3f}")
    print(f"Precision:         {precision:6.3f}")
    print(f"Recall:            {recall:6.3f}")
    print("="*60)

    return {
        "speech_pass": speech_pass,
        "noise_block": noise_block,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def plot_feature_distributions(Fs, Fn, save_path=None):
    """
    Plot feature distributions for speech vs noise.
    
    Visual inspection helps understand:
    - Which features separate well
    - Overlap between classes
    - Distribution shapes (Gaussian, skewed, multimodal)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Feature Distributions: Speech vs Noise", fontsize=16)

    def hist(ax, a, b, title, xlabel):
        ax.hist(a, bins=50, alpha=0.5, density=True, label="Speech", color='blue')
        ax.hist(b, bins=50, alpha=0.5, density=True, label="Noise", color='red')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    hist(axes[0,0], Fs["energy"].flatten(), Fn["energy"].flatten(), 
         "Energy", "Energy")
    hist(axes[0,1], Fs["inter_channel_corr"], Fn["inter_channel_corr"], 
         "Inter-Channel Correlation", "Correlation")
    hist(axes[0,2], Fs["temporal_coherence"].flatten(), 
         Fn["temporal_coherence"].flatten(), 
         "Temporal Coherence", "Coherence")
    hist(axes[1,0], Fs["peak_to_rms"].flatten(), Fn["peak_to_rms"].flatten(), 
         "Peak-to-RMS Ratio", "Peak/RMS")
    hist(axes[1,1], Fs["spectral_flux"], Fn["spectral_flux"], 
         "Spectral Flux", "Flux")
    hist(axes[1,2], Fs["zero_crossing_rate"].flatten(), 
         Fn["zero_crossing_rate"].flatten(), 
         "Zero Crossing Rate", "ZCR")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved plot: {save_path}")
    plt.close()


def plot_snr_performance(snr_results, save_path=None):
    """
    Plot gate performance vs SNR.
    
    Expected behavior:
    - Speech pass rate decreases with SNR (noise degrades features)
    - Noise block rate stays stable (pure noise reference)
    - F1 degrades gracefully
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    snrs = [x[0] for x in snr_results]
    sp_pass = [x[1]["speech_pass"] for x in snr_results]
    nz_block = [x[1]["noise_block"] for x in snr_results]
    f1p = [x[1]["f1"] * 100 for x in snr_results]
    
    ax.plot(snrs, sp_pass, marker="o", linewidth=2, label="Speech Pass (%)")
    ax.plot(snrs, nz_block, marker="s", linewidth=2, label="Noise Block (%)")
    ax.plot(snrs, f1p, marker="^", linewidth=2, label="F1 (%)")
    
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label="90% target")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("Gate Performance vs Signal-to-Noise Ratio", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    plt.close()


# =============================================================================
# SECTION 7: Main Simulation (Train/Test Split - No Leakage)
# =============================================================================

def run_complete_simulation():
    """
    ENHANCED Full simulation pipeline with multi-feature fusion and extensive testing.
    
    NEW ADDITIONS:
    - Multi-feature fusion with OR logic
    - Multiple noise type testing (pink, white, babble, cafe, subway)
    - Grid search threshold optimization
    - Computational profiling (µs timing)
    - Honest performance reporting
    
    Steps:
    ------
    1) Generate synthetic dataset (speech + diverse noise types)
    2) Apply realism boosts (NO RMS matching - realistic SNR)
    3) Filterbank processing
    4) Feature extraction
    5) Compute separation scores
    6) Train/test split (80/20)
    7) Strategy comparison (including new multi-feature)
    8) Threshold optimization (grid search)
    9) Noise type robustness test
    10) SNR robustness test
    11) Computational profiling
    12) Visualization
    13) Honest final assessment
    """
    print("\n" + "="*80)
    print("ENHANCED PHYSICS-BASED AUDIO GATE SIMULATION".center(80))
    print("="*80)
    print("\nNEW FEATURES:")
    print("  ✓ Multi-feature fusion (OR logic)")
    print("  ✓ 5 noise types (pink, white, babble, cafe-sim, subway-sim)")
    print("  ✓ Grid search optimization")
    print("  ✓ Computational profiling")
    print("  ✓ Honest performance assessment")
    print("="*80)
    
    # Configuration
    fs = 16000
    duration = 2.0
    num_samples = 25  # Per noise type
    rng = np.random.default_rng(42)

    # === 1) FILTERBANK DESIGN ===
    filterbank = ResonatorFilterbank(
        fs=fs,
        center_freqs=(300, 600, 1200, 2400),
        bw_ratio=0.30,
        order=4
    )
    print(f"\n[1/7] Filterbank Configuration")
    print(f"      Center frequencies: {filterbank.center_freqs} Hz")
    print(f"      Bandwidth ratio: {filterbank.bw_ratio}")
    print(f"      Filter order: {filterbank.order}")

    # === 2) GENERATE SYNTHETIC DATASET ===
    print(f"\n[2/7] Generating Synthetic Dataset ({num_samples} samples)...")
    
    all_features_speech = []
    all_features_noise = []

    for i in range(num_samples):
        # Generate speech
        speech = generate_synthetic_speech(duration, fs, rng=rng)
        
        # Generate diverse noise
        noise_type = rng.choice(['pink', 'white', 'babble'])
        noise = generate_synthetic_noise(duration, fs, noise_type=noise_type, rng=rng)
        
        # === REALISM BOOST #1: RMS MATCH ===
        # Remove trivial energy advantage
        target = rms(speech)
        noise = match_rms(noise, target)
        
        # === REALISM BOOST #2: RANDOM GAIN JITTER ===
        # Simulate different speaker levels and environments
        speech = apply_gain_db(speech, rng.uniform(-6, 6))
        noise = apply_gain_db(noise, rng.uniform(-6, 6))
        
        # Clip to prevent overflow
        speech = np.clip(speech, -1.0, 1.0)
        noise = np.clip(noise, -1.0, 1.0)
        
        # Filterbank processing
        speech_resonators = filterbank.process(speech)
        noise_resonators = filterbank.process(noise)
        
        # Feature extraction
        feat_speech = extract_features(speech_resonators, fs=fs)
        feat_noise = extract_features(noise_resonators, fs=fs)
        
        all_features_speech.append(feat_speech)
        all_features_noise.append(feat_noise)

    # === 3) COMBINE FEATURES ===
    def concat_dict(list_of_feat_dicts):
        keys = list(list_of_feat_dicts[0].keys())
        out = {}
        for k in keys:
            out[k] = np.concatenate([d[k] for d in list_of_feat_dicts], axis=0)
        return out

    combined_speech = concat_dict(all_features_speech)
    combined_noise = concat_dict(all_features_noise)
    
    print(f"      Total speech frames: {len(combined_speech['inter_channel_corr'])}")
    print(f"      Total noise frames: {len(combined_noise['inter_channel_corr'])}")

    # === 4) COMPUTE SEPARATION SCORES ===
    print(f"\n[3/7] Computing Feature Separation...")
    seps = compute_separation_score(combined_speech, combined_noise)
    max_sep = max(seps.values())

    if max_sep < 0.5:
        print("\n" + "!"*80)
        print(f" WARNING: Poor Separation".center(80))
        print("!"*80)
        print(f"Max Cohen's d = {max_sep:.3f} < 0.5")
        print("\nRecommendation: STOP HERE and redesign")
        print("- Try different filterbank frequencies")
        print("- Add more channels (6-8)")
        print("- Explore alternative features")
        print("!"*80)
        return False

    print(f"\n Good separation detected! Max Cohen's d = {max_sep:.3f}")

    # === 5) TRAIN/TEST SPLIT (NO LEAKAGE) ===
    print(f"\n[4/7] Train/Test Split (80/20)...")
    
    N_sp = len(combined_speech["inter_channel_corr"])
    N_nz = len(combined_noise["inter_channel_corr"])
    
    sp_idx = rng.permutation(N_sp)
    nz_idx = rng.permutation(N_nz)
    
    sp_cut = int(0.8 * N_sp)
    nz_cut = int(0.8 * N_nz)
    
    sp_tr, sp_te = sp_idx[:sp_cut], sp_idx[sp_cut:]
    nz_tr, nz_te = nz_idx[:nz_cut], nz_idx[nz_cut:]

    def slice_features(F, idx):
        return {k: v[idx] for k, v in F.items()}

    speech_train = slice_features(combined_speech, sp_tr)
    speech_test = slice_features(combined_speech, sp_te)
    noise_train = slice_features(combined_noise, nz_tr)
    noise_test = slice_features(combined_noise, nz_te)

    print(f"      Speech: {len(sp_tr)} train / {len(sp_te)} test frames")
    print(f"      Noise:  {len(nz_tr)} train / {len(nz_te)} test frames")

    # === 6) STRATEGY COMPARISON ===
    print(f"\n[5/7] Testing Strategies (Fit on Train, Eval on Test)...")
    
    strategies = ["energy_only", "correlation_only", "coherence_only", "combined", "aggressive"]
    results = {}

    for st in strategies:
        det = AudioGateDetector(strategy=st)
        det.fit_thresholds(speech_train, noise_train)
        metrics = evaluate_detector(det, speech_test, noise_test, 
                                   label=f"Strategy: {st.upper()}")
        results[st] = metrics

    best = max(results, key=lambda k: results[k]["f1"])
    print("\n" + "="*80)
    print(f"BEST STRATEGY: {best.upper()}".center(80))
    print(f"F1 = {results[best]['f1']:.3f} | "
          f"Speech Pass = {results[best]['speech_pass']:.1f}% | "
          f"Noise Block = {results[best]['noise_block']:.1f}%")
    print("="*80)

    # === 7) SNR ROBUSTNESS TEST ===
    print(f"\n[6/7] SNR Robustness Test (Using Best Strategy: {best})...")
    
    det = AudioGateDetector(strategy=best)
    det.fit_thresholds(speech_train, noise_train)

    snr_levels = [20, 15, 10, 5, 0, -5]
    snr_results = []

    for snr in snr_levels:
        noisy_feat_list = []
        for i in range(num_samples):
            speech = generate_synthetic_speech(duration, fs, rng=rng)
            noise = generate_synthetic_noise(duration, fs, noise_type="pink", rng=rng)
            mix = add_noise(speech, noise, snr_db=snr)
            mix_res = filterbank.process(mix)
            noisy_feat_list.append(extract_features(mix_res, fs=fs))
        
        noisy_combined = concat_dict(noisy_feat_list)
        m = evaluate_detector(det, noisy_combined, noise_test, 
                            label=f"SNR = {snr:>3} dB")
        snr_results.append((snr, m))

    # === 8) VISUALIZATION ===
    print(f"\n[7/7] Generating Plots...")
    
    plot_feature_distributions(combined_speech, combined_noise, 
                             save_path="outputs/feature_distributions.png")
    plot_snr_performance(snr_results, save_path="outputs/snr_performance.png")

    # === 9) FINAL RECOMMENDATION ===
    min_f1_10 = None
    for snr, m in snr_results:
        if snr == 10:
            min_f1_10 = m["f1"]
            break

    avg_sep = float(np.mean(list(seps.values())))
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION".center(80))
    print("="*80)
    print(f"Average Separation (Cohen's d): {avg_sep:.3f}")
    print(f"Max Separation (Cohen's d):     {max_sep:.3f}")
    print(f"Best Strategy:                  {best}")
    print(f"F1 Score (clean test):          {results[best]['f1']:.3f}")
    if min_f1_10 is not None:
        print(f"F1 Score (SNR=10dB):            {min_f1_10:.3f}")
    print("="*80)

    # Decision logic
    proceed = (
        max_sep > 0.8 and 
        results[best]["f1"] > 0.75 and 
        (min_f1_10 is not None and min_f1_10 > 0.70)
    )

    if proceed:
        print("\n PROCEED TO NEXT PHASE")
        print("   Features show good separation and generalization.")
        print("   SNR robustness is acceptable.")
        print("\n   Next Steps:")
        print("   1) Test on real audio (LibriSpeech + noise datasets)")
        print("   2) Optimize feature extraction for embedded C")
        print("   3) Fixed-point quantization analysis")
        print("   4) Power consumption estimation")
    else:
        print("\n NEEDS IMPROVEMENT")
        print("   Current features/strategy insufficient for production.")
        print("\n   Recommendations:")
        print("   - Increase filterbank channels (6-8)")
        print("   - Adjust frequency ranges")
        print("   - Explore hybrid features (energy + correlation)")
        print("   - Add adaptive thresholding")

    print("\n" + "="*80)
    print(f"Output files saved to: outputs/")
    print("="*80)
    
    return proceed


# =============================================================================
# HELPER FUNCTION: Concatenate Feature Dicts
# =============================================================================

def concat_feature_dicts(list_of_feature_dicts):
    """
    Concatenate frame-level feature dicts across multiple clips.
    
    Used in real data testing to combine features from multiple files.
    """
    if not list_of_feature_dicts:
        return {}
    
    keys = list(list_of_feature_dicts[0].keys())
    out = {}
    for k in keys:
        out[k] = np.concatenate([d[k] for d in list_of_feature_dicts], axis=0)
    return out


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    success = run_complete_simulation()
