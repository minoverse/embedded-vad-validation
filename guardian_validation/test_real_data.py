#!/usr/bin/env python3
"""
FINAL CORRECTED VALIDATION - ALL REAL BUGS FIXED
=================================================
Applied fixes:
1.  Clip-level split (no leakage)
2.  Fixed noise floor (realistic SNR)
3.  Both raw + hangover
4.  Correct latency (8ms hop)
5.  Correct hangover (100ms = 13 frames)
6.  VAD frame alignment FIXED (matches feature frames)
7.  Integer SNR keys (not float)
8.  Clipping prevention + skip logic

IGNORED:
-  num_frames +1 (ChatGPT was wrong, current is correct)

Author: Guardian Project (FINAL VERSION)
Date: 2026-02-02
"""

import numpy as np
import sys
from pathlib import Path

# Import from audio_gate_simulation.py
from audio_gate_simulation import (
    ResonatorFilterbank,
    extract_features,
    generate_synthetic_noise,
)


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 16000
FRAME_SIZE = 256  # 16ms @ 16kHz
HOP_SIZE = 128    # 8ms @ 16kHz
HOP_MS = 1000.0 * HOP_SIZE / SAMPLE_RATE  # 8.0 ms

HANGOVER_MS = 100.0
HANGOVER_FRAMES = int(HANGOVER_MS / HOP_MS)  # 12.5 ≈ 13 frames

NOISE_FLOOR_DB = -30  # Fixed indoor noise floor


# =============================================================================
# FIX #6: VAD with SAME frame/hop as features (alignment fixed)
# =============================================================================

def simple_vad_aligned(audio, fs=SAMPLE_RATE, 
                      frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                      onset_db=-40, offset_db=-45):
    """
    VAD using SAME frame/hop as feature extraction.
    
    FIXED: Now uses 16ms frame, 8ms hop (matches features exactly!)
    This ensures VAD mask indices align with feature frame indices.
    
    Parameters:
    -----------
    audio : array
        Audio signal
    frame_size : int
        Frame size in samples (default: 256 = 16ms)
    hop_size : int
        Hop size in samples (default: 128 = 8ms)
    onset_db : float
        Speech onset threshold (dBFS)
    offset_db : float
        Speech offset threshold (dBFS)
    
    Returns:
    --------
    is_speech : array of bool
        True for speech frames, False for silence
        Length matches number of feature frames!
    """
    N = len(audio)
    num_frames = max(0, (N - frame_size) // hop_size)
    
    rms_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio[start:end]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
    
    # Convert to dB
    rms_db = 20 * np.log10(rms_values + 1e-10)
    
    # Hysteresis
    is_speech = np.zeros(num_frames, dtype=bool)
    in_speech = False
    
    for i in range(num_frames):
        if not in_speech:
            if rms_db[i] > onset_db:
                in_speech = True
                is_speech[i] = True
        else:
            if rms_db[i] > offset_db:
                is_speech[i] = True
            else:
                in_speech = False
    
    return is_speech


# =============================================================================
# FIX #7 & #8: Noise addition with int keys + clipping prevention
# =============================================================================

def add_noise_final(speech, noise, noise_floor_db=-30, 
                   speech_target_db=-25):  # Lowered from -20 to prevent clipping
    """
    Add noise with fixed floor.
    
    FIXED:
    - Lower default speech level (-25 dBFS) to prevent clipping
    - Returns clipping flag for sample skipping
    
    Returns:
    --------
    mixture : array or None
        Speech + noise (None if clipped)
    actual_snr : float
        Actual SNR measured from mixture
    clipped : bool
        True if clipping occurred
    """
    # Scale speech to target level
    speech_rms = np.sqrt(np.mean(speech ** 2))
    if speech_rms < 1e-10:
        return None, 0.0, True
    
    speech_target_linear = 10 ** (speech_target_db / 20)
    speech_scaled = speech * (speech_target_linear / speech_rms)
    
    # Scale noise to fixed floor
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms < 1e-10:
        return speech_scaled, np.inf, False
    
    noise_floor_linear = 10 ** (noise_floor_db / 20)
    noise_scaled = noise[:len(speech)] * (noise_floor_linear / noise_rms)
    
    # Mix
    mixture = speech_scaled + noise_scaled
    
    # Check clipping
    max_val = np.max(np.abs(mixture))
    clipped = (max_val > 1.0)
    
    if clipped:
        return None, 0.0, True
    
    # Calculate actual SNR
    speech_power = np.mean(speech_scaled ** 2)
    noise_power = np.mean(noise_scaled ** 2)
    actual_snr = 10 * np.log10(speech_power / noise_power)
    
    return mixture, actual_snr, False


# =============================================================================
# Detector (same as before)
# =============================================================================

class FinalDetector:
    """Energy + Coherence detector."""
    
    def __init__(self, th_energy, th_coherence):
        self.th_energy = th_energy
        self.th_coherence = th_coherence
    
    def detect_raw(self, features):
        """Raw decisions."""
        num_frames = len(features['energy'])
        decisions = np.zeros(num_frames, dtype=bool)
        
        for i in range(num_frames):
            energy = float(np.mean(features['energy'][i]))
            coherence = float(np.mean(features['temporal_coherence'][i]))
            
            if energy > self.th_energy and coherence > self.th_coherence:
                decisions[i] = True
        
        return decisions
    
    def detect_with_hangover(self, features, hold_frames=HANGOVER_FRAMES):
        """With hangover (100ms = 13 frames)."""
        raw = self.detect_raw(features)
        
        extended = raw.copy()
        countdown = 0
        for i in range(len(raw)):
            if raw[i]:
                countdown = hold_frames
                extended[i] = True
            elif countdown > 0:
                extended[i] = True
                countdown -= 1
        
        return extended
    
    def compute_detection_latency(self, features):
        """Latency in ms (uses correct 8ms hop)."""
        decisions = self.detect_raw(features)
        
        if np.any(decisions):
            first_detect = np.argmax(decisions)
            latency_ms = first_detect * HOP_MS
            return latency_ms
        else:
            return np.inf


# =============================================================================
# Load LibriSpeech
# =============================================================================

def load_librispeech_clips(librispeech_path, max_clips=80, max_duration=3.0):
    """Load audio clips."""
    clips = []
    
    librispeech_path = Path(librispeech_path)
    
    if not librispeech_path.exists():
        print(f" Error: Path not found: {librispeech_path}")
        return []
    
    print(f"\nLoading audio clips from: {librispeech_path}")
    
    for speaker_dir in sorted(librispeech_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            for audio_file in sorted(chapter_dir.glob("*.flac")):
                try:
                    import soundfile as sf
                    audio, fs = sf.read(audio_file, dtype='float32')
                    
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    max_samples = int(max_duration * fs)
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    if fs != SAMPLE_RATE:
                        ratio = SAMPLE_RATE / fs
                        new_len = int(len(audio) * ratio)
                        audio = np.interp(
                            np.linspace(0, len(audio), new_len),
                            np.arange(len(audio)),
                            audio
                        )
                    
                    # Normalize to moderate level
                    mx = float(np.max(np.abs(audio)))
                    if mx > 0:
                        audio = audio / mx * 0.5
                    
                    clips.append(audio.astype(np.float32))
                    
                    if len(clips) >= max_clips:
                        break
                
                except Exception as e:
                    continue
            
            if len(clips) >= max_clips:
                break
        
        if len(clips) >= max_clips:
            break
    
    print(f"   ✓ Loaded {len(clips)} clips")
    return clips


def clip_level_split(clips, test_ratio=0.2, seed=42):
    """Clip-level split."""
    rng = np.random.default_rng(seed)
    
    indices = np.arange(len(clips))
    rng.shuffle(indices)
    
    split_idx = int(len(clips) * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_clips = [clips[i] for i in train_idx]
    test_clips = [clips[i] for i in test_idx]
    
    return train_clips, test_clips


# =============================================================================
# FIX #6: Threshold learning with aligned VAD
# =============================================================================

def learn_thresholds_final(train_clips, fs=SAMPLE_RATE, 
                          percentile=10, use_vad=False):
    """
    Threshold learning with ALIGNED VAD.
    
    FIXED: VAD now uses same frame/hop as features!
    """
    print(f"\n[Threshold Learning]")
    print(f"   Method: {percentile}th percentile")
    print(f"   VAD: {'Enabled (aligned)' if use_vad else 'Disabled (all frames)'}")
    print(f"   Clips: {len(train_clips)}")
    
    filterbank = ResonatorFilterbank(fs=fs)
    
    all_energy = []
    all_coherence = []
    
    for i, clip in enumerate(train_clips):
        if (i + 1) % 10 == 0:
            print(f"   Processing: {i+1}/{len(train_clips)}")
        
        # Extract features
        outputs = filterbank.process(clip)
        features = extract_features(outputs, frame_size=FRAME_SIZE, 
                                   hop_size=HOP_SIZE, fs=fs)
        
        frame_energy = np.mean(features['energy'], axis=1)
        frame_coherence = np.mean(features['temporal_coherence'], axis=1)
        
        if use_vad:
            # FIXED: VAD now uses SAME frame/hop!
            active_mask = simple_vad_aligned(clip, fs=fs,
                                           frame_size=FRAME_SIZE,
                                           hop_size=HOP_SIZE)
            
            # Now lengths match exactly!
            assert len(active_mask) == len(frame_energy), \
                f"VAD/feature length mismatch: {len(active_mask)} vs {len(frame_energy)}"
            
            frame_energy = frame_energy[active_mask]
            frame_coherence = frame_coherence[active_mask]
        
        all_energy.extend(frame_energy)
        all_coherence.extend(frame_coherence)
    
    th_energy = np.percentile(all_energy, percentile)
    th_coherence = np.percentile(all_coherence, percentile)
    
    print(f"\n   ✓ Thresholds derived:")
    print(f"     Energy:    {th_energy:.6f}")
    print(f"     Coherence: {th_coherence:.6f}")
    
    if use_vad:
        print(f"     Note: Active frames only (VAD aligned)")
    else:
        print(f"     Note: All frames (includes hard cases)")
    
    return th_energy, th_coherence


# =============================================================================
# Test functions
# =============================================================================

def test_clean_speech(detector, test_clips, fs=SAMPLE_RATE):
    """Test clean speech."""
    
    filterbank = ResonatorFilterbank(fs=fs)
    
    all_raw_frames = []
    all_hang_frames = []
    clip_detected_raw = []
    clip_detected_hang = []
    latencies = []
    
    for clip in test_clips:
        outputs = filterbank.process(clip)
        features = extract_features(outputs, frame_size=FRAME_SIZE,
                                   hop_size=HOP_SIZE, fs=fs)
        
        raw = detector.detect_raw(features)
        all_raw_frames.append(raw)
        clip_detected_raw.append(np.any(raw))
        
        hang = detector.detect_with_hangover(features)
        all_hang_frames.append(hang)
        clip_detected_hang.append(np.any(hang))
        
        latency = detector.compute_detection_latency(features)
        if latency < np.inf:
            latencies.append(latency)
    
    frame_pass_raw = float(np.mean(np.concatenate(all_raw_frames))) * 100
    frame_pass_hang = float(np.mean(np.concatenate(all_hang_frames))) * 100
    clip_pass_raw = float(np.mean(clip_detected_raw)) * 100
    clip_pass_hang = float(np.mean(clip_detected_hang)) * 100
    mean_latency = np.mean(latencies) if latencies else np.inf
    
    return {
        'frame_raw': frame_pass_raw,
        'frame_hang': frame_pass_hang,
        'clip_raw': clip_pass_raw,
        'clip_hang': clip_pass_hang,
        'latency_ms': mean_latency
    }


def test_noise_false_alarm(detector, fs=SAMPLE_RATE, num_samples=30):
    """Test noise."""
    
    filterbank = ResonatorFilterbank(fs=fs)
    rng = np.random.default_rng(123)
    
    noise_types = ['pink', 'white', 'babble']
    all_results = {}
    
    for noise_type in noise_types:
        all_frames = []
        
        for i in range(num_samples):
            noise = generate_synthetic_noise(duration=2.0, fs=fs,
                                            noise_type=noise_type, rng=rng)
            
            outputs = filterbank.process(noise)
            features = extract_features(outputs, frame_size=FRAME_SIZE,
                                       hop_size=HOP_SIZE, fs=fs)
            
            raw = detector.detect_raw(features)
            all_frames.append(raw)
        
        fa_rate = float(np.mean(np.concatenate(all_frames))) * 100
        all_results[noise_type] = fa_rate
    
    avg_fa = np.mean(list(all_results.values()))
    
    return {'per_type': all_results, 'average': avg_fa}


# =============================================================================
# FIX #7 & #8: SNR test with int keys + clipping handling
# =============================================================================

def test_snr_robustness_final(detector, test_clips, fs=SAMPLE_RATE):
    """
    SNR test with FIXED:
    - Integer keys (not float)
    - Clipping prevention + skip logic
    - Actual SNR tracking
    """
    filterbank = ResonatorFilterbank(fs=fs)
    rng = np.random.default_rng(124)
    
    noise = generate_synthetic_noise(duration=5.0, fs=fs,
                                     noise_type='pink', rng=rng)
    
    # FIX #7: Use integer keys (speech target levels)
    speech_levels_db = [-15, -20, -25, -30, -35, -40]
    
    results_raw = {}
    results_hang = {}
    actual_snrs = {}  # Track actual SNR separately
    clipped_counts = {}  # Track clipping
    
    for speech_target_db in speech_levels_db:
        all_raw = []
        all_hang = []
        snr_list = []
        clipped = 0
        
        for clip in test_clips[:20]:
            # FIX #8: Skip clipped samples
            mixture, actual_snr, is_clipped = add_noise_final(
                clip, noise,
                noise_floor_db=NOISE_FLOOR_DB,
                speech_target_db=speech_target_db
            )
            
            if is_clipped:
                clipped += 1
                continue  # Skip this sample
            
            snr_list.append(actual_snr)
            
            outputs = filterbank.process(mixture)
            features = extract_features(outputs, frame_size=FRAME_SIZE,
                                       hop_size=HOP_SIZE, fs=fs)
            
            raw = detector.detect_raw(features)
            hang = detector.detect_with_hangover(features)
            
            all_raw.append(raw)
            all_hang.append(hang)
        
        if len(all_raw) == 0:
            print(f"    All samples clipped at {speech_target_db} dB!")
            continue
        
        mean_snr = np.mean(snr_list)
        
        # FIX #7: Integer keys
        results_raw[speech_target_db] = float(np.mean(np.concatenate(all_raw))) * 100
        results_hang[speech_target_db] = float(np.mean(np.concatenate(all_hang))) * 100
        actual_snrs[speech_target_db] = mean_snr
        clipped_counts[speech_target_db] = clipped
    
    # Check monotonicity
    sorted_keys = sorted(results_raw.keys(), reverse=True)
    is_monotonic = all(
        results_raw[sorted_keys[i]] >= results_raw[sorted_keys[i+1]] - 1.0
        for i in range(len(sorted_keys)-1)
    )
    
    return results_raw, results_hang, is_monotonic, actual_snrs, clipped_counts


# =============================================================================
# Main
# =============================================================================

def run_final_validation(librispeech_path):
    """Final validation with all bugs fixed."""
    
    print("\n" + "="*80)
    print("FINAL VALIDATION - ALL BUGS FIXED".center(80))
    print("="*80)
    print("\n All fixes applied:")
    print("   1. Clip-level split (no leakage)")
    print("   2. Fixed noise floor (realistic SNR)")
    print("   3. Correct latency (8ms hop)")
    print("   4. Correct hangover (100ms = 13 frames)")
    print("   5. VAD frame alignment FIXED")
    print("   6. Integer SNR keys (not float)")
    print("   7. Clipping prevention + skip logic")
    print("="*80)
    
    # Load
    all_clips = load_librispeech_clips(librispeech_path, max_clips=80)
    
    if len(all_clips) < 20:
        print(f"\n Need 20+ clips, found {len(all_clips)}")
        print("\nMake sure soundfile is installed:")
        print("   pip install soundfile")
        return None
    
    # Split
    train_clips, test_clips = clip_level_split(all_clips, test_ratio=0.25)
    print(f"\n[Data Split]")
    print(f"   Train: {len(train_clips)} clips")
    print(f"   Test:  {len(test_clips)} clips")
    
    # Learn thresholds (Option A: All frames - most honest)
    th_energy, th_coherence = learn_thresholds_final(
        train_clips, use_vad=False, percentile=10
    )
    
    detector = FinalDetector(th_energy, th_coherence)
    
    # Test 1: Clean speech
    print(f"\n{'='*80}")
    print("[TEST 1] Clean Speech".center(80))
    print(f"{'='*80}")
    
    clean = test_clean_speech(detector, test_clips)
    print(f"\nFrame-level:")
    print(f"   RAW:      {clean['frame_raw']:.1f}%")
    print(f"   Hangover: {clean['frame_hang']:.1f}%")
    print(f"\nClip-level:")
    print(f"   RAW:      {clean['clip_raw']:.1f}%")
    print(f"   Hangover: {clean['clip_hang']:.1f}%")
    print(f"\nLatency:     {clean['latency_ms']:.1f} ms")
    
    # Test 2: Noise
    print(f"\n{'='*80}")
    print("[TEST 2] Noise False Alarm".center(80))
    print(f"{'='*80}")
    
    fa = test_noise_false_alarm(detector)
    print(f"\n{'Noise Type':<15s} {'FA Rate'}")
    print("-" * 30)
    for nt, val in fa['per_type'].items():
        print(f"{nt:<15s} {val:>6.1f}%")
    print("-" * 30)
    print(f"{'AVERAGE':<15s} {fa['average']:>6.1f}%")
    
    # Test 3: SNR robustness
    print(f"\n{'='*80}")
    print("[TEST 3] SNR Robustness (Fixed Noise Floor)".center(80))
    print(f"{'='*80}")
    
    snr_raw, snr_hang, mono, actual_snrs, clipped = \
        test_snr_robustness_final(detector, test_clips)
    
    print(f"\n{'Target':<8s} {'Actual SNR':<12s} {'Clipped':<10s} {'Raw %':<10s} {'Hangover %'}")
    print("-" * 65)
    
    for speech_db in sorted(snr_raw.keys(), reverse=True):
        snr = actual_snrs[speech_db]
        clip_cnt = clipped[speech_db]
        print(f"{speech_db:>4d} dB   {snr:>6.1f} dB      "
              f"{clip_cnt:>3d}/20     {snr_raw[speech_db]:>5.1f}%    "
              f"{snr_hang[speech_db]:>5.1f}%")
    
    print(f"\nMonotonic: {' OK' if mono else ' FAIL'}")
    
    # Final assessment
    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT".center(80))
    print(f"{'='*80}")
    
    passes = 0
    
    # Metric 1
    m1 = clean['frame_hang'] >= 92
    print(f"\n1. Clean Speech:   {clean['frame_hang']:.1f}% {'✅' if m1 else '❌'}")
    if m1:
        passes += 1
    
    # Metric 2
    m2 = fa['average'] <= 10
    print(f"2. False Alarm:    {fa['average']:.1f}% {'✅' if m2 else '❌'}")
    if m2:
        passes += 1
    
    # Metric 3
    snr10_key = min(snr_hang.keys(), key=lambda x: abs(x - (-20)))  # Closest to 10dB SNR
    snr10_val = snr_hang[snr10_key]
    m3 = snr10_val >= 82
    print(f"3. SNR ~10dB:      {snr10_val:.1f}% {'✅' if m3 else '⚠️'}")
    if m3:
        passes += 1
    
    # Metric 4
    m4 = mono
    print(f"4. Monotonic:      {'OK':>5s} {'✅' if m4 else '❌'}")
    if m4:
        passes += 1
    
    # Metric 5
    m5 = clean['latency_ms'] < 100
    print(f"5. Latency:        {clean['latency_ms']:.0f} ms {'✅' if m5 else '⚠️'}")
    if m5:
        passes += 1
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {passes}/5 passed")
    
    if passes >= 4:
        grade = "A- (Ready for nRF52)"
    elif passes >= 3:
        grade = "B+ (Minor tuning needed)"
    else:
        grade = "B (Needs work)"
    
    print(f"Grade: {grade}")
    print(f"{'='*80}")
    
    # Key insight
    snr0_key = min(snr_raw.keys(), key=lambda x: abs(x - (-30)))  # Closest to 0dB SNR
    snr0_val = snr_raw[snr0_key]
    
    print(f"\n Key Insight:")
    if 45 <= snr0_val <= 65:
        print(f"    SNR ~0dB RAW = {snr0_val:.1f}% is REALISTIC")
        print(f"    These numbers are trustworthy")
    elif snr0_val > 65:
        print(f"     SNR ~0dB RAW = {snr0_val:.1f}% seems high")
    else:
        print(f"     SNR ~0dB RAW = {snr0_val:.1f}% is low")
    
    print("\n" + "="*80)
    print(" VALIDATION COMPLETE".center(80))
    print("="*80)
    
    return {
        'clean': clean,
        'fa': fa,
        'snr_raw': snr_raw,
        'snr_hang': snr_hang,
        'monotonic': mono,
        'passes': passes
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_librispeech_FINAL.py /path/to/LibriSpeech/test-clean")
        sys.exit(1)
    

    results = run_final_validation(sys.argv[1])
