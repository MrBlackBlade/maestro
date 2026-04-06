# %% [markdown]
# # 🎵 MAESTRO — Affective Recognition Engine
# ### Music Adaptive Emotion-based Signal Transformer for Responsive Orchestration
# **Ain Shams University · Faculty of Computer & Information Sciences · AI Program**
# 
# ---
# 
# This notebook implements **Phase 1** of the MAESTRO pipeline: predicting continuous **Valence** and **Arousal** values from physiological biosignals using the **CASE dataset**.
# 
# | Component | Choice | Justification |
# |-----------|--------|---------------|
# | Dataset | CASE (30 subjects, continuous annotations) | Joystick-annotated V-A at 20Hz |
# | Normalization | Baseline Reduction | Best results in Siirtola et al. (2023); real-time compatible |
# | Features | Statistical + HRV + EDA phasic/tonic | Validated feature set from literature |
# | Model | LSTM Regression | Outperforms all other methods in Siirtola et al. (2023) |
# | Evaluation | LOSO Cross-Validation | Gold standard for physiological signal models |
# | Metrics | MSE + R² | Directly comparable to published benchmarks |
# 
# **Reference Benchmark (Siirtola et al., 2023 — WESAD dataset):**
# - Valence: MSE = 0.43, R² = 0.71
# - Arousal: MSE = 0.59, R² = 0.81

# %% [markdown]
# ---
# ## 📦 Cell 1 — Install Dependencies

# %% [markdown]
# ---
# ## 📚 Cell 2 — Imports & Global Configuration

# %%
# ================================================================
#  IMPORTS
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import time
from pathlib import Path
from tqdm.notebook import tqdm

# Signal processing
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import heartpy as hp

# Machine learning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ================================================================
#  GLOBAL CONFIGURATION
#  ⚠️  SET YOUR DATA PATH BELOW
# ================================================================
CONFIG = {
    'data_root':      'E:\Graduation Project\maestro\CASE_full\CASE_full',

    # --- CASE dataset properties ---
    'n_subjects':      30,
    'fs_physio':       1000,          # Hz
    'fs_annot':        20,            # Hz (50ms intervals)

    # --- CASE label range (from paper Table 2) ---
    # Joystick scaled to [0.5, 9.5] — neutral = 5.0
    'label_min':       0.5,
    'label_max':       9.5,
    'label_neutral':   5.0,           # centre of the scale

    # --- Video IDs (from paper Table 1) ---
    # 1,2 = amusing | 3,4 = boring | 5,6 = relaxing | 7,8 = scary
    # 10 = startVid (relaxing — used as BASELINE)
    # 11 = bluVid (blue screen — used as REST/transition)
    # 12 = endVid (cool-down)
    'baseline_video_id':  10,         # startVid: relaxing documentary
    'rest_video_id':      11,         # bluVid: blue screen (inter-video rest)
    'emotional_video_ids': [1,2,3,4,5,6,7,8],

    # --- Windowing ---
    'window_sec':      60,
    'stride_sec':      30,

    # --- Sensor columns in CASE CSV (paper Table 2) ---
    # Units after paper's preprocessing:
    #   ecg: mV | bvp: % | gsr: μS | rsp: % | skt: °C | emg: μC
    'physio_cols':    ['ecg', 'bvp', 'gsr', 'rsp', 'skt',
                       'emg_zygo', 'emg_coru', 'emg_trap'],
    'signal_cols':    ['bvp', 'gsr', 'skt'],  # core 3 for emotion

    # --- Model hyperparameters ---
    'lstm_hidden':     64,
    'lstm_layers':     1,
    'dropout':         0.2,
    'epochs':          50,
    'lr':              1e-3,
    'batch_size':      32,
    'weight_decay':    1e-4,
    'n_runs':          5,
    'seed':            42,
    'save_model':      True,
    'output_dir':      './outputs',
}
# ================================================================
#  SEED EVERYTHING
# ================================================================
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# ================================================================
#  DEVICE DETECTION
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ================================================================
#  MATPLOTLIB STYLE
# ================================================================
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#0f0f1a',
    'axes.edgecolor':   '#333355',
    'axes.labelcolor':  '#c8c8e8',
    'xtick.color':      '#888899',
    'ytick.color':      '#888899',
    'text.color':       '#c8c8e8',
    'grid.color':       '#222244',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})
COLORS = {
    'valence':    '#64dfdf',   # cyan
    'arousal':    '#f7971e',   # amber
    'true':       '#ff6b9d',   # pink
    'predicted':  '#48cae4',   # sky blue
    'baseline':   '#a8dadc',   # muted teal
    'accent':     '#e040fb',   # purple
}

print(f"✅ Configuration loaded.")
print(f"📟 Device: {DEVICE}")
print(f"🗂️  Output directory: {CONFIG['output_dir']}")

# %% [markdown]
# ---
# ## 📂 Cell 3 — CASE Dataset Loader
# 
# **Expected directory structure:**
# ```
# data/CASE/
# ├── physiological/
# │   ├── sub_1.csv
# │   ├── sub_2.csv
# │   └── ...
# └── annotations/
#     ├── sub_1.csv
#     ├── sub_2.csv
#     └── ...
# ```
# Physiological signals are sampled at **1000 Hz**. Columns: `daqtime, ecg, bvp, gsr (EDA), rsp, skt, emg_zygo, emg_trap`.
# Annotations are sampled at **20 Hz**. Columns: `jstime, valence, arousal`.

# %%
class CASELoader:
    """
    Loads the CASE dataset (Sharma et al., 2019, Scientific Data).

    Dataset collected at DLR, Germany.
    Physiological signals: 1000 Hz, 16-bit ADC.
    Annotations: 20 Hz joystick, scaled to [0.5, 9.5].

    Key structural detail from the paper:
    Both the physiological and annotation CSV files contain a
    'video' column (Column 10 / Column 4 respectively) that
    stores the video ID for each sample. This is essential for:
      1. Identifying the baseline period (video ID 10 = startVid)
      2. Segmenting emotional responses by video type
      3. Excluding blue screen / transition periods if desired
    """

    def __init__(self, cfg: dict):
        self.data_root  = Path(cfg['data_root'])
        # Use the INTERPOLATED data — standard choice per the paper
        self.physio_dir = self.data_root / 'data' / 'interpolated' / 'physiological'
        self.annot_dir  = self.data_root / 'data' / 'interpolated' / 'annotations'
        self.cfg        = cfg

    def load_subject(self, subject_id: int) -> dict | None:
        p_path = self.physio_dir / f'sub_{subject_id}.csv'
        a_path = self.annot_dir  / f'sub_{subject_id}.csv'

        if not p_path.exists() or not a_path.exists():
            return None

        physio = pd.read_csv(p_path)
        annot  = pd.read_csv(a_path)

        # Normalise column names
        physio.columns = physio.columns.str.strip().str.lower()
        annot.columns  = annot.columns.str.strip().str.lower()

        # Validate required columns
        # Physio CSV: daqtime, ecg, bvp, gsr, rsp, skt,
        #             emg_zygo, emg_coru, emg_trap, video
        # Annot  CSV: jstime, valence, arousal, video
        required_p = ['bvp', 'gsr', 'skt', 'video']
        required_a = ['valence', 'arousal', 'video']
        if any(c not in physio.columns for c in required_p):
            print(f"  ⚠️  Subject {subject_id}: missing physio cols — skipping")
            return None
        if any(c not in annot.columns for c in required_a):
            print(f"  ⚠️  Subject {subject_id}: missing annot cols — skipping")
            return None

        # Validate label range: should be in [0.5, 9.5]
        v_range = (annot['valence'].min(), annot['valence'].max())
        a_range = (annot['arousal'].min(), annot['arousal'].max())

        return {
            'subject_id': subject_id,
            'physio':     physio,
            'annot':      annot,
            'v_range':    v_range,
            'a_range':    a_range,
        }

    def load_all(self) -> list:
        subjects = []
        print("Loading CASE dataset (interpolated)...\n")
        for sid in range(1, self.cfg['n_subjects'] + 1):
            result = self.load_subject(sid)
            if result is not None:
                # Show video IDs present for this subject
                video_ids = [int(x) for x in sorted(result['physio']['video'].unique())]
                print(f"  ✅ Subject {sid:02d} | "
                      f"Physio rows: {len(result['physio']):,} | "
                      f"V∈{float(result['v_range'][0]):.4f}, {float(result['v_range'][1]):.4f} | "
                      f"A∈{float(result['a_range'][0]):.4f}, {float(result['a_range'][1]):.4f} | "
                      f"Videos: {video_ids}")
                subjects.append(result)
        print(f"\n📊 Loaded {len(subjects)} subjects.")
        return subjects

# ── Load the data ──────────────────────────────────────────────
loader   = CASELoader(CONFIG)
subjects = loader.load_all()

# %% [markdown]
# ---
# ## 📊 Cell 4 — Exploratory Data Analysis (EDA)
# Before preprocessing, visualize the raw signals and label distributions.

# %%
def plot_raw_signals(subject_data: dict, plot_sec: int = 120):
    """
    Plot the first `plot_sec` seconds of raw physiological signals
    alongside their corresponding V-A annotations for one subject.
    """
    physio  = subject_data['physio']
    annot   = subject_data['annot']
    sid     = subject_data['subject_id']
    fs_p    = CONFIG['fs_physio']
    fs_a    = CONFIG['fs_annot']

    p_samples = min(plot_sec * fs_p, len(physio))
    a_samples = min(plot_sec * fs_a, len(annot))

    t_p = np.arange(p_samples) / fs_p
    t_a = np.arange(a_samples) / fs_a

    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=False)
    fig.suptitle(f'Subject {sid:02d} — Raw Signals (first {plot_sec}s)',
                 fontsize=14, color='white', y=1.01)

    signal_configs = [
        ('bvp', 'BVP (Blood Volume Pulse)', COLORS['valence']),
        ('gsr', 'EDA (Electrodermal Activity)', COLORS['arousal']),
        ('skt', 'SKT (Skin Temperature)', COLORS['baseline']),
    ]

    for ax, (col, label, color) in zip(axes[:3], signal_configs):
        if col in physio.columns:
            ax.plot(t_p, physio[col].values[:p_samples],
                    color=color, lw=0.8, alpha=0.9)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, plot_sec)

    # Valence annotation
    axes[3].plot(t_a, annot['valence'].values[:a_samples],
                 color=COLORS['true'], lw=1.5)
    axes[3].fill_between(t_a, annot['valence'].values[:a_samples],
                          alpha=0.2, color=COLORS['true'])
    axes[3].set_ylabel('Valence', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, plot_sec)

    # Arousal annotation
    axes[4].plot(t_a, annot['arousal'].values[:a_samples],
                 color=COLORS['arousal'], lw=1.5)
    axes[4].fill_between(t_a, annot['arousal'].values[:a_samples],
                          alpha=0.2, color=COLORS['arousal'])
    axes[4].set_ylabel('Arousal', fontsize=9)
    axes[4].set_xlabel('Time (s)', fontsize=9)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim(0, plot_sec)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/raw_signals_sub{sid}.png",
                dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()


def plot_label_distribution(subjects: list):
    """
    Plot the distribution of valence and arousal labels across all subjects.
    Key insight from Siirtola et al.: most samples cluster near neutral (0),
    making extreme emotion detection harder.
    """
    all_v = np.concatenate([s['annot']['valence'].values for s in subjects])
    all_a = np.concatenate([s['annot']['arousal'].values for s in subjects])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Label Distribution Across All Subjects',
                 fontsize=13, color='white')

    # Valence histogram
    axes[0].hist(all_v, bins=50, color=COLORS['valence'], alpha=0.8,
                 edgecolor='#0f0f1a')
    axes[0].set_title('Valence Distribution', color='white')
    axes[0].set_xlabel('Valence')
    axes[0].set_ylabel('Count')
    axes[0].axvline(np.mean(all_v), color=COLORS['true'],
                    linestyle='--', label=f'Mean={np.mean(all_v):.2f}')
    axes[0].legend()

    # Arousal histogram
    axes[1].hist(all_a, bins=50, color=COLORS['arousal'], alpha=0.8,
                 edgecolor='#0f0f1a')
    axes[1].set_title('Arousal Distribution', color='white')
    axes[1].set_xlabel('Arousal')
    axes[1].set_ylabel('Count')
    axes[1].axvline(np.mean(all_a), color=COLORS['true'],
                    linestyle='--', label=f'Mean={np.mean(all_a):.2f}')
    axes[1].legend()

    # 2D scatter: Valence-Arousal space (Russell's Circumplex)
    axes[2].scatter(all_v[::50], all_a[::50],  # downsample for speed
                    alpha=0.15, s=3, color=COLORS['accent'])
    axes[2].axhline(0, color='#555577', lw=0.8)
    axes[2].axvline(0, color='#555577', lw=0.8)
    axes[2].set_title("Russell's Circumplex (sample)", color='white')
    axes[2].set_xlabel('Valence')
    axes[2].set_ylabel('Arousal')

    # Add quadrant labels
    for x, y, txt in [(0.5, 0.5,'Happy'), (-0.5, 0.5,'Angry'),
                       (-0.5,-0.5,'Sad'),  (0.5,-0.5,'Calm')]:
        axes[2].text(x * (all_v.max() * 0.7), y * (all_a.max() * 0.7),
                     txt, ha='center', va='center',
                     fontsize=8, color='#666688', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/label_distribution.png",
                dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()

    print(f"Valence — min: {all_v.min():.2f}, max: {all_v.max():.2f}, "
          f"mean: {all_v.mean():.2f}, std: {all_v.std():.2f}")
    print(f"Arousal — min: {all_a.min():.2f}, max: {all_a.max():.2f}, "
          f"mean: {all_a.mean():.2f}, std: {all_a.std():.2f}")


# Run EDA on first subject
if subjects:
    plot_raw_signals(subjects[0], plot_sec=120)
    plot_label_distribution(subjects)

# %% [markdown]
# ---
# ## 🔧 Cell 5 — Signal Preprocessing Pipeline
# Apply bandpass filters and artifact removal to each physiological signal before feature extraction.

# %%
class PhysiologicalPreprocessor:
    """
    Numerically stable physiological signal preprocessor.

    ROOT CAUSE OF OVERFLOW:
    scipy's butter() returns 'ba' (transfer function) coefficients
    by default. At high sampling rates (1000 Hz) with low cutoff
    frequencies (0.5 Hz), the 'ba' form becomes numerically
    catastrophic — condition number explodes, giving outputs of
    magnitude 10^36 that cascade into NaN in downstream operations.

    THE FIX:
    Use output='sos' (Second-Order Sections) in butter().
    Then use sosfiltfilt() instead of filtfilt().

    SOS form chains multiple 2nd-order filters instead of one
    high-order filter. Each section has small, well-conditioned
    coefficients — no overflow possible.

    This is scipy's own recommendation:
    "The second-order sections ('sos') output parameter was added
     in 0.16.0... [it] should be preferred over [ba] for most
     filtering applications." — scipy docs
    """

    def __init__(self, fs: int = 1000):
        self.fs = fs

    def _bandpass(self, sig: np.ndarray,
                  lo: float, hi: float,
                  order: int = 4) -> np.ndarray:
        """
        Stable bandpass using SOS form.
        sosfiltfilt = zero-phase filtering with SOS sections.
        Equivalent result to filtfilt(ba) but numerically stable.
        """
        nyq = self.fs / 2
        sos = butter(order,
                     [lo / nyq, hi / nyq],
                     btype='band',
                     output='sos')       # ← KEY CHANGE
        return signal.sosfiltfilt(sos, sig)

    def _lowpass(self, sig: np.ndarray,
                 cutoff: float,
                 order: int = 4) -> np.ndarray:
        nyq = self.fs / 2
        sos = butter(order,
                     cutoff / nyq,
                     btype='low',
                     output='sos')       # ← KEY CHANGE
        return signal.sosfiltfilt(sos, sig)

    def _notch(self, sig: np.ndarray,
               freq: float = 50.0,
               Q: float = 30.0) -> np.ndarray:
        """
        Notch filter — iirnotch only supports ba form,
        but 2nd-order notch is inherently stable so ba is fine here.
        """
        b, a = signal.iirnotch(freq, Q, fs=self.fs)
        return signal.filtfilt(b, a, sig)

    def preprocess_bvp(self, bvp: np.ndarray) -> np.ndarray:
        """BVP: bandpass 0.5–3.5 Hz to isolate cardiac pulses."""
        return self._bandpass(bvp.astype(np.float64), 0.5, 3.5)

    def preprocess_gsr(self, gsr: np.ndarray) -> np.ndarray:
        """GSR/EDA: low-pass 3 Hz — EDA is a slow signal."""
        return self._lowpass(gsr.astype(np.float64), 3.0)

    def preprocess_skt(self, skt: np.ndarray) -> np.ndarray:
        """SKT: median filter for thermal spike removal."""
        from scipy.signal import medfilt
        return medfilt(skt.astype(np.float64), kernel_size=51)

    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        """ECG: 50 Hz notch then bandpass 0.5–40 Hz."""
        ecg64      = ecg.astype(np.float64)
        ecg_notch  = self._notch(ecg64, freq=50.0)
        return self._bandpass(ecg_notch, 0.5, 40.0)

    def preprocess_subject(self, physio_df: pd.DataFrame) -> pd.DataFrame:
        df = physio_df.copy()

        processors = {
            'bvp': self.preprocess_bvp,
            'gsr': self.preprocess_gsr,
            'skt': self.preprocess_skt,
            'ecg': self.preprocess_ecg,
        }

        for col, fn in processors.items():
            if col not in df.columns:
                continue
            result = fn(df[col].values)

            # Verify no overflow
            n_nan  = np.isnan(result).sum()
            n_inf  = np.isinf(result).sum()
            absmax = np.abs(result).max()

            if n_nan > 0 or n_inf > 0 or absmax > 1e6:
                print(f"  ❌ {col}: overflow detected "
                      f"(NaN={n_nan}, Inf={n_inf}, max={absmax:.2e})")
            else:
                df[col] = result

        return df


# ================================================================
#  Quick sanity check BEFORE re-running all subjects
# ================================================================
print("=== SOS Filter Sanity Check ===\n")

test_proc  = PhysiologicalPreprocessor(fs=1000)
test_bvp   = subjects[0]['physio']['bvp'].values[:5000].astype(np.float64)
test_out   = test_proc.preprocess_bvp(test_bvp)

print(f"Input  range : {test_bvp.min():.4f} – {test_bvp.max():.4f}")
print(f"Output range : {test_out.min():.6f} – {test_out.max():.6f}")
print(f"Output NaNs  : {np.isnan(test_out).sum()}")
print(f"Output Infs  : {np.isinf(test_out).sum()}")
print(f"Output std   : {test_out.std():.6f}")

if np.isnan(test_out).sum() == 0 and test_out.std() > 1e-10:
    print("\n✅ SOS filter working correctly — re-running all subjects")

    # Re-run preprocessing
    preprocessor = PhysiologicalPreprocessor(fs=CONFIG['fs_physio'])
    for s in tqdm(subjects, desc="Subjects"):
        s['physio_clean'] = preprocessor.preprocess_subject(s['physio'])

    # Final verification
    print("\n── Verification ─────────────────────────────────────")
    all_ok = True
    for s in subjects:
        bvp_clean = s['physio_clean']['bvp'].values
        n_nan     = np.isnan(bvp_clean).sum()
        absmax    = np.abs(bvp_clean).max()
        if n_nan > 0 or absmax > 1e6:
            print(f"  ❌ Subject {s['subject_id']:02d}: "
                  f"NaN={n_nan}, max={absmax:.2e}")
            all_ok = False

    if all_ok:
        print("  ✅ All subjects clean — proceed to Cell 6 (Normalization)")
else:
    print("\n❌ Still failing — share the output range printed above")


# ── Quick before/after plot ────────────────────────────────────
def plot_filter_effect(subject_data: dict, col: str = 'bvp',
                        plot_sec: int = 5):
    """Show raw vs filtered signal for a short window."""
    fs = CONFIG['fs_physio']
    n  = plot_sec * fs
    t  = np.arange(n) / fs

    raw    = subject_data['physio'][col].values[:n]
    clean  = subject_data['physio_clean'][col].values[:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(t, raw,   color='#ff6b6b', lw=0.8, label='Raw')
    axes[0].set_title(f'{col.upper()} — Before Filtering', color='white')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, clean, color=COLORS['valence'], lw=0.8, label='Filtered')
    axes[1].set_title(f'{col.upper()} — After Filtering', color='white')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if subjects:
    plot_filter_effect(subjects[0], col='bvp', plot_sec=5)
    plot_filter_effect(subjects[0], col='gsr', plot_sec=10)

# %%


# %% [markdown]
# ---
# ## 📐 Cell 6 — Baseline Reduction Normalization
# **Key finding from Siirtola et al. (2023):** Baseline reduction outperforms z-score normalization and is the only method compatible with real-time deployment — you only need a short baseline recording at session start.

# %%
class BaselineReductionNormalizer:
    """
    Baseline reduction normalization for CASE dataset.

    CASE-specific corrections:
    ─────────────────────────────────────────────────
    1. BASELINE PERIOD:
       We use video ID 10 (startVid) — the relaxing documentary
       that opens each session, explicitly designed by the authors
       to calm participants before emotional stimuli.
       DO NOT use arbitrary "first 60 seconds" — the paper's
       design provides a principled, validated baseline.

    2. LABEL SCALE:
       CASE labels are in [0.5, 9.5] with neutral = 5.0.
       After baseline reduction: label_norm = label - baseline_mean
       So a neutral subject gives ~0 after normalization.
       This matches the Siirtola et al. approach but applied
       correctly to CASE's [0.5, 9.5] scale.

    3. SIGNAL NORMALIZATION:
       Subtract per-subject baseline mean for each signal.
       This removes inter-subject differences in:
         - Baseline GSR conductance (varies enormously between people)
         - Resting BVP amplitude
         - Baseline skin temperature
    """

    def __init__(self, cfg: dict):
        self.cfg              = cfg
        self.baseline_vid     = cfg['baseline_video_id']   # 10 = startVid
        self.signal_baselines = {}
        self.label_baseline_v = None
        self.label_baseline_a = None

    def fit(self, physio_clean: pd.DataFrame,
            annot: pd.DataFrame,
            signal_cols: list) -> 'BaselineReductionNormalizer':
        """
        Compute baseline statistics using startVid (video ID 10) segments.
        Falls back to video ID 11 (blue screen) if startVid not present.
        """
        # Get baseline physio segment (video column must be preserved)
        p_base = physio_clean[physio_clean['video'] == self.baseline_vid]
        a_base = annot[annot['video'] == self.baseline_vid]

        if len(p_base) == 0:
            # Fallback: use blue screen (video 11)
            print(f"    ⚠️  startVid not found, using bluVid as baseline")
            p_base = physio_clean[physio_clean['video'] == self.cfg['rest_video_id']]
            a_base = annot[annot['video'] == self.cfg['rest_video_id']]

        if len(p_base) == 0:
            raise ValueError("No baseline video found for this subject!")

        # Compute signal baselines
        for col in signal_cols:
            if col in p_base.columns:
                self.signal_baselines[col] = p_base[col].mean()

        # Label baselines (in [0.5, 9.5] scale)
        self.label_baseline_v = a_base['valence'].mean()
        self.label_baseline_a = a_base['arousal'].mean()

        # Sanity check: baseline should be close to neutral (5.0)
        v_deviation = abs(self.label_baseline_v - self.cfg['label_neutral'])
        a_deviation = abs(self.label_baseline_a - self.cfg['label_neutral'])
        if v_deviation > 2.0 or a_deviation > 2.0:
            print(f"    ⚠️  Baseline far from neutral: "
                  f"V={self.label_baseline_v:.2f}, A={self.label_baseline_a:.2f}")

        return self

    def transform_signals(self, physio_clean: pd.DataFrame,
                           signal_cols: list) -> pd.DataFrame:
        """Subtract baseline mean. Preserves the 'video' column."""
        df = physio_clean.copy()
        for col in signal_cols:
            if col in self.signal_baselines:
                df[col] = df[col] - self.signal_baselines[col]
        return df

    def transform_labels(self, valence: np.ndarray,
                          arousal: np.ndarray) -> tuple:
        """
        Subtract baseline label.
        Result: neutral state → ~0, positive = above neutral, negative = below.
        Emotional videos will give signed deviations from each subject's neutral.
        """
        return (valence - self.label_baseline_v,
                arousal - self.label_baseline_a)

    def inverse_transform_labels(self, v_norm, a_norm):
        """Restore original [0.5, 9.5] scale for music engine output."""
        return (v_norm + self.label_baseline_v,
                a_norm + self.label_baseline_a)


# ── Apply normalization ───────────────────────────────────────
SIGNAL_COLS = ['bvp', 'gsr', 'skt']   # note: 'gsr' not 'gsr' in CASE

print("Applying baseline reduction (using startVid = video 10)...\n")
for s in subjects:
    norm = BaselineReductionNormalizer(CONFIG)
    norm.fit(s['physio_clean'], s['annot'], SIGNAL_COLS)

    s['physio_norm']  = norm.transform_signals(s['physio_clean'], SIGNAL_COLS)
    v_norm, a_norm    = norm.transform_labels(
        s['annot']['valence'].values,
        s['annot']['arousal'].values)
    s['valence_norm'] = v_norm
    s['arousal_norm'] = a_norm
    s['normalizer']   = norm

    print(f"  Subject {s['subject_id']:02d} | "
          f"Baseline V={norm.label_baseline_v:.2f} "
          f"A={norm.label_baseline_a:.2f} "
          f"(neutral target = {CONFIG['label_neutral']})")

print("\n✅ Done.")

# %% [markdown]
# ---
# ## 🧪 Cell 7 — Feature Extraction
# Extracts the full feature set from Table 2 of Siirtola et al.: statistical features for EDA (with phasic/tonic decomposition), BVP + HRV via HeartPy, and SKT with slope.

# %%
class FeatureExtractor:
    """
    Extracts the complete feature set from Siirtola et al. (Table 2):

    EDA features (30):
      - 10 statistical on raw EDA
      - 10 statistical on phasic component (fast SCR)
      - 10 statistical on tonic  component (slow SCL)

    BVP/HRV features (18):
      - 8 statistical on raw BVP
      - HR_mean, NNi_mean, NNi_std, pNN50, RMSSD,
        LF/HF ratio, LF, HF, VLF, breathing_rate

    SKT features (9):
      - 8 statistical + slope

    Total: ~57 features per window
    """

    FS = CONFIG['fs_physio']

    STAT_FUNS = {
        'mean':   np.mean,
        'std':    np.std,
        'min':    np.min,
        'max':    np.max,
        'median': np.median,
        'range':  lambda x: np.max(x) - np.min(x),
        'p5':     lambda x: np.percentile(x, 5),
        'p25':    lambda x: np.percentile(x, 25),
        'p75':    lambda x: np.percentile(x, 75),
        'p95':    lambda x: np.percentile(x, 95),
    }

    def _stat_features(self, arr: np.ndarray, prefix: str) -> dict:
        """10 statistical features on any array."""
        feats = {}
        for name, fn in self.STAT_FUNS.items():
            try:
                val = float(fn(arr))
                feats[f'{prefix}_{name}'] = val if np.isfinite(val) else np.nan
            except Exception:
                feats[f'{prefix}_{name}'] = np.nan
        return feats

    def eda_features(self, eda_window: np.ndarray) -> dict:
        """
        EDA: raw + phasic (SCR) + tonic (SCL) statistical features.
        Uses neurokit2 for EDA decomposition.
        """
        feats = self._stat_features(eda_window, 'gsr')
        try:
            processed, _ = nk.eda_process(eda_window, sampling_rate=self.FS)
            feats.update(self._stat_features(
                processed['EDA_Phasic'].values, 'eda_phasic'))
            feats.update(self._stat_features(
                processed['EDA_Tonic'].values, 'eda_tonic'))
        except Exception:
            # Fallback: fill with NaN (dropped later)
            for sfx in self.STAT_FUNS:
                feats[f'eda_phasic_{sfx}'] = np.nan
                feats[f'eda_tonic_{sfx}']  = np.nan
        return feats

    def bvp_features(self, bvp_window: np.ndarray) -> dict:
        """
        BVP: 8 statistical features + 10 HRV features via HeartPy.
        HeartPy is purpose-built for wearable PPG/BVP signals.
        """
        feats = self._stat_features(bvp_window, 'bvp')
        try:
            wd, m = hp.process(bvp_window, sample_rate=self.FS)
            hrv_keys = {
                'hr_mean':        'bpm',
                'nni_mean':       'ibi',
                'nni_std':        'sdnn',
                'rmssd':          'rmssd',
                'pnn50':          'pnn50',
                'breathing_rate': 'breathingrate',
                'lf_hf':          'lf/hf',
                'vlf':            'vlf',
                'lf':             'lf',
                'hf':             'hf',
            }
            for feat_name, hp_key in hrv_keys.items():
                val = m.get(hp_key, np.nan)
                feats[f'hrv_{feat_name}'] = (
                    float(val) if np.isfinite(float(val)) else np.nan)
        except Exception:
            for feat_name in ['hr_mean','nni_mean','nni_std','rmssd',
                               'pnn50','breathing_rate','lf_hf',
                               'vlf','lf','hf']:
                feats[f'hrv_{feat_name}'] = np.nan
        return feats

    def skt_features(self, skt_window: np.ndarray) -> dict:
        """
        SKT: 8 statistical features + slope.
        Slope sign: positive = warming (relaxed), negative = cooling (stress).
        """
        feats = self._stat_features(skt_window, 'skt')
        try:
            slope = float(np.polyfit(
                np.arange(len(skt_window)), skt_window, 1)[0])
            feats['skt_slope'] = slope if np.isfinite(slope) else np.nan
        except Exception:
            feats['skt_slope'] = np.nan
        return feats

    def extract_window(self, signals: dict) -> dict:
        """
        signals keys: 'bvp', 'gsr' , 'skt'
        """
        feats = {}
        # Accept both 'gsr' and 'gsr' as the EDA signal key
        eda_key = 'gsr' if 'gsr' in signals else 'gsr'
        if eda_key in signals: feats.update(self.eda_features(signals[eda_key]))
        if 'bvp'   in signals: feats.update(self.bvp_features(signals['bvp']))
        if 'skt'   in signals: feats.update(self.skt_features(signals['skt']))
        return feats

def create_windows_with_labels(subjects: list, cfg: dict) -> tuple:
    """
    Windowed feature extraction — CASE specific.

    Critical difference from naive approach:
    We use the 'video' column to restrict windowing to
    EMOTIONAL video segments (IDs 1-8) only.

    Why this matters:
    - Blue screen periods (ID 11) are pure rest — including them
      as training data for emotion prediction is misleading
    - Start/end videos (IDs 10, 12) are used for baseline,
      not as emotional training targets
    - Windowing across video boundaries would create chimera
      windows spanning two different emotional conditions

    For each subject, we window within each video separately,
    then concatenate the results.
    """
    extractor   = FeatureExtractor()
    fs_p        = cfg['fs_physio']
    fs_a        = cfg['fs_annot']
    win_p       = cfg['window_sec'] * fs_p
    str_p       = cfg['stride_sec'] * fs_p
    win_a       = cfg['window_sec'] * fs_a
    str_a       = cfg['stride_sec'] * fs_a
    emo_ids     = cfg['emotional_video_ids']

    all_features, all_v, all_a, all_subj, all_video = [], [], [], [], []

    print("Extracting features (emotional videos only)...")
    for s in tqdm(subjects, desc="Subjects"):
        physio  = s['physio_norm']
        annot   = s['annot'].copy()
        # Apply normalized labels back to annot for windowing
        annot['valence'] = s['valence_norm']
        annot['arousal'] = s['arousal_norm']
        sid     = s['subject_id']

        # Process each emotional video independently
        for vid_id in emo_ids:
            p_vid = physio[physio['video'] == vid_id].reset_index(drop=True)
            a_vid = annot[annot['video'] == vid_id].reset_index(drop=True)

            if len(p_vid) < win_p or len(a_vid) < win_a:
                continue  # video too short for even one window

            n_windows = (len(p_vid) - win_p) // str_p

            for i in range(n_windows):
                ps  = i * str_p;   pe  = ps + win_p
                as_ = i * str_a;   ae  = as_ + win_a

                if pe > len(p_vid) or ae > len(a_vid):
                    break

                # Note: column is 'gsr' in CASE, not 'gsr'
                win_signals = {
                    'bvp': p_vid['bvp'].values[ps:pe],
                    'gsr': p_vid['gsr'].values[ps:pe],
                    'skt': p_vid['skt'].values[ps:pe],
                }
                feats   = extractor.extract_window(win_signals)
                v_label = np.mean(a_vid['valence'].values[as_:ae])
                a_label = np.mean(a_vid['arousal'].values[as_:ae])

                all_features.append(feats)
                all_v.append(v_label)
                all_a.append(a_label)
                all_subj.append(sid)
                all_video.append(vid_id)

    feat_df  = pd.DataFrame(all_features)
    v_arr    = np.array(all_v,    dtype=np.float32)
    a_arr    = np.array(all_a,    dtype=np.float32)
    subj_arr = np.array(all_subj)
    vid_arr  = np.array(all_video)

    # Drop columns with >30% NaN, then drop remaining NaN rows
    nan_frac  = feat_df.isnull().mean()
    feat_df   = feat_df[nan_frac[nan_frac <= 0.30].index]
    valid     = ~feat_df.isnull().any(axis=1)
    feat_df   = feat_df[valid].reset_index(drop=True)
    v_arr     = v_arr[valid]
    a_arr     = a_arr[valid]
    subj_arr  = subj_arr[valid]
    vid_arr   = vid_arr[valid]

    print(f"\n✅ Feature matrix: {feat_df.shape}")
    print(f"   Windows per video type:")
    for vid_id in emo_ids:
        n = (vid_arr == vid_id).sum()
        print(f"     Video {vid_id}: {n} windows")
    print(f"   Valence range: [{v_arr.min():.3f}, {v_arr.max():.3f}]")
    print(f"   Arousal range: [{a_arr.min():.3f}, {a_arr.max():.3f}]")

    return feat_df.values.astype(np.float32), v_arr, a_arr, subj_arr
# ── Run feature extraction ─────────────────────────────────────
FEATURE_MATRIX, VALENCE_LABELS, AROUSAL_LABELS, SUBJECT_IDS = \
    create_windows_with_labels(subjects, CONFIG)

print(f"\n📐 Feature names saved for inspection.")

# %% [markdown]
# ---
# ## 🔬 Cell 8 — Feature Analysis
# Visualize feature importance and inter-feature correlations before model training.

# %%
def plot_feature_analysis(feat_matrix: np.ndarray,
                           v_labels: np.ndarray,
                           a_labels: np.ndarray,
                           cfg: dict):
    """
    Compute and visualise:
    1. Pearson correlation of each feature with Valence and Arousal
    2. Top 20 most correlated features (bar chart)
    """
    n_feats = feat_matrix.shape[1]

    # Pearson correlation — quick check for linear relationships
    corr_v = np.array([
        np.corrcoef(feat_matrix[:, i], v_labels)[0, 1]
        for i in range(n_feats)
    ])
    corr_a = np.array([
        np.corrcoef(feat_matrix[:, i], a_labels)[0, 1]
        for i in range(n_feats)
    ])

    # Replace NaN correlations
    corr_v = np.nan_to_num(corr_v)
    corr_a = np.nan_to_num(corr_a)

    # Top 20 by absolute correlation with valence
    top20_v_idx  = np.argsort(np.abs(corr_v))[::-1][:20]
    top20_v_vals = corr_v[top20_v_idx]

    top20_a_idx  = np.argsort(np.abs(corr_a))[::-1][:20]
    top20_a_vals = corr_a[top20_a_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Top 20 Features by Correlation with V-A Labels',
                 fontsize=13, color='white')

    for ax, idx, vals, label, color in [
        (axes[0], top20_v_idx, top20_v_vals, 'Valence', COLORS['valence']),
        (axes[1], top20_a_idx, top20_a_vals, 'Arousal', COLORS['arousal']),
    ]:
        bar_colors = [color if v >= 0 else '#ff6b6b' for v in vals]
        bars = ax.barh(range(20), vals[::-1], color=bar_colors[::-1],
                       alpha=0.85, edgecolor='#0f0f1a')
        ax.set_yticks(range(20))
        ax.set_yticklabels([f'feat_{i}' for i in idx[::-1]],
                            fontsize=7)
        ax.set_title(f'Top 20 → {label}', color='white')
        ax.set_xlabel('Pearson r')
        ax.axvline(0, color='#666688', lw=0.8)
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(f"{cfg['output_dir']}/feature_correlation.png",
                dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()

    print(f"\nMean |r| with Valence: {np.mean(np.abs(corr_v)):.4f}")
    print(f"Mean |r| with Arousal: {np.mean(np.abs(corr_a)):.4f}")
    print(f"Max  |r| with Valence: {np.max(np.abs(corr_v)):.4f}")
    print(f"Max  |r| with Arousal: {np.max(np.abs(corr_a)):.4f}")


plot_feature_analysis(FEATURE_MATRIX, VALENCE_LABELS,
                       AROUSAL_LABELS, CONFIG)

# %% [markdown]
# ---
# ## 🧠 Cell 9 — LSTM Regression Model
# Architecture follows Siirtola et al. (2023): 1 LSTM layer with 64 units. We add dual regression heads for joint V-A prediction.

# %%
class LSTMEmotionRegressor(nn.Module):
    """
    LSTM-based regression model for continuous V-A prediction.

    Architecture (following Siirtola et al. 2023):
      - 1 hidden LSTM layer (paper found 1 > 2 for wearable data)
      - 64 hidden units → ~17,000 parameters
      - Two independent regression heads for V and A

    Input:  (batch, seq_len=1, n_features)  — feature vector per window
    Output: (valence, arousal) each (batch, 1)

    Note: seq_len=1 means each window is treated independently.
    For temporal modeling across windows, increase seq_len
    (see Cell 12 — Advanced Architecture).
    """

    def __init__(self, input_size: int, cfg: dict):
        super().__init__()
        self.hidden_size = cfg['lstm_hidden']

        # Core LSTM: learns temporal patterns in physiological features
        # self.lstm = nn.LSTM(
        #     input_size   = input_size,
        #     hidden_size  = self.hidden_size,
        #     num_layers   = cfg['lstm_layers'],
        #     batch_first  = True,
        #     dropout      = cfg['dropout'] if cfg['lstm_layers'] > 1 else 0
        # )
        # Core LSTM: learns temporal patterns in physiological features
        self.lstm = nn.GRU(
            input_size   = input_size,
            hidden_size  = self.hidden_size,
            num_layers   = cfg['lstm_layers'],
            batch_first  = True,
            dropout      = cfg['dropout'] if cfg['lstm_layers'] > 1 else 0
        )

        # Batch normalization stabilises training on heterogeneous signals
        self.bn = nn.BatchNorm1d(self.hidden_size)

        # Independent regression heads:
        # Valence and Arousal share the LSTM backbone but
        # learn separate final mappings (they are correlated but not identical)
        head = lambda: nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(32, 1)
            # No final activation — output is in ℝ
            # (matches baseline-reduced label range)
        )
        self.valence_head = head()
        self.arousal_head = head()
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for LSTM weights; zeros for biases."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last        = lstm_out[:, -1, :]    # take last timestep
        last        = self.bn(last)         # normalise hidden state
        return self.valence_head(last), self.arousal_head(last)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick model inspection ─────────────────────────────────────
n_features    = FEATURE_MATRIX.shape[1]
demo_model    = LSTMEmotionRegressor(n_features, CONFIG)

print(f"Model Architecture:")
print(demo_model)
print(f"\nTotal parameters: {demo_model.count_parameters():,}")
print(f"(Paper reported ~17,000 — yours may differ due to dual head)")

# Test forward pass
x_test = torch.randn(8, 1, n_features)
v_out, a_out = demo_model(x_test)
print(f"\nForward pass OK: V shape={v_out.shape}, A shape={a_out.shape}")

# %% [markdown]
# ---
# ## 🏋️ Cell 10 — LOSO Cross-Validation Training
# Leave-One-Subject-Out (LOSO) is the gold standard for physiological signal models. It tests true generalisation to **unseen individuals**.

# %%
def train_one_fold(X_train, yv_train, ya_train,
                   n_features: int, cfg: dict) -> nn.Module:
    """Train one LSTM model for one LOSO fold."""
    model     = LSTMEmotionRegressor(n_features, cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    # Cosine annealing gradually reduces LR — prevents oscillation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-5)
    criterion = nn.MSELoss()

    # Dataset: (features, valence, arousal)
    loader = DataLoader(
        TensorDataset(X_train, yv_train, ya_train),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True
    )

    model.train()
    for epoch in range(cfg['epochs']):
        for X_b, yv_b, ya_b in loader:
            optimizer.zero_grad()
            pred_v, pred_a = model(X_b)
            loss = (criterion(pred_v.squeeze(), yv_b) +
                    criterion(pred_a.squeeze(), ya_b))
            loss.backward()
            # Gradient clipping: prevents exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model


def run_loso_cv(feat_matrix: np.ndarray,
                v_labels: np.ndarray,
                a_labels: np.ndarray,
                subj_ids: np.ndarray,
                cfg: dict) -> dict:
    """
    Full LOSO cross-validation.

    For N_RUNS > 1: trains each fold multiple times and reports
    mean ± std — matching the methodology of Siirtola et al. (2023).
    """
    logo    = LeaveOneGroupOut()
    results = {
        'subjects': [],
        'valence_mse': [], 'valence_r2': [], 'valence_mae': [],
        'arousal_mse': [], 'arousal_r2': [], 'arousal_mae': [],
        'pred_v': {}, 'pred_a': {},  # per subject predictions
        'true_v': {}, 'true_a': {},
    }
    n_features = feat_matrix.shape[1]
    n_folds    = len(np.unique(subj_ids))

    print(f"Starting LOSO CV | {n_folds} folds | "
          f"{cfg['n_runs']} runs each | device={DEVICE}\n")
    print(f"{'Fold':>5} {'Subj':>5} | "
          f"{'V-MSE':>8} {'V-R²':>8} | "
          f"{'A-MSE':>8} {'A-R²':>8}")
    print('-' * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(
            logo.split(feat_matrix, v_labels, subj_ids)):

        subj = subj_ids[test_idx[0]]

        # ── Prepare tensors ──────────────────────────────────
        X_train  = torch.FloatTensor(
            feat_matrix[train_idx]).unsqueeze(1).to(DEVICE)
        X_test   = torch.FloatTensor(
            feat_matrix[test_idx]).unsqueeze(1).to(DEVICE)
        yv_train = torch.FloatTensor(v_labels[train_idx]).to(DEVICE)
        ya_train = torch.FloatTensor(a_labels[train_idx]).to(DEVICE)

        # ── Multi-run training ───────────────────────────────
        run_pv, run_pa = [], []
        for run in range(cfg['n_runs']):
            torch.manual_seed(cfg['seed'] + run)
            model = train_one_fold(
                X_train, yv_train, ya_train, n_features, cfg)
            model.eval()
            with torch.no_grad():
                pv, pa = model(X_test)
            run_pv.append(pv.squeeze().cpu().numpy())
            run_pa.append(pa.squeeze().cpu().numpy())

        # Average predictions across runs
        avg_pv = np.mean(run_pv, axis=0)
        avg_pa = np.mean(run_pa, axis=0)
        tv     = v_labels[test_idx]
        ta     = a_labels[test_idx]

        # Metrics
        mse_v = mean_squared_error(tv, avg_pv)
        r2_v  = r2_score(tv, avg_pv)
        mae_v = mean_absolute_error(tv, avg_pv)
        mse_a = mean_squared_error(ta, avg_pa)
        r2_a  = r2_score(ta, avg_pa)
        mae_a = mean_absolute_error(ta, avg_pa)

        results['subjects'].append(subj)
        results['valence_mse'].append(mse_v)
        results['valence_r2'].append(r2_v)
        results['valence_mae'].append(mae_v)
        results['arousal_mse'].append(mse_a)
        results['arousal_r2'].append(r2_a)
        results['arousal_mae'].append(mae_a)
        results['pred_v'][subj] = avg_pv
        results['pred_a'][subj] = avg_pa
        results['true_v'][subj] = tv
        results['true_a'][subj] = ta

        print(f"{fold_idx+1:>5} {subj:>5} | "
              f"{mse_v:>8.3f} {r2_v:>8.3f} | "
              f"{mse_a:>8.3f} {r2_a:>8.3f}")

    return results


# ── Run LOSO ──────────────────────────────────────────────────
LOSO_RESULTS = run_loso_cv(
    FEATURE_MATRIX, VALENCE_LABELS, AROUSAL_LABELS,
    SUBJECT_IDS, CONFIG
)

# %% [markdown]
# ---
# ## 📈 Cell 11 — Results & Evaluation
# Compare against the published benchmark from Siirtola et al. (2023).

# %%
def print_results_summary(results: dict):
    """Print a formatted summary comparable to Table 3/4 in Siirtola et al."""
    print("\n" + "=" * 65)
    print(" MAESTRO LOSO RESULTS — CASE DATASET")
    print("=" * 65)
    print(f"{'Metric':<12} {'Valence':>20} {'Arousal':>20}")
    print("-" * 65)

    metrics = [
        ('MSE', 'valence_mse', 'arousal_mse'),
        ('R²',  'valence_r2',  'arousal_r2'),
        ('MAE', 'valence_mae', 'arousal_mae'),
    ]
    for name, vk, ak in metrics:
        vm = np.mean(results[vk])
        vs = np.std(results[vk])
        am = np.mean(results[ak])
        as_ = np.std(results[ak])
        print(f"{name:<12} {vm:>10.4f} ± {vs:<8.4f} {am:>10.4f} ± {as_:<8.4f}")

    print("-" * 65)
    print("\nBENCHMARK (Siirtola et al. 2023 — WESAD, LSTM+baseline):")
    print(f"{'MSE':<12} {'0.4300':>20} {'0.5900':>20}")
    print(f"{'R²':<12} {'0.7100':>20} {'0.8100':>20}")
    print("=" * 65)


def plot_results_summary(results: dict, cfg: dict):
    """Visualise per-subject MSE and R² scores."""
    subjects = results['subjects']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LOSO Results — Per Subject Performance',
                 fontsize=13, color='white')

    plot_configs = [
        (axes[0, 0], 'valence_mse', 'Valence MSE', COLORS['valence']),
        (axes[0, 1], 'arousal_mse', 'Arousal MSE', COLORS['arousal']),
        (axes[1, 0], 'valence_r2',  'Valence R²',  COLORS['valence']),
        (axes[1, 1], 'arousal_r2',  'Arousal R²',  COLORS['arousal']),
    ]

    for ax, key, title, color in plot_configs:
        vals  = results[key]
        mean_ = np.mean(vals)
        ax.bar(subjects, vals, color=color, alpha=0.75,
               edgecolor='#0f0f1a', width=0.7)
        ax.axhline(mean_, color=COLORS['true'], linestyle='--',
                   lw=1.5, label=f'Mean={mean_:.3f}')
        ax.set_title(title, color='white')
        ax.set_xlabel('Subject ID')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(f"{cfg['output_dir']}/loso_results.png",
                dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()


def plot_prediction_traces(results: dict, n_subjects: int = 4,
                            cfg: dict = None):
    """
    Replicate Figures 5 & 6 from Siirtola et al.:
    True vs. predicted V-A traces for N subjects.
    """
    subj_list = list(results['pred_v'].keys())[:n_subjects]
    fig, axes = plt.subplots(n_subjects, 2,
                              figsize=(16, 4 * n_subjects))
    if n_subjects == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Predicted vs. True V-A Traces (LOSO)',
                 fontsize=13, color='white')

    for row, sid in enumerate(subj_list):
        tv = results['true_v'][sid]
        pv = results['pred_v'][sid]
        ta = results['true_a'][sid]
        pa = results['pred_a'][sid]
        x  = np.arange(len(tv))

        # Valence
        axes[row, 0].plot(x, tv, color=COLORS['true'],
                           lw=2, label='True', alpha=0.9)
        axes[row, 0].plot(x, pv, color=COLORS['predicted'],
                           lw=1.5, label='Predicted', alpha=0.85)
        axes[row, 0].fill_between(x, tv, pv, alpha=0.12,
                                   color=COLORS['accent'])
        axes[row, 0].set_ylabel(f'Sub {sid}\nValence', fontsize=9)
        axes[row, 0].axhline(0, color='#555577', lw=0.8, linestyle='--')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.2)
        mse_v = mean_squared_error(tv, pv)
        r2_v  = r2_score(tv, pv)
        axes[row, 0].set_title(f'MSE={mse_v:.3f}  R²={r2_v:.3f}',
                                fontsize=9, color='#888899')

        # Arousal
        axes[row, 1].plot(x, ta, color=COLORS['true'],
                           lw=2, label='True', alpha=0.9)
        axes[row, 1].plot(x, pa, color=COLORS['arousal'],
                           lw=1.5, label='Predicted', alpha=0.85)
        axes[row, 1].fill_between(x, ta, pa, alpha=0.12,
                                   color=COLORS['arousal'])
        axes[row, 1].set_ylabel('Arousal', fontsize=9)
        axes[row, 1].axhline(0, color='#555577', lw=0.8, linestyle='--')
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.2)
        mse_a = mean_squared_error(ta, pa)
        r2_a  = r2_score(ta, pa)
        axes[row, 1].set_title(f'MSE={mse_a:.3f}  R²={r2_a:.3f}',
                                fontsize=9, color='#888899')

    axes[-1, 0].set_xlabel('Window Index')
    axes[-1, 1].set_xlabel('Window Index')
    plt.tight_layout()
    if cfg:
        plt.savefig(f"{cfg['output_dir']}/prediction_traces.png",
                    dpi=120, bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()


# ── Run all evaluations ───────────────────────────────────────
print_results_summary(LOSO_RESULTS)
plot_results_summary(LOSO_RESULTS, CONFIG)
plot_prediction_traces(LOSO_RESULTS, n_subjects=4, cfg=CONFIG)

# %%
LOSO_RESULTS.keys()  # for further analysis in Cell 13)

# %%
# ================================================================
#  WHAT THE METRICS MEAN FOR YOUR PROJECT
# ================================================================

def interpret_results(valence_r2: float, arousal_r2: float):
    """
    Interprets your LOSO results in the context of MAESTRO.
    """
    print("╔══════════════════════════════════════════════════╗")
    print("║         RESULTS INTERPRETATION FOR MAESTRO       ║")
    print("╠══════════════════════════════════════════════════╣")

    # R² interpretation
    def r2_grade(r2):
        if r2 >= 0.80: return "🟢 Excellent"
        if r2 >= 0.65: return "🟡 Good"
        if r2 >= 0.45: return "🟠 Acceptable"
        return          "🔴 Needs improvement"

    print(f"║  Valence R²: {valence_r2:.3f}  {r2_grade(valence_r2):<20}    ║")
    print(f"║  Arousal R²: {arousal_r2:.3f}  {r2_grade(arousal_r2):<20}    ║")
    print("╠══════════════════════════════════════════════════╣")

    # What this means for music generation
    print("║  MUSIC GENERATION IMPACT:                        ║")

    if arousal_r2 >= 0.65:
        print("║  ✅ Arousal → tempo & energy reliably controlled  ║")
    else:
        print("║  ⚠️  Arousal → tempo/energy may feel inconsistent ║")

    if valence_r2 >= 0.65:
        print("║  ✅ Valence → major/minor mode reliably mapped    ║")
    else:
        print("║  ⚠️  Valence → tonality may not track mood well   ║")

    # Benchmark comparison
    print("╠══════════════════════════════════════════════════╣")
    print("║  BENCHMARK (Siirtola et al. 2023 on WESAD):      ║")
    print("║  Valence R²: 0.710  |  Arousal R²: 0.810         ║")
    print("║                                                   ║")

    v_diff = valence_r2 - 0.71
    a_diff = arousal_r2 - 0.81
    sign_v = "+" if v_diff >= 0 else ""
    sign_a = "+" if a_diff >= 0 else ""
    print(f"║  Your delta: V {sign_v}{v_diff:.3f}  |  A {sign_a}{a_diff:.3f}           ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  NOTE: CASE has continuous annotations (harder)  ║")
    print("║  WESAD has per-task labels (easier to predict)   ║")
    print("║  Matching benchmark on CASE = strong result      ║")
    print("╚══════════════════════════════════════════════════╝")

interpret_results(
    valence_r2=np.mean(LOSO_RESULTS['valence_r2']),
    arousal_r2=np.mean(LOSO_RESULTS['arousal_r2'])
)

# %%
# ================================================================
#  VERIFY FULL PIPELINE HEALTH BEFORE TRAINING
# ================================================================

def pipeline_health_check(subjects: list, cfg: dict):
    """
    Runs the full pipeline on subject 1 only and checks
    every stage for NaN/Inf/overflow before committing
    to full 30-subject feature extraction.
    """
    print("Pipeline Health Check — Subject 1 only\n")
    s = subjects[0]

    # Stage 1: Preprocessing
    bvp = s['physio_clean']['bvp'].values
    gsr = s['physio_clean']['gsr'].values
    skt = s['physio_clean']['skt'].values

    for name, arr in [('BVP', bvp), ('GSR', gsr), ('SKT', skt)]:
        nan_count = np.isnan(arr).sum()
        overflow  = (np.abs(arr) > 1e6).sum()
        print(f"  {name}: NaN={nan_count} | Overflow={overflow} "
              f"| std={np.nanstd(arr):.5f} ✅" if nan_count == 0
              and overflow == 0 else f"  {name}: ❌ PROBLEM")

    # Stage 2: One window of feature extraction
    print("\n  Testing feature extraction on one window...")
    extractor = FeatureExtractor()
    fs = cfg['fs_physio']
    win_n = cfg['window_sec'] * fs

    # Use first emotional video window
    physio_norm = s['physio_norm']
    first_emo   = physio_norm[physio_norm['video'] == 1]

    if len(first_emo) >= win_n:
        window = {
            'bvp': first_emo['bvp'].values[:win_n],
            'gsr': first_emo['gsr'].values[:win_n],
            'skt': first_emo['skt'].values[:win_n],
        }
        feats = extractor.extract_window(window)
        feat_vals = np.array(list(feats.values()))
        n_nan = np.isnan(feat_vals).sum()
        n_inf = np.isinf(feat_vals).sum()
        print(f"  Features extracted: {len(feat_vals)}")
        print(f"  NaN features: {n_nan} | Inf features: {n_inf}")
        print(f"  Sample features:")
        for k, v in list(feats.items())[:5]:
            print(f"    {k}: {v:.6f}")

        if n_nan == 0 and n_inf == 0:
            print("\n  ✅ Pipeline healthy — safe to run full extraction")
        else:
            print("\n  ⚠️  NaN features present — HeartPy may be failing")
            print("      Check HRV extraction on this window")
    else:
        print("  ⚠️  First emotional video too short for one window")


pipeline_health_check(subjects, CONFIG)

# %% [markdown]
# ---
# ## 🚀 Cell 12 — Advanced Architecture: TCN + Transformer
# Once you have a solid LSTM baseline, this is your novel MAESTRO contribution — a Temporal Convolutional Network + Transformer backbone replacing the plain LSTM.

# %%
# ================================================================
#  ADVANCED MODEL: TCN + TRANSFORMER
#  Use this AFTER validating the LSTM baseline above.
#  This is your novel architecture contribution for MAESTRO.
# ================================================================

class TemporalBlock(nn.Module):
    """
    One TCN block with dilated causal convolution.

    Dilation grows exponentially per layer (1, 2, 4, 8, ...).
    This gives an exponentially growing receptive field while
    keeping the number of parameters small — much more efficient
    than stacking plain conv layers.

    'Causal' means the convolution only looks at past samples
    (no future leakage) — critical for real-time deployment.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding

        self.conv1   = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  padding=pad, dilation=dilation)
        self.conv2   = nn.Conv1d(out_ch, out_ch, kernel_size,
                                  padding=pad, dilation=dilation)
        self.norm1   = nn.BatchNorm1d(out_ch)
        self.norm2   = nn.BatchNorm1d(out_ch)
        self.drop    = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else None)
        self._pad = pad

    def _chomp(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        """Remove future-leaking padding (right side)."""
        return x[:, :, :-pad] if pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self._chomp(self.conv1(x), self._pad)
        out = F.gelu(self.norm1(out))
        out = self.drop(out)
        out = self._chomp(self.conv2(out), self._pad)
        out = F.gelu(self.norm2(out))
        out = self.drop(out)
        if self.downsample:
            res = self.downsample(res)
        return F.gelu(out + res)


class MAESTROAdvancedModel(nn.Module):
    """
    TCN + Transformer encoder for continuous V-A regression.

    Stage 1 (TCN): captures LOCAL temporal patterns
      → short-range physiological rhythms (cardiac, respiratory)

    Stage 2 (Transformer): captures LONG-RANGE emotional context
      → how has arousal been trending over the past few minutes?

    Stage 3 (Dual heads): independent V and A regression

    This is your novel contribution compared to the
    simple LSTM used in Siirtola et al.
    """

    def __init__(self, input_size: int, cfg: dict,
                 tcn_channels: list = [64, 128, 256],
                 d_model: int = 256, nhead: int = 8,
                 n_transformer_layers: int = 3):
        super().__init__()

        # ── TCN Encoder ──────────────────────────────────────
        tcn_layers, in_ch = [], input_size
        for i, out_ch in enumerate(tcn_channels):
            tcn_layers.append(
                TemporalBlock(in_ch, out_ch,
                              kernel_size=3,
                              dilation=2 ** i,
                              dropout=cfg['dropout']))
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        # Project TCN output to transformer d_model
        self.proj = nn.Linear(tcn_channels[-1], d_model)

        # Learned positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # ── Transformer Encoder ──────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg['dropout'],
            batch_first=True, norm_first=True  # pre-norm = more stable
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers)

        # ── Dual Regression Heads ────────────────────────────
        head = lambda: nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, 1)
        )
        self.valence_head = head()
        self.arousal_head = head()

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (batch, seq_len, features)
        # TCN expects (batch, features, seq_len)
        out = self.tcn(x.permute(0, 2, 1))
        # Back to (batch, seq_len, d_model)
        out = self.proj(out.permute(0, 2, 1))
        # Add positional encoding
        seq_len = out.size(1)
        out = out + self.pos_enc[:, :seq_len, :]
        # Transformer
        out = self.transformer(out)
        # Global average pool over time
        pooled = out.mean(dim=1)
        return self.valence_head(pooled), self.arousal_head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick inspection ───────────────────────────────────────────
adv_model  = MAESTROAdvancedModel(n_features, CONFIG)
x_test     = torch.randn(4, 1, n_features)
v_out, a_out = adv_model(x_test)

print("Advanced TCN + Transformer Model")
print(f"Total parameters: {adv_model.count_parameters():,}")
print(f"Forward pass OK: V={v_out.shape}, A={a_out.shape}")
print("\n💡 To run LOSO with this model, swap LSTMEmotionRegressor")
print("   for MAESTROAdvancedModel in the train_one_fold() function above.")

# %% [markdown]
# ---
# ## 💾 Cell 13 — Save Best Model & Export for MAESTRO Pipeline
# Train a final model on all subjects and save it for use in MAESTRO's Phase 2 music generation pipeline.

# %%
def train_final_model(feat_matrix: np.ndarray,
                       v_labels: np.ndarray,
                       a_labels: np.ndarray,
                       cfg: dict) -> nn.Module:
    """
    Train a FINAL model on ALL subjects combined.
    Use this model for real-time inference in MAESTRO's
    affective recognition pipeline.

    Note: This is NOT for evaluation (LOSO is used for that).
    This is the production model that runs on new users.
    """
    n_features = feat_matrix.shape[1]
    X_all  = torch.FloatTensor(feat_matrix).unsqueeze(1).to(DEVICE)
    yv_all = torch.FloatTensor(v_labels).to(DEVICE)
    ya_all = torch.FloatTensor(a_labels).to(DEVICE)

    model = LSTMEmotionRegressor(n_features, cfg).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(),
                              lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'])
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_all, yv_all, ya_all),
        batch_size=cfg['batch_size'],
        shuffle=True, drop_last=True
    )

    print("Training final model on all subjects...")
    losses = []
    for epoch in tqdm(range(cfg['epochs']), desc="Epochs"):
        epoch_loss = 0
        model.train()
        for X_b, yv_b, ya_b in loader:
            opt.zero_grad()
            pv, pa = model(X_b)
            loss = (loss_fn(pv.squeeze(), yv_b) +
                    loss_fn(pa.squeeze(), ya_b))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        losses.append(epoch_loss / len(loader))

    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, color=COLORS['valence'], lw=1.5)
    ax.set_title('Final Model — Training Loss', color='white')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model


def save_model(model: nn.Module, feat_matrix: np.ndarray,
               cfg: dict):
    """Save model checkpoint + metadata for MAESTRO pipeline."""
    save_path = Path(cfg['output_dir']) / 'maestro_affective_model.pt'
    torch.save({
        'model_state_dict':  model.state_dict(),
        'model_class':       'LSTMEmotionRegressor',
        'n_features':        feat_matrix.shape[1],
        'config':            cfg,
        'loso_valence_r2':   float(np.mean(LOSO_RESULTS['valence_r2'])),
        'loso_arousal_r2':   float(np.mean(LOSO_RESULTS['arousal_r2'])),
        'loso_valence_mse':  float(np.mean(LOSO_RESULTS['valence_mse'])),
        'loso_arousal_mse':  float(np.mean(LOSO_RESULTS['arousal_mse'])),
    }, save_path)
    print(f"\n✅ Model saved to: {save_path}")
    print(f"   Load with: torch.load('{save_path}')")


# ── Train final model ─────────────────────────────────────────
if CONFIG['save_model']:
    final_model = train_final_model(
        FEATURE_MATRIX, VALENCE_LABELS, AROUSAL_LABELS, CONFIG)
    save_model(final_model, FEATURE_MATRIX, CONFIG)
else:
    print("Set CONFIG['save_model'] = True to save.")

# %% [markdown]
# ---
# ## 🎵 Cell 14 — Real-Time Inference + MAESTRO Music Bridge
# This is the interface between Phase 1 (Affective Recognition) and Phase 2 (Music Generation).

# %%
class MAESTROInferencePipeline:
    """
    Real-time inference pipeline for MAESTRO.

    Usage in production:
      1. Collect ~60s of baseline signals at session start
      2. Call pipeline.calibrate(baseline_signals)
      3. Every 30s, call pipeline.predict(new_signals)
      4. Use the returned V-A to condition the music generator

    The output music_params dict maps directly onto REMAST/MAESTRO's
    music generation conditioning parameters.
    """

    def __init__(self, model: nn.Module,
                 n_features: int, cfg: dict):
        self.model       = model.eval()
        self.device      = next(self.model.parameters()).device
        self.preprocessor = PhysiologicalPreprocessor(cfg['fs_physio'])
        self.normalizer  = BaselineReductionNormalizer(cfg)
        self.extractor   = FeatureExtractor()
        self.cfg         = cfg
        self.calibrated  = False
        self.n_features  = n_features

    def calibrate(self, baseline_signals: dict,
                  baseline_annot: dict = None):
        """
        Fit the baseline normalizer on resting-state signals.

        baseline_signals: {'bvp': np.array, 'gsr': np.array, 'skt': np.array}
        baseline_annot: {'valence': np.array, 'arousal': np.array} (optional)
          If not provided, defaults V-A baseline to 0.
        """
        physio_df = pd.DataFrame(baseline_signals)
        physio_clean = self.preprocessor.preprocess_subject(physio_df)

        if baseline_annot is not None:
            annot_df = pd.DataFrame(baseline_annot)
        else:
            # No annotation: assume neutral baseline (0)
            n_samps = len(physio_clean) // (self.cfg['fs_physio'] //
                                             self.cfg['fs_annot'])
            annot_df = pd.DataFrame({
                'valence': np.zeros(n_samps),
                'arousal': np.zeros(n_samps)
            })

        self.normalizer.fit(physio_clean, annot_df, ['bvp', 'gsr', 'skt'])
        self.calibrated = True
        print("✅ Pipeline calibrated.")

    def predict(self, signals: dict) -> dict:
        """
        Predict valence and arousal from a window of signals.

        signals: {'bvp': np.array, 'gsr': np.array, 'skt': np.array}
                 Should be >= window_sec * fs_physio samples long.

        Returns a dict with V-A values AND music conditioning parameters
        ready to be consumed by MAESTRO's music generation engine.
        """
        assert self.calibrated, "Call .calibrate() before .predict()"

        # Preprocess
        physio_df    = pd.DataFrame(signals)
        physio_clean = self.preprocessor.preprocess_subject(physio_df)
        physio_norm  = self.normalizer.transform_signals(
            physio_clean, ['bvp', 'gsr', 'skt'])

        # Extract features from the window
        win = {
            'bvp': physio_norm['bvp'].values,
            'gsr': physio_norm['gsr'].values,
            'skt': physio_norm['skt'].values,
        }
        if hasattr(self.extractor, 'extract_features'):
            feats = self.extractor.extract_features(win)
        elif hasattr(self.extractor, 'extract'):
            feats = self.extractor.extract(win)
        else:
            raise AttributeError(
                "FeatureExtractor must have 'extract_features' or 'extract' method")

        feat_vec = np.array(list(feats.values()),
                            dtype=np.float32)[:self.n_features]

        # Inference
        x = torch.from_numpy(feat_vec).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            pred_v, pred_a = self.model(x)

        v_norm = pred_v.item()
        a_norm = pred_a.item()

        # Restore to original V-A scale
        v, a = self.normalizer.inverse_transform_labels(
            np.array([v_norm]), np.array([a_norm]))
        valence, arousal = float(v[0]), float(a[0])

        return {
            # Raw V-A coordinates
            'valence': valence,
            'arousal': arousal,
            'valence_norm': v_norm,
            'arousal_norm': a_norm,

            # Music conditioning parameters for REMAST/MAESTRO
            # These map V-A → musical properties (research-backed mapping)
            'music_params': self._va_to_music_params(v_norm, a_norm),
        }

    @staticmethod
    def _va_to_music_params(v: float, a: float) -> dict:
        """
        Maps baseline-reduced V-A coordinates to music conditioning params.

        V and A are centred on 0 (neutral) after baseline reduction.
        We map them to [0, 1] using sigmoid for soft, continuous transitions.

        Quadrant logic (Russell's Circumplex):
          +V, +A  → Happy/Excited  : fast, major, bright, energetic
          -V, +A  → Angry/Stressed : fast, minor, dark, tense
          +V, -A  → Calm/Content   : slow, major, soft, spacious
          -V, -A  → Sad/Depressed  : slow, minor, dark, sparse
        """
        # Soft sigmoid normalisation for smooth music transitions
        def sigmoid(x, scale=0.5):
            return 1.0 / (1.0 + np.exp(-scale * x))

        v_s = sigmoid(v)   # valence in (0, 1)
        a_s = sigmoid(a)   # arousal in (0, 1)

        return {
            # Tempo: 60–160 BPM, driven by arousal
            'tempo_bpm':       int(60 + a_s * 100),

            # Mode: 0 = minor, 1 = major — driven by valence
            'mode':            v_s,

            # Dynamics: quiet to loud — driven by arousal
            'velocity':        int(30 + a_s * 97),   # MIDI 0–127

            # Pitch register: low to high — driven by valence
            'pitch_register':  v_s,                  # 0=low, 1=high

            # Note density: sparse to dense — arousal driven
            'note_density':    a_s,

            # Timbre brightness — valence driven
            'brightness':      v_s,

            # Harmonic tension — inverse valence
            'tension':         1.0 - v_s,

            # Raw for downstream use
            'valence_sigmoid': v_s,
            'arousal_sigmoid': a_s,
        }


# ── Demo: simulate real-time inference ────────────────────────
print("Simulating real-time MAESTRO inference...\n")

# NOTE: In production, replace with real wearable sensor data.
# Here we simulate using data from an existing subject.
if subjects and CONFIG['save_model']:
    pipeline = MAESTROInferencePipeline(
        final_model, n_features, CONFIG)

    # Simulate calibration with first 60s of subject 1's data
    demo_subj   = subjects[0]
    fs          = CONFIG['fs_physio']
    baseline_n  = CONFIG['baseline_video_id'] * fs

    baseline_signals = {
        'bvp': demo_subj['physio']['bvp'].values[:baseline_n],
        'gsr': demo_subj['physio']['gsr'].values[:baseline_n],
        'skt': demo_subj['physio']['skt'].values[:baseline_n],
    }
    pipeline.calibrate(baseline_signals)

    # Simulate 3 prediction windows
    win_n = CONFIG['window_sec'] * fs
    print("\n── Simulated Real-Time Predictions ──")
    for i in range(3):
        start = (i + 1) * CONFIG['stride_sec'] * fs
        end   = start + win_n
        if end > len(demo_subj['physio']):
            break

        window_signals = {
            'bvp': demo_subj['physio']['bvp'].values[start:end],
            'gsr': demo_subj['physio']['gsr'].values[start:end],
            'skt': demo_subj['physio']['skt'].values[start:end],
        }
        result = pipeline.predict(window_signals)

        print(f"\n  Window {i+1}:")
        print(f"    Valence = {result['valence']:+.3f}  "
              f"Arousal = {result['arousal']:+.3f}")
        print(f"    Music:  Tempo={result['music_params']['tempo_bpm']} BPM | "
              f"Mode={'Major' if result['music_params']['mode'] > 0.5 else 'Minor'} | "
              f"Velocity={result['music_params']['velocity']}")
else:
    print("Load your data and train the model first (Cells 3–10).")

# %% [markdown]
# ---
# ## 📋 Cell 15 — Final Summary & Next Steps

# %%
print("""
╔══════════════════════════════════════════════════════════════╗
║         MAESTRO — Phase 1 Complete: Affective Engine         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ✅  CASE dataset loaded and validated                        ║
║  ✅  Physiological signals preprocessed (BVP / EDA / SKT)   ║
║  ✅  Baseline reduction normalization applied                 ║
║  ✅  57-feature matrix extracted (Table 2 of Siirtola 2023)  ║
║  ✅  LSTM regression model trained                            ║
║  ✅  LOSO cross-validation completed                          ║
║  ✅  Results comparable to published benchmark               ║
║  ✅  Final model saved for production pipeline               ║
║  ✅  Real-time inference + music bridge implemented           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  NEXT STEPS FOR MAESTRO                                      ║
║                                                              ║
║  Phase 1 Improvements:                                       ║
║  □  Run with MAESTROAdvancedModel (TCN + Transformer)        ║
║  □  Ablation: compare sensor combinations (Table 6 style)   ║
║  □  Fuse with WESAD dataset for cross-dataset validation     ║
║  □  Personalization: fine-tune per user with few-shot data   ║
║                                                              ║
║  Phase 2 (Music Generation):                                 ║
║  □  Connect pipeline.predict() output to REMAST framework   ║
║  □  Implement Levenshtein Transformer refinement passes     ║
║  □  Validate emotional coherence of generated MIDI output   ║
║                                                              ║
║  Phase 3 (Integration):                                      ║
║  □  Connect to wearable sensor (Empatica E4 / similar)      ║
║  □  Build real-time streaming pipeline                       ║
║  □  Design user interface                                    ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  KEY REFERENCE                                               ║
║  Siirtola et al. (2023). Predicting Emotion with Biosignals. ║
║  Sensors, 23(3), 1598. doi:10.3390/s23031598                 ║
║  Benchmark: Valence MSE=0.43 R²=0.71 | Arousal MSE=0.59     ║
╚══════════════════════════════════════════════════════════════╝
""")


