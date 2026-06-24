import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
import neurokit2 as nk
import heartpy as hp

class PhysiologicalPreprocessor:
    """
    Numerically stable physiological signal preprocessor.
    Uses output='sos' (Second-Order Sections) in butter() to avoid overflow at high fs.
    """
    def __init__(self, fs: int = 1000):
        self.fs = fs

    def _bandpass(self, sig: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
        nyq = self.fs / 2
        sos = signal.butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
        return signal.sosfiltfilt(sos, sig)

    def _lowpass(self, sig: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        nyq = self.fs / 2
        sos = signal.butter(order, cutoff / nyq, btype='low', output='sos')
        return signal.sosfiltfilt(sos, sig)

    def _notch(self, sig: np.ndarray, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
        b, a = signal.iirnotch(freq, Q, fs=self.fs)
        return signal.filtfilt(b, a, sig)

    def preprocess_bvp(self, bvp: np.ndarray) -> np.ndarray:
        return self._bandpass(bvp.astype(np.float64), 0.5, 3.5)

    def preprocess_gsr(self, gsr: np.ndarray) -> np.ndarray:
        return self._lowpass(gsr.astype(np.float64), 3.0)

    def preprocess_skt(self, skt: np.ndarray) -> np.ndarray:
        from scipy.signal import medfilt
        return medfilt(skt.astype(np.float64), kernel_size=51)

    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        ecg64 = ecg.astype(np.float64)
        ecg_notch = self._notch(ecg64, freq=50.0)
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

            n_nan = np.isnan(result).sum()
            n_inf = np.isinf(result).sum()
            absmax = np.abs(result).max()

            if n_nan > 0 or n_inf > 0 or absmax > 1e6:
                print(f"  ❌ {col}: overflow detected (NaN={n_nan}, Inf={n_inf}, max={absmax:.2e})")
            else:
                df[col] = result

        return df


class BaselineReductionNormalizer:
    """
    Baseline reduction normalization for CASE dataset.
    Subtracts per-subject baseline mean for each signal to remove inter-subject differences.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.baseline_vid = cfg.get('baseline_video_id', 10)
        self.signal_baselines = {}
        self.label_baseline_v = None
        self.label_baseline_a = None

    def fit(self, physio_clean: pd.DataFrame, annot: pd.DataFrame, signal_cols: list) -> 'BaselineReductionNormalizer':
        if 'video' in physio_clean.columns:
            p_base = physio_clean[physio_clean['video'] == self.baseline_vid]
            a_base = annot[annot['video'] == self.baseline_vid]
        else:
            # If no video column (e.g. inference), assume the entire passed df is the baseline
            p_base = physio_clean
            a_base = annot

        if len(p_base) == 0:
            raise ValueError("No baseline video found for this subject/session!")

        for col in signal_cols:
            if col in p_base.columns:
                self.signal_baselines[col] = p_base[col].mean()

        self.label_baseline_v = a_base['valence'].mean() if 'valence' in a_base.columns else 0.0
        self.label_baseline_a = a_base['arousal'].mean() if 'arousal' in a_base.columns else 0.0

        return self

    def transform_signals(self, physio_clean: pd.DataFrame, signal_cols: list) -> pd.DataFrame:
        df = physio_clean.copy()
        for col in signal_cols:
            if col in self.signal_baselines:
                df[col] = df[col] - self.signal_baselines[col]
        return df

    def transform_labels(self, valence: np.ndarray, arousal: np.ndarray) -> tuple:
        return (valence - self.label_baseline_v, arousal - self.label_baseline_a)

    def inverse_transform_labels(self, v_norm: np.ndarray, a_norm: np.ndarray) -> tuple:
        return (v_norm + self.label_baseline_v, a_norm + self.label_baseline_a)


class FeatureExtractor:
    """Extracts the complete feature set from Siirtola et al. (Table 2)"""
    FS = 1000

    STAT_FUNS = {
        'mean':   np.mean,
        'std':    np.std,
        'min':    np.min,
        'max':    np.max,
        'median': np.median,
        'range':  lambda x: np.max(x) - np.min(x) if len(x) > 0 else np.nan,
        'p5':     lambda x: np.percentile(x, 5) if len(x) > 0 else np.nan,
        'p25':    lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan,
        'p75':    lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan,
        'p95':    lambda x: np.percentile(x, 95) if len(x) > 0 else np.nan,
    }

    def _stat_features(self, arr: np.ndarray, prefix: str) -> dict:
        feats = {}
        for name, fn in self.STAT_FUNS.items():
            try:
                val = float(fn(arr))
                feats[f'{prefix}_{name}'] = val if np.isfinite(val) else np.nan
            except Exception:
                feats[f'{prefix}_{name}'] = np.nan
        return feats

    def eda_features(self, eda_window: np.ndarray) -> dict:
        feats = self._stat_features(eda_window, 'gsr')
        try:
            processed, _ = nk.eda_process(eda_window, sampling_rate=self.FS)
            feats.update(self._stat_features(processed['EDA_Phasic'].values, 'eda_phasic'))
            feats.update(self._stat_features(processed['EDA_Tonic'].values, 'eda_tonic'))
        except Exception:
            for sfx in self.STAT_FUNS:
                feats[f'eda_phasic_{sfx}'] = np.nan
                feats[f'eda_tonic_{sfx}']  = np.nan
        return feats

    def bvp_features(self, bvp_window: np.ndarray) -> dict:
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
            }
            for feat_name, hp_key in hrv_keys.items():
                val = m.get(hp_key, np.nan)
                feats[f'hrv_{feat_name}'] = float(val) if np.isfinite(float(val)) else np.nan
        except Exception:
            for feat_name in ['hr_mean','nni_mean','nni_std','rmssd','pnn50','breathing_rate']:
                feats[f'hrv_{feat_name}'] = np.nan
        return feats

    def skt_features(self, skt_window: np.ndarray) -> dict:
        feats = self._stat_features(skt_window, 'skt')
        try:
            slope = float(np.polyfit(np.arange(len(skt_window)), skt_window, 1)[0])
            feats['skt_slope'] = slope if np.isfinite(slope) else np.nan
        except Exception:
            feats['skt_slope'] = np.nan
        return feats

    def extract_features(self, signals: dict) -> dict:
        feats = {}
        eda_key = 'gsr' if 'gsr' in signals else 'gsr'
        if eda_key in signals: feats.update(self.eda_features(signals[eda_key]))
        if 'bvp'   in signals: feats.update(self.bvp_features(signals['bvp']))
        if 'skt'   in signals: feats.update(self.skt_features(signals['skt']))
        return feats


class MAESTROInferencePipeline:
    """
    Real-time inference pipeline for MAESTRO.
    """
    def __init__(self, model_v: nn.Module, model_a: nn.Module, n_features: int, cfg: dict):
        self.model_v      = model_v.eval()
        self.model_a      = model_a.eval()
        self.device       = next(self.model_v.parameters()).device
        self.preprocessor = PhysiologicalPreprocessor(cfg.get('fs_physio', 1000))
        self.normalizer   = BaselineReductionNormalizer(cfg)
        self.extractor    = FeatureExtractor()
        self.cfg          = cfg
        self.calibrated   = False
        self.n_features   = n_features

    def calibrate(self, baseline_signals: dict, baseline_annot: dict = None):
        physio_df = pd.DataFrame(baseline_signals)
        physio_clean = self.preprocessor.preprocess_subject(physio_df)

        if baseline_annot is not None:
            annot_df = pd.DataFrame(baseline_annot)
        else:
            fs_physio = self.cfg.get('fs_physio', 1000)
            fs_annot = self.cfg.get('fs_annot', 20)
            neutral = self.cfg.get('label_neutral', 5.0)
            n_samps = len(physio_clean) // (fs_physio // fs_annot)
            annot_df = pd.DataFrame({
                'valence': np.full(n_samps, neutral),
                'arousal': np.full(n_samps, neutral)
            })

        self.normalizer.fit(physio_clean, annot_df, ['bvp', 'gsr', 'skt'])
        self.calibrated = True

    def predict(self, signals: dict) -> dict:
        assert self.calibrated, "Call .calibrate() before .predict()"

        physio_df    = pd.DataFrame(signals)
        physio_clean = self.preprocessor.preprocess_subject(physio_df)
        physio_norm  = self.normalizer.transform_signals(physio_clean, ['bvp', 'gsr', 'skt'])

        win = {
            'bvp': physio_norm['bvp'].values,
            'gsr': physio_norm['gsr'].values,
            'skt': physio_norm['skt'].values,
        }
        
        feats = self.extractor.extract_features(win)
        feat_vec = np.array(list(feats.values()), dtype=np.float32)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0)[:self.n_features]

        x = torch.from_numpy(feat_vec).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_v = self.model_v(x)
            pred_a = self.model_a(x)

        v_norm = pred_v.item()
        a_norm = pred_a.item()

        v, a = self.normalizer.inverse_transform_labels(np.array([v_norm]), np.array([a_norm]))
        valence, arousal = float(v[0]), float(a[0])

        return {
            'valence': valence,
            'arousal': arousal,
            'valence_norm': v_norm,
            'arousal_norm': a_norm,
            'music_params': self._va_to_music_params(v_norm, a_norm),
        }

    @staticmethod
    def _va_to_music_params(v: float, a: float) -> dict:
        def sigmoid(x, scale=0.5):
            return 1.0 / (1.0 + np.exp(-scale * x))

        v_s = sigmoid(v)
        a_s = sigmoid(a)

        return {
            'tempo_bpm':       int(60 + a_s * 100),
            'mode':            v_s,
            'velocity':        int(30 + a_s * 97),
            'pitch_register':  v_s,
            'note_density':    a_s,
            'brightness':      v_s,
            'tension':         1.0 - v_s,
            'valence_sigmoid': v_s,
            'arousal_sigmoid': a_s,
        }
