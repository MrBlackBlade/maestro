from symusic import Score
from symusic import Synthesizer
from torch._refs import to
from src.core.config import Config
from src.core.utils import get_tokenizer
from miditok import TokSequence
# import pygame.midi
import io
import threading
import queue
import time
import numpy as np
import soundfile as sf
from midi2audio import FluidSynth
import sounddevice as sd
import os
import tempfile
import subprocess
from typing import List

tokenizer = get_tokenizer()

class AudioEngine:
    def __init__(
        self, 
        soundfont: str = Config.RESOURCES_DIR / "FluidR3_GM.sf2",
        sample_rate: int = 48000,
        bar_duration: int = 2,
    ):
        self.sample_rate = sample_rate
        self.bar_duration = bar_duration
        self.bar_samples = int(self.sample_rate * self.bar_duration)

        self.stream = sd.OutputStream(samplerate=48000, channels=2, dtype='float32')
        self.stream.start()
        self.soundfont = soundfont

        self.live_token_buffer: List[int] = []
        self.bars_buffer_queue = queue.Queue()

        self.render_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.prev_audio = None

        self.playback_done = threading.Event()

        self.first_bar = False

        threading.Thread(target=self.render_worker, daemon=True).start()
        threading.Thread(target=self.audio_worker, daemon=True).start()

    def push_token(self, token_id: int, stop=False):
        self.live_token_buffer.append(token_id)
        
        if token_id == 4 and len(self.live_token_buffer) > 0:
            self.bars_buffer_queue.put(self.live_token_buffer)
            self.live_token_buffer = []
        
        if stop:
            self.bars_buffer_queue.put(self.live_token_buffer)
            self.live_token_buffer = []
        
        if (not self.first_bar and self.bars_buffer_queue.qsize() > 1):
            current_bar = self.bars_buffer_queue.get()
            tok_sequence = TokSequence(ids=current_bar)
            tokenizer.complete_sequence(tok_sequence)
            score = tokenizer.decode(tok_sequence)
            self.render_queue.put(score)
            self.first_bar = True
        
        if (self.first_bar and self.bars_buffer_queue.qsize() > 0):
            current_bar = self.bars_buffer_queue.get()
            tok_sequence = TokSequence(ids=current_bar)
            tokenizer.complete_sequence(tok_sequence)
            score = tokenizer.decode(tok_sequence)
            self.render_queue.put(score)
        
        if stop:
            while self.bars_buffer_queue.qsize() > 0:
                current_bar = self.bars_buffer_queue.get()
                tok_sequence = TokSequence(ids=current_bar)
                tokenizer.complete_sequence(tok_sequence)
                score = tokenizer.decode(tok_sequence)
                self.render_queue.put(score)
            self.render_queue.put(None)
    
    def render_to_array(self, score):
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mid_f:
            mid_path = mid_f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
            wav_path = wav_f.name
        try:
            score.dump_midi(mid_path)
            subprocess.run(
                ["fluidsynth", "-ni", "-F", wav_path, "-r", "48000", self.soundfont, mid_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return sf.read(wav_path)
        finally:
            os.unlink(mid_path)
            os.unlink(wav_path)
    
    def write_async(self, stream, audio):
        stream.write(audio.astype(np.float32))

    def mix_tail(self, current_audio, next_audio, bar_samples):
        tail = current_audio[bar_samples:]  # reverb tail beyond bar boundary
        if len(tail) == 0:
            return next_audio
        # pad next_audio if tail is longer
        if len(tail) > len(next_audio):
            next_audio = np.pad(next_audio, ((0, len(tail) - len(next_audio)), (0, 0)))
        next_audio = next_audio.copy()
        next_audio[:len(tail)] += tail
        return next_audio
    
    def render_worker(self):
        while True:
            score = self.render_queue.get()
            if score is None:
                self.audio_queue.put(None)
                break
            audio, sr = self.render_to_array(score)
            self.audio_queue.put((audio, sr))

    def audio_worker(self):
        try:
            while True:
                item = self.audio_queue.get()
                
                if item is None:
                    break
                audio, sr = item
                audio = audio.astype(np.float32)
                if self.prev_audio is not None:
                    audio = self.mix_tail(self.prev_audio, audio, self.bar_samples)
                self.stream.write(audio[:self.bar_samples])
                # sd.sleep(int(1.8 * 1000))
                self.prev_audio = audio
        except Exception as e:
            print(f"Audio worker error: {e}")
        finally:
            self.stream.stop()
            self.stream.close()
            self.playback_done.set()