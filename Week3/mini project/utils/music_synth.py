"""
ABC Notation to Synthesized Audio Converter.
Parses generated ABC notation tunes into notes, durations, and synthesizes 
melodic audio (WAV format) using pure Python, NumPy, and SciPy.
"""

import io
import re
import numpy as np
from scipy.io import wavfile

# Base MIDI note numbers (octave 4)
NOTE_TO_SEMITONE = {
    'C': 60,
    'D': 62,
    'E': 64,
    'F': 65,
    'G': 67,
    'A': 69,
    'B': 71
}

def midi_to_freq(midi_num):
    """Converts a MIDI note number to frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))

def parse_abc_to_notes(abc_text, default_bpm=120):
    """
    Parses ABC notation text into a list of (frequency, duration_in_seconds).
    """
    lines = abc_text.strip().split("\n")
    music_lines = []
    default_length = 1.0 / 8.0 # Default L: 1/8
    key = "C"

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("L:"):
            # e.g., L: 1/8 or L: 1/16
            m = re.search(r"(\d+)/(\d+)", line)
            if m:
                default_length = float(m.group(1)) / float(m.group(2))
        elif line.startswith("K:"):
            # e.g. K: Dmaj, K: G
            key = line[2:].strip()
        elif line.startswith(("X:", "T:", "M:", "R:", "C:", "B:", "D:", "F:", "G:", "H:", "N:", "O:", "P:", "S:", "W:", "Z:", "%")):
            # Other headers or comments
            continue
        else:
            music_lines.append(line)

    body = " ".join(music_lines)
    seconds_per_whole_note = (60.0 / default_bpm) * 4.0
    base_note_duration = default_length * seconds_per_whole_note

    # Regex to capture note patterns:
    # optional accidental: ^ or _ or =
    # note letter: [A-Ga-gz]
    # octave modifier: , or '
    # duration modifier: e.g. 2, 3, 4, /2, /
    pattern = re.compile(r"([\^_=]?)([A-Ga-gz])([,']*)(\d*(?:/\d*)?)")
    matches = pattern.finditer(body)

    notes_sequence = []
    for match in matches:
        accidental, note_letter, octave_mod, dur_str = match.groups()
        if not note_letter:
            continue

        # Parse duration
        duration_mult = 1.0
        if dur_str:
            if "/" in dur_str:
                parts = dur_str.split("/")
                num = float(parts[0]) if parts[0] else 1.0
                den = float(parts[1]) if parts[1] else 2.0
                duration_mult = num / den
            else:
                try:
                    duration_mult = float(dur_str)
                except ValueError:
                    duration_mult = 1.0

        dur_sec = max(0.05, base_note_duration * duration_mult)

        # Handle rests (z)
        if note_letter.lower() == 'z':
            notes_sequence.append((0.0, dur_sec))
            continue

        # Determine octave
        is_upper = note_letter.isupper()
        base_char = note_letter.upper()
        if base_char not in NOTE_TO_SEMITONE:
            continue

        base_midi = NOTE_TO_SEMITONE[base_char]
        # lowercase in ABC is octave 5
        if not is_upper:
            base_midi += 12

        # Octave modifiers
        if octave_mod:
            for ch in octave_mod:
                if ch == "'":
                    base_midi += 12
                elif ch == ",":
                    base_midi -= 12

        # Accidental
        if accidental == "^":
            base_midi += 1
        elif accidental == "_":
            base_midi -= 1

        freq = midi_to_freq(base_midi)
        notes_sequence.append((freq, dur_sec))

    return notes_sequence


def synthesize_audio(notes_sequence, sample_rate=22050):
    """
    Synthesizes a list of (freq, duration) into a smooth WAV byte stream.
    Adds gentle harmonic overtones and ADSR envelope for musical warmth.
    """
    if not notes_sequence:
        # Generate 1 second of silence
        data = np.zeros(sample_rate, dtype=np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, data)
        return buffer.getvalue()

    total_samples = []

    for freq, duration in notes_sequence:
        num_samples = int(sample_rate * duration)
        if num_samples <= 0:
            continue

        if freq <= 0:
            # Rest: silence
            wave = np.zeros(num_samples)
        else:
            t = np.linspace(0, duration, num_samples, endpoint=False)
            # Rich harmonic timbre (flute / bell / piano)
            wave = (
                0.60 * np.sin(2 * np.pi * freq * t) +
                0.25 * np.sin(2 * np.pi * 2 * freq * t) +
                0.10 * np.sin(2 * np.pi * 3 * freq * t) +
                0.05 * np.sin(2 * np.pi * 4 * freq * t)
            )

            # ADSR Envelope
            attack = int(sample_rate * 0.02)
            decay = int(sample_rate * 0.04)
            release = int(sample_rate * 0.04)

            env = np.ones(num_samples)
            if num_samples > attack:
                env[:attack] = np.linspace(0, 1, attack)
            if num_samples > attack + decay:
                env[attack : attack + decay] = np.linspace(1, 0.75, decay)
            if num_samples > release:
                env[-release:] = np.linspace(0.75, 0, release)

            wave = wave * env

        total_samples.append(wave)

    if not total_samples:
        combined = np.zeros(sample_rate, dtype=np.float32)
    else:
        combined = np.concatenate(total_samples)

    # Normalize audio
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = combined / max_val * 0.85

    # Convert to 16-bit PCM
    audio_pcm = (combined * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_pcm)
    return buffer.getvalue()


def abc_to_wav_bytes(abc_text, bpm=130):
    """High-level function: takes ABC text string and returns WAV audio bytes."""
    notes = parse_abc_to_notes(abc_text, default_bpm=bpm)
    return synthesize_audio(notes)
