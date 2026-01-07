import unittest
import numpy as np
from transcribe.segmenter import f0_to_note_events
from beatmap import NoteEvent

class TestSegmenter(unittest.TestCase):
    def test_segmentation_simple(self):
        # Create a synthetic F0 with 2 notes
        # 1 sec silence, 1 sec 440Hz (A4), 1 sec silence, 1 sec 880Hz (A5)
        # SR=100, Hop=1 => 100 frames per sec
        
        sr = 100
        audio = np.ones(400) # Mock audio for RMS
        f0 = np.zeros(400)
        
        # Note 1: Frames 100-200 (1s to 2s) -> 440Hz
        f0[100:200] = 440.0
        
        # Note 2: Frames 300-400 (3s to 4s) -> 880Hz
        f0[300:400] = 880.0
        
        events = f0_to_note_events(f0, audio, sr, hop_length=1)
        
        self.assertEqual(len(events), 2)
        
        # Check Note 1
        n1 = events[0]
        self.assertAlmostEqual(n1.time, 1.0, delta=0.1)
        self.assertAlmostEqual(n1.duration, 1.0, delta=0.1)
        self.assertAlmostEqual(n1.pitch, 69.0, delta=1.0) # A4 is 69
        
        # Check Note 2
        n2 = events[1]
        self.assertAlmostEqual(n2.time, 3.0, delta=0.1)
        self.assertAlmostEqual(n2.pitch, 81.0, delta=1.0) # A5 is 81
        
    def test_hysteresis(self):
        # Vibrato test: Oscillate around 440 (+- 0.3 semitones)
        # Verify it stays as one note
        sr = 100
        audio = np.ones(200)
        f0 = np.zeros(200)
        
        # Base midi 69 (440Hz). +/- 0.3 semitones
        # 440Hz * 2^(0.3/12)
        base = 440.0
        c1 = base * (2**(0.2/12))
        c2 = base * (2**(-0.2/12))
        
        # Alternate every 10 frames
        for i in range(200):
            f0[i] = c1 if (i // 10) % 2 == 0 else c2
            
        events = f0_to_note_events(f0, audio, sr, hop_length=1)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].pitch, 69.0, delta=0.5)

if __name__ == '__main__':
    unittest.main()
