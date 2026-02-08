from dataclasses import dataclass
from typing import List, Optional

@dataclass
class NoteEvent:
    """
    A raw musical event detected by the 'Ear'.
    """
    time: float
    duration: float
    midi: float # Float because pitch bends exist
    velocity: float
    source: str # 'vocals', 'drums', 'piano', etc.
    confidence: float = 1.0

@dataclass
class GameNote:
    """
    A final object in the rhythm game chart.
    """
    time: float
    lane: int
    duration: float
    type: str # 'tap' or 'hold'
    midi: int # Quantized for visuals
