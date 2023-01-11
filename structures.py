from typing import List

class Letter:
    def __init__(self, idx, letter, x, y, w, h) -> None:
        self.idx = idx
        self.letter = letter
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

class Word:
    def __init__(self, letters: List[Letter]) -> None:
        self.letters = letters

    @property
    def bbox(self):
        x1s = [l.x for l in self.letters]
        y1s = [l.y for l in self.letters]
        
        x2s = [l.x + l.w for l in self.letters]
        y2s = [l.y + l.h for l in self.letters]  

        x1 = min(min(x1s), min(x2s))
        y1 = min(min(y1s), min(y2s))
        x2 = max(max(x1s), max(x2s))
        y2 = max(max(y1s), max(y2s))

        return x1, y1, x2, y2

class Option:   
    def __init__(self, letter, x, y, w, h) -> None:
        self.letter = letter
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__ (self) -> str:
        return str(self.letter)