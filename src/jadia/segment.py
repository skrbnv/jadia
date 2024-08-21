import numpy as np
from datetime import timedelta
from os.path import basename, isfile


class Segment:
    def __init__(
        self,
        start: float = 0.0,
        end: float = 0.0,
        speaker: str = "",
        confidence: float | None = None,
        filename: str | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker
        self.filename = filename
        self.confidence = 1.0 if confidence is None else confidence

    def len(self):
        return self.end - self.start

    def __repr__(self) -> str:
        return f"<Segment({self.start}, {self.end}), speaker {self.speaker}>"

    def __str__(self) -> str:
        return f"{self.pprint(self.start)} -> {self.pprint(self.end)} : {self.speaker if self.speaker is not None else '[silence]'}"

    def pprint(self, t: timedelta, precision: int = 2):
        """
        Return pretty string representation of timedelta
        Arguments:
            - t: timedelta
            - precision: int - number of digits after dot
        """
        return f"{str(timedelta(seconds=t//1))}" + (
            f".{t%1:.2f}"[2 : 2 + precision] if precision > 0 else ""
        )


class Segments:
    """
    Class containing list of Segment instances
    """

    def __init__(self, segments: list[Segment] | None = None) -> None:
        self._data: list[Segment] = [] if segments is None else segments

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def max(self) -> float:
        if len(self._data) > 0:
            return self._data[-1].end
        else:
            return 0

    def filter(self, speaker):
        out = Segments()
        for el in self._data:
            if el.speaker == speaker:
                out.append(el)
        return out

    def min(self) -> float:
        if len(self._data) > 0:
            return self._data[0].start
        else:
            return 0

    def __getitem__(self, index) -> Segment:
        return self._data[index]

    def __delitem__(self, index) -> None:
        del self._data[index]

    def __add__(self, other):
        if isinstance(other, Segments):
            return Segments(
                segments=self._data + other._data,
            )
        return NotImplemented

    def append(self, el: Segment) -> None:
        self._data.append(el)

    def adjust(self, edges: np.ndarray, tolerance: float):
        """
        Moves both ends of segments towards edges.
        Arguments:
            - edges: np.array of reference edges
            - tolerance: max amount of moving it allowed to
        """
        for i in range(len(self._data)):
            if self._data[i].speaker is None:
                continue
            if (i + 1) < len(self._data) and self._data[i + 1].speaker != self._data[
                i
            ].speaker:
                pred_edge = self._data[i].end
                closest = np.argmin(np.abs(edges - pred_edge))
                dist = self._data[i].end - edges[closest]
                absdist = np.abs(dist)
                if absdist > 0 and absdist < tolerance:
                    self._data[i].end = edges[closest] + (0 if dist <= 0 else -1)
                    self._data[i + 1].start = edges[closest] + (1 if dist <= 0 else 0)

    def merge(self, collapse: bool = False, tolerance: float = 0.2):
        """
        Merges segments with same speaker. If collapse is True then ignores silent intervals between.
        Arguments:
            - collapse: bool [False]
        """
        i = len(self._data)
        while i > 0:
            i -= 1
            for j in reversed(range(i)):
                if self._data[j].speaker == None:
                    if collapse is True:
                        continue
                    else:
                        break
                elif self._data[j].speaker != self._data[i].speaker:
                    break
                else:
                    if self._data[i].start - self._data[j].end <= tolerance:
                        self._data[j].end = self._data[i].end
                        for k in reversed(range(j + 1, i + 1)):
                            del self._data[k]
                    i = j + 1
                    break

    def drop_short(self, min_len: float = 0.1):
        """
        Removes short segments
        Arguments:
            - min_len: min length to remove, in seconds
        """
        for i in reversed(range(len(self._data))):
            if (self._data[i].end - self._data[i].start) <= min_len:
                del self._data[i]

    def drop_silent(self):
        """
        Removes silent|no-speech segments
        """
        for i in reversed(range(len(self._data))):
            if self._data[i].speaker == None:
                del self._data[i]

    def speakers(self) -> list:
        """
        Returns _sorted_ list of _unique_ speakers from segments
        """
        return sorted(list(set([el.speaker for el in self._data])))

    def __repr__(self):
        return f"<Segments [{len(self._data)}]>"

    def __str__(self):
        return "\n".join([str(segment) for segment in self._data])

    def __list__(self):
        return list(self)

    def to_rttm(self, filename: str):
        fname = basename(filename).split(".")[0]
        lines = []
        for seg in self._data:
            lines.append(
                f"SPEAKER {fname} 1 {seg.start:.6f} {(seg.end-seg.start):.6f} <NA> <NA> {seg.speaker} {seg.confidence if seg.confidence is not None else '<NA>'} <NA>\n"
            )
        return lines

    def to_rttm_save(self, fpath: str, overwrite: bool = False):
        if isfile(fpath) and overwrite is False:
            input("File already exists, press Enter to overwrite..")
        lines = self.to_rttm(fpath)
        with open(fpath, "wt") as f:
            f.writelines(lines)

    def from_rttm(self, rttm: str | list):
        """
        Reads data from rttm (could be string or list)
        Arguments:
            - rttm:
        """
        if isinstance(rttm, str):
            rttm = rttm.splitlines()
        for line in rttm:
            split = line.split(" ")
            assert len(split) == 10, "Incorrect number of elements in rttm string"
            filename = split[1]
            onset = float(split[3])
            duration = float(split[4])
            speaker = split[7]
            try:
                confidence = float(split[8])
            except:
                confidence = None
            self._data.append(
                Segment(
                    start=onset,
                    end=onset + duration,
                    speaker=speaker,
                    filename=filename,
                    confidence=confidence,
                )
            )
        return self

    def from_rttm_load(self, fpath: str):
        with open(fpath, "rt") as f:
            self.from_rttm(f.readlines())
        return self

    def to_lab(self):
        lines = []
        for seg in self._data:
            lines.append(f"{seg.start:.6f} {seg.end:.6f} {seg.speaker}\n")
        return lines

    def to_lab_save(self, fpath):
        lines = self.to_lab()
        with open(fpath, "wt") as f:
            f.writelines(lines)

    def from_lab(self, lab: str | list):
        if isinstance(lab, str):
            lab = lab.splitlines()
        for line in lab:
            split = line.split(" ")
            assert len(split) == 3, "Incorrect number of elements in lab string"
            start = float(split[0])
            end = float(split[1])
            speaker = split[2]
            self._data.append(Segment(start=start, end=end, speaker=speaker))

    def from_lab_load(self, fpath):
        with open(fpath, "wt") as f:
            self.from_lab(f.readlines())
