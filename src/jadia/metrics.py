import numpy as np
from .segment import Segments, Segment
from .hungarian import optimize
import warnings


def total_length(segments: Segments):
    return sum([segment.end - segment.start for segment in segments])


def intersection_length(a: Segment, b: Segment):
    """Compute the intersection length of two Segments."""
    max_start = max(a.start, b.start)
    min_end = min(a.end, b.end)
    return max(0.0, min_end - max_start)


def invert_segments(segments: Segments, maxtime: float | None = None):
    negative = Segments()
    start = 0.0
    for s in segments:
        candidate = Segment(start=start, end=s.start, speaker="")
        if candidate.len() > 0:
            negative.append(candidate)
        start = s.end
    if maxtime is not None and negative.max() < maxtime:
        negative.append(Segment(start=negative.max(), end=maxtime))
    return negative


def false_alarm(segments: Segments, ground_truth: Segments):
    ngt = invert_segments(ground_truth)
    fa = 0.0

    for s in segments:
        for gt in ngt:
            fa += intersection_length(s, gt)
    return fa


def missed_detection(segments: Segments, ground_truth: Segments):
    ###   ground truth: ........xxxxxxxxxx.......xxxxxxxx........
    ###    predictions: .....xxxxxxx.......xxxxxxxxx.............
    ###         missed:             ++++++          +++++

    ns = invert_segments(segments, ground_truth.max())
    md = 0.0

    for gt in ground_truth:
        for s in ns:
            md += intersection_length(s, gt)
    return md


def overlap(a: Segments, b: Segments) -> float:
    ov = 0.0
    for sa in a:
        for sb in b:
            ov += intersection_length(sa, sb)
    return ov


def confusion(
    segments: Segments, ground_truth: Segments, rows: np.ndarray, cols: np.ndarray
):
    cnf = 0.0
    rows = list(rows)
    cols = list(cols)
    gtspeakers = ground_truth.speakers()
    for i, speaker in enumerate(segments.speakers()):
        sps = [el for el in segments if el.speaker == speaker]
        gtspeaker_id = cols[rows.index(i)]
        gtps = [el for el in ground_truth if el.speaker != gtspeakers[gtspeaker_id]]
        for s in sps:
            for gt in gtps:
                if (ilen := intersection_length(s, gt)) > 0:
                    cnf += ilen

    return cnf


def speaker_index(segments: Segments):
    """Build the index for the speakers.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Returns:
        a dict from speaker to integer
    """
    speaker_set = sorted({seg.speaker for seg in segments})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def compute_merged_total_length(ref: Segments, hyp: Segments):
    """Compute the total length of the union of reference and hypothesis.

    Arguments:
        - ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        - hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a float number for the union total length
    """
    # Remove speaker label and merge.
    merged = [(seg.start, seg.end) for seg in (ref + hyp)]
    # Sort by start.
    merged = sorted(merged, key=lambda el: el[0])
    i = len(merged) - 2
    while i >= 0:
        if merged[i][1] >= merged[i + 1][0]:
            max_end = max(merged[i][1], merged[i + 1][1])
            merged[i] = (merged[i][0], max_end)
            del merged[i + 1]
            if i == len(merged) - 1:
                i -= 1
        else:
            i -= 1
    total_length = sum([el[1] - el[0] for el in merged])
    return total_length


def cost_matrix(reference: Segments, hypothesis: Segments):
    """Build the cost matrix.

    Arguments:
        - ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        - hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a 2-dim numpy array, whose element (i, j) is the overlap between
            `i`th reference speaker and `j`th hypothesis speaker
    """
    assert len(reference.speakers()) == len(
        hypothesis.speakers()
    ), "Number of speakers between reference and hypothesis do not match"
    reference_index = speaker_index(reference)
    hypothesis_index = speaker_index(hypothesis)

    cmatrix = np.zeros((len(reference_index), len(hypothesis_index)))
    for reference_el in reference:
        for hypothesis_el in hypothesis:
            i = reference_index[reference_el.speaker]
            j = hypothesis_index[hypothesis_el.speaker]
            cmatrix[i, j] += intersection_length(reference_el, hypothesis_el)
    return cmatrix


def DER(reference: Segments, hypothesis: Segments) -> float:
    def check_overlap(reference):
        rs = reference.speakers()
        others = rs.copy()
        for speaker in rs:
            others.remove(speaker)
            for other in others:
                for sa in reference.filter(speaker):
                    for sb in reference.filter(other):
                        if intersection_length(sa, sb) > 0:
                            return True
        return False

    if check_overlap(reference=reference):
        warnings.warn(
            "Current DER implementation does not support multispeaker with overlap. Try using other packages."
        )
    ref_duration = total_length(reference)
    cmatrix = cost_matrix(reference, hypothesis)
    rows, cols = optimize(-cmatrix)

    fa = false_alarm(segments=hypothesis, ground_truth=reference)
    correct = cmatrix[rows, cols].sum()
    total = ref_duration
    md = missed_detection(segments=hypothesis, ground_truth=reference)
    cnf = confusion(segments=hypothesis, ground_truth=reference, rows=rows, cols=cols)

    # union_total_length = compute_merged_total_length(reference, hypothesis)
    # der = (union_total_length - correct) / ref_duration
    der = (fa + md + cnf) / ref_duration
    return {
        "false alarm": fa,
        "correct": correct,
        "total": total,
        "missed detection": md,
        "confusion": cnf,
        "diarization error rate": der,
    }
