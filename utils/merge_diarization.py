# utils/merge_diarization.py
from typing import List, Dict

def _overlap(a_start, a_end, b_start, b_end):
    """Return overlap duration between intervals [a_start,a_end] & [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_speaker_to_transcript(whisper_segments: List[Dict], diarization_segments: List[Dict]):
    """
    For each whisper segment, find the diarization segment with max overlap and assign its speaker.
    Returns list of segments: {"start","end","text","speaker"}
    """
    assigned = []
    for w in whisper_segments:
        best_speaker = None
        best_overlap = 0.0
        for d in diarization_segments:
            ov = _overlap(w["start"], w["end"], d["start"], d["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = d["speaker"]
        # if no overlap (rare), fallback to "Unknown"
        if best_speaker is None:
            best_speaker = "Unknown"
        assigned.append({
            "start": w["start"],
            "end": w["end"],
            "text": w["text"],
            "speaker": best_speaker
        })
    return assigned

def coalesce_segments(assigned_segments: List[Dict], gap_tolerance: float = 0.5):
    """
    Merge consecutive segments with the same speaker and small gaps.
    gap_tolerance: max seconds between segments to be merged
    Returns coalesced list.
    """
    if not assigned_segments:
        return []

    merged = []
    cur = assigned_segments[0].copy()
    for seg in assigned_segments[1:]:
        if seg["speaker"] == cur["speaker"] and seg["start"] - cur["end"] <= gap_tolerance:
            # merge
            cur["end"] = seg["end"]
            cur["text"] = cur["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(cur)
            cur = seg.copy()
    merged.append(cur)
    return merged

def produce_readable_transcript(coalesced_segments: List[Dict]):
    """
    Return a readable transcript where consecutive lines from the same speaker
    are merged into one block without timestamps.

    Example:
      [SPEAKER_00]: Hello everyone. Thanks for joining.
    """
    lines = []
    prev_speaker = None
    buffer = []

    for seg in coalesced_segments:
        speaker = seg["speaker"]
        text = seg["text"].strip()

        if speaker != prev_speaker:
            # Flush previous speaker block
            if prev_speaker is not None:
                lines.append(f"[{prev_speaker}]: {' '.join(buffer)}")
            # Start new block
            prev_speaker = speaker
            buffer = [text]
        else:
            # Same speaker → append text to buffer
            buffer.append(text)

    # Flush last speaker block
    if prev_speaker is not None:
        lines.append(f"[{prev_speaker}]: {' '.join(buffer)}")

    return "\n".join(lines)


