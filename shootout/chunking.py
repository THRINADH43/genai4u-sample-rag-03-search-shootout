import re


def fixed_chunks(text, size=25, overlap=5):
    """Fixed windows of `size` words with `overlap` words shared."""
    words = text.split()
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def sentence_chunks(text):
    """One chunk per sentence."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


STRATEGIES = {"fixed25": fixed_chunks, "sentence": sentence_chunks}
