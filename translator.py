#!/usr/bin/env python3
"""
Lyrics translation engine.

Fetches song lyrics (lyrics.ovh, with an optional Genius fallback) and
translates them with Meta's M2M100 multilingual model, preserving stanza
breaks. The model is loaded once and cached, and lines are translated in
batches rather than one at a time (about 4x faster on CPU).
"""

import os
import sys
import requests
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/m2m100_418M")
# Generation settings. 4 beams costs ~62% more time than greedy but gives
# better translations; batching more than pays that back. Both measurable
# with measure.py.
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# ---------------------------------------------------------------------------
# Model: lazy-loaded singleton (loaded once, reused for every request)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


def get_translator():
    """Load the M2M100 tokenizer + model once and cache them in memory."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL_NAME} …", file=sys.stderr)
        _tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        _model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Optional Genius fallback (only if python-dotenv + lyricsgenius + token exist)
# ---------------------------------------------------------------------------
def _get_genius():
    try:
        from dotenv import load_dotenv
        import lyricsgenius
        load_dotenv()
        token = os.getenv("GENIUS_TOKEN")
        if token:
            return lyricsgenius.Genius(token, timeout=15, retries=3)
    except ImportError:
        pass
    return None


_genius = _get_genius()


# ---------------------------------------------------------------------------
# Lyrics fetching
# ---------------------------------------------------------------------------
def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch lyrics via lyrics.ovh, falling back to Genius if configured."""
    try:
        resp = requests.get(
            f"https://api.lyrics.ovh/v1/{artist}/{title}", timeout=15
        )
        if resp.status_code == 200:
            text = resp.json().get("lyrics", "").strip()
            if text:
                return text
    except requests.RequestException:
        pass

    if _genius:
        try:
            song = _genius.search_song(title, artist)
            if song and song.lyrics:
                return song.lyrics.strip()
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_batch(texts: list[str], target_code: str,
                    batch_size: int = BATCH_SIZE) -> list[str]:
    """Translate many lines in grouped forward passes.

    Translating one line at a time wastes most of the work: the model can
    process a whole batch in a single pass for barely more time than one
    line. We pad each group to its longest member so the tensors line up,
    and the attention mask the tokenizer returns tells the model to ignore
    the padding, so batched output matches line-by-line output.

    Measured on CPU with M2M100-418M: 0.57s/line one at a time, 0.13s/line
    at batch_size=8. See measure.py.
    """
    if not texts:
        return []
    tokenizer, model = get_translator()
    results: list[str] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt",
                           padding=True, truncation=True)
        longest = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(target_code),
                max_length=longest + 50,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return results


def translate_line(text: str, target_code: str) -> str:
    """Translate a single line. Kept for the one-off case; batches of one."""
    return translate_batch([text], target_code, batch_size=1)[0]


def translate_lyrics(lyrics: str, target_code: str,
                     batch_size: int = BATCH_SIZE) -> str:
    """Translate a full lyrics block, preserving blank lines and line order.

    Blank lines are never sent to the model. We remember the position of
    each non-blank line, translate only those, then put the results back
    where they came from so the verse and chorus structure survives.
    """
    lines = lyrics.split("\n")
    positions = [i for i, line in enumerate(lines) if line.strip()]
    translated = translate_batch([lines[i] for i in positions],
                                 target_code, batch_size)

    out = [""] * len(lines)
    for position, text in zip(positions, translated):
        out[position] = text
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    print("\n🎵  Lyrics Fetch & Line-by-Line Translate  🎵\n")
    artist = input("Artist name: ").strip()
    title = input("Song title:  ").strip()
    if not artist or not title:
        print("Error: both artist and title are required.", file=sys.stderr)
        sys.exit(1)

    lyrics = fetch_lyrics(artist, title)
    if not lyrics:
        print(f"❌ Lyrics not found for “{title}” by {artist}.", file=sys.stderr)
        sys.exit(2)

    print(f"\n--- Original Lyrics ({artist} – {title}) ---\n{lyrics}\n")

    target_code = input("Target ISO code (e.g. ar, fa, ur, hi, es): ").strip()
    print("\n--- Translated Lyrics ---\n")
    print(translate_lyrics(lyrics, target_code))


if __name__ == "__main__":
    main()
