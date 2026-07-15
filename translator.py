#!/usr/bin/env python3
"""
Lyrics translation engine.

Fetches song lyrics (lyrics.ovh, with an optional Genius fallback) and
translates them line by line with Meta's M2M100 multilingual model,
preserving stanza breaks. The model is loaded once and cached, so
repeated translations are fast.
"""

import os
import sys
import requests
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/m2m100_418M")

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
def translate_line(text: str, target_code: str) -> str:
    """Translate a single line into the target language (ISO 639-1 code)."""
    tokenizer, model = get_translator()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs["forced_bos_token_id"] = tokenizer.get_lang_id(target_code)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 50,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def translate_lyrics(lyrics: str, target_code: str) -> str:
    """Translate a full lyrics block line by line, preserving blank lines."""
    out = []
    for line in lyrics.split("\n"):
        out.append("" if not line.strip() else translate_line(line, target_code))
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
