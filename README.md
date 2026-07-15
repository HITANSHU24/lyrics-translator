# Multilingual Lyrics Translator

Fetch a song's lyrics and translate them **line by line into 100+ languages** with Meta's [M2M100](https://huggingface.co/facebook/m2m100_418M) model — preserving the verse-and-chorus structure instead of flattening it into a wall of text.

This is the standalone translation engine I built as part of a Monash University team project. It's extracted here on its own — just the NLP/translation pipeline and a minimal API, with none of the wider app's database, auth, or unrelated features.

## Why line-by-line

Song lyrics aren't prose. Translating a whole block at once loses the line breaks that make lyrics readable and singable. This pipeline translates each line independently and preserves blank lines between stanzas, so the translated version keeps the same shape as the original.

## How it works

```
  artist + title            lyrics text                translated lyrics
 ┌──────────────┐  fetch  ┌──────────────┐  M2M100   ┌──────────────────┐
 │ lyrics.ovh   │ ──────▶ │ line-split + │ ────────▶ │ target language, │
 │ (Genius fb.) │         │ per-line MT  │           │ structure kept   │
 └──────────────┘         └──────────────┘           └──────────────────┘
```

1. **Fetch** — pulls lyrics from the free lyrics.ovh API, with an optional Genius fallback if you supply a token.
2. **Translate** — forces the M2M100 decoder to the target language (`forced_bos_token_id`) and generates with beam search, line by line.
3. **Serve** — a minimal Flask API exposes fetch + translate; the model is **loaded once at startup and cached**, so requests stay fast rather than reloading the model every call.

## Quick start

```bash
pip install -r requirements.txt

# CLI
python translator.py

# or run the API
python app.py     # serves on http://localhost:5000
```

### API

```bash
# fetch original lyrics
curl "http://localhost:5000/api/search_lyrics?artist=Coldplay&title=Yellow"

# translate a lyrics block (target is an ISO 639-1 code: es, fr, hi, ar, ja …)
curl -X POST http://localhost:5000/api/translate \
     -H "Content-Type: application/json" \
     -d '{"lyrics": "Look at the stars\nLook how they shine for you", "target_code": "es"}'
```

## Project results

As part of the original project this pipeline was evaluated across a large lyrics set:

- **100+ languages** supported via M2M100
- **~87% peak BLEU** on the evaluated set
- Structure-preserving, line-by-line output with cached inference for real-time use

## Tech

Python · Hugging Face Transformers · Meta M2M100 · PyTorch · Flask · REST API

## Files

| File | Role |
|------|------|
| `translator.py` | Core engine — fetch lyrics + M2M100 line-by-line translation (also a CLI) |
| `app.py` | Minimal Flask API (`/api/search_lyrics`, `/api/translate`) |
| `.env.example` | Optional `GENIUS_TOKEN` for the Genius fallback |

## Notes

The M2M100 weights are downloaded from Hugging Face on first run and are intentionally git-ignored. This engine was extracted from a larger Monash team project; the translation pipeline here is my own work.

---

Built by [Hitanshu Joshi](https://www.linkedin.com/in/hitanshujoshi/) — AI Engineer.
