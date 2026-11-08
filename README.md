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
3. **Batch** — non-blank lines are grouped and translated in a single forward pass per batch, then placed back in their original positions so stanza structure survives. About 4.5x faster than one line at a time; set `BATCH_SIZE` to change it.
4. **Serve** — a minimal Flask API exposes fetch + translate; the model is **loaded once at startup and cached**, so requests stay fast rather than reloading the model every call.

## Quick start

```bash
pip install -r requirements.txt

# run the tests (fast, no model download)
python test_translator.py

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

## Measured performance

Numbers below were produced by `measure.py` on this repo, on CPU, with
M2M100-418M. Reproduce them with `python measure.py latency --lang fr --n 30`.

| Configuration | Per line |
|---|---|
| Greedy decoding, one line at a time | 0.350 s |
| 4-beam search, one line at a time | 0.568 s |
| Greedy, batched at 8 | 0.076 s |
| 4-beam, batched at 8 | 0.131 s |

Two things that fall out of this. Beam search costs roughly 62% more time
than greedy, which is the price of better translations and why the default
is 4 beams. And batching is worth about 4.5x, which is why lines are now
grouped rather than sent one at a time.

**BLEU is not currently measured.** `measure.py bleu` will score the pipeline
against a parallel corpus such as FLORES-200 or a WMT test set, but no score
is claimed here until that has been run.

100+ languages are supported, which is a property of M2M100 itself.

## Tech

Python · Hugging Face Transformers · Meta M2M100 · PyTorch · Flask · REST API

## Files

| File | Role |
|------|------|
| `translator.py` | Core engine — fetch lyrics + batched M2M100 translation (also a CLI) |
| `app.py` | Minimal Flask API (`/api/search_lyrics`, `/api/translate`) |
| `measure.py` | Latency and BLEU benchmarks |
| `test_translator.py` | Tests for batching and line reassembly; no model download needed |
| `.env.example` | Optional `GENIUS_TOKEN` for the Genius fallback |

## Notes

The M2M100 weights are downloaded from Hugging Face on first run and are intentionally git-ignored. This engine was extracted from a larger Monash team project; the translation pipeline here is my own work.

---

Built by [Hitanshu Joshi](https://www.linkedin.com/in/hitanshujoshi/) — AI Engineer.
