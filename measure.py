#!/usr/bin/env python3
"""
measure.py - measure what your translator actually does.

Two things, both of which end up as numbers you can defend in an interview:

  1. LATENCY  - how long one line takes to translate, line-by-line vs batched.
  2. BLEU     - translation quality against human reference translations.

Run:
    python measure.py latency --lang fr --n 30
    python measure.py bleu    --lang fr --n 100

Install first:
    pip install torch transformers sentencepiece sacrebleu

The model (facebook/m2m100_418M) is ~2GB and downloads once to ~/.cache.
CPU is fine; it is just slower. The script prints which device it used,
because a latency number without the hardware attached is meaningless.
"""

import argparse
import statistics
import time
import sys

MODEL_NAME = "facebook/m2m100_418M"


def _deps():
    """Import the heavy deps lazily, so a missing one gives advice, not a traceback."""
    try:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        return torch, M2M100ForConditionalGeneration, M2M100Tokenizer
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e.name}\n\n"
            "Install everything with:\n"
            "    pip install torch transformers sentencepiece sacrebleu\n"
        )


def load_model():
    """Load tokenizer and model once, the same way a served app would."""
    torch, M2M100ForConditionalGeneration, M2M100Tokenizer = _deps()
    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    tok = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"loaded. device={device}, threads={torch.get_num_threads()}", file=sys.stderr)
    return tok, model, device


def translate(lines, tok, model, device, src="en", tgt="fr", batch_size=1, beams=4):
    """Translate a list of lines. batch_size=1 reproduces line-by-line behaviour."""
    import torch
    tok.src_lang = src
    out = []
    for i in range(0, len(lines), batch_size):
        chunk = lines[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=200)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tok.get_lang_id(tgt),   # force the decoder to the target language
                num_beams=beams,
                max_length=200,
            )
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


# --------------------------------------------------------------------- latency
SAMPLE_LINES = [
    "I never meant to let you go",
    "We were dancing in the dark",
    "Every road leads back to you",
    "The city lights are fading out",
    "Hold on, it's almost over now",
    "I remember summer rain",
    "You said forever, then you left",
    "Nothing hurts the way you do",
    "Call my name across the water",
    "We burn like paper in the wind",
]


def cmd_latency(args):
    tok, model, device = load_model()
    lines = (SAMPLE_LINES * ((args.n // len(SAMPLE_LINES)) + 1))[:args.n]

    # Warm-up. The first call allocates buffers and is never representative.
    translate(lines[:2], tok, model, device, tgt=args.lang, beams=args.beams)

    # --- line by line: what the repo currently does ---
    per_line = []
    for line in lines:
        t0 = time.perf_counter()
        translate([line], tok, model, device, tgt=args.lang, beams=args.beams)
        per_line.append(time.perf_counter() - t0)

    # --- batched: same work, grouped ---
    t0 = time.perf_counter()
    translate(lines, tok, model, device, tgt=args.lang, batch_size=args.batch, beams=args.beams)
    batched_total = time.perf_counter() - t0

    srt = sorted(per_line)
    p95 = srt[min(len(srt) - 1, int(0.95 * len(srt)))]

    print("\n================ LATENCY ================")
    print(f"device                 : {device}")
    print(f"lines                  : {len(lines)}   beams: {args.beams}   target: {args.lang}")
    print(f"line-by-line median    : {statistics.median(per_line):.3f} s/line")
    print(f"line-by-line mean      : {statistics.mean(per_line):.3f} s/line")
    print(f"line-by-line p95       : {p95:.3f} s/line")
    print(f"batched (size {args.batch:>2})       : {batched_total / len(lines):.3f} s/line"
          f"   [{batched_total:.1f}s total]")
    speedup = (statistics.mean(per_line) * len(lines)) / batched_total
    print(f"batching speed-up      : {speedup:.2f}x")
    print("=========================================")
    print("\nWrite the resume line from the median, and name the device.")


# ------------------------------------------------------------------------ bleu
def cmd_bleu(args):
    import sacrebleu

    with open(args.src) as f:
        src_lines = [l.strip() for l in f if l.strip()][:args.n]
    with open(args.ref) as f:
        ref_lines = [l.strip() for l in f if l.strip()][:args.n]

    if len(src_lines) != len(ref_lines):
        sys.exit(f"source has {len(src_lines)} lines, reference has {len(ref_lines)} - they must match")

    tok, model, device = load_model()
    print(f"translating {len(src_lines)} lines ...", file=sys.stderr)
    t0 = time.perf_counter()
    hyp = translate(src_lines, tok, model, device, src=args.src_lang, tgt=args.lang,
                    batch_size=args.batch, beams=args.beams)
    elapsed = time.perf_counter() - t0

    # Use the metric objects, not the module-level helpers, so we can print the
    # signature. The signature records tokenisation and version: without it a
    # BLEU number is not reproducible and a reviewer cannot check it.
    bleu_metric = sacrebleu.BLEU()
    chrf_metric = sacrebleu.CHRF()
    bleu = bleu_metric.corpus_score(hyp, [ref_lines])
    chrf = chrf_metric.corpus_score(hyp, [ref_lines])

    print("\n================= BLEU ==================")
    print(f"pair                   : {args.src_lang} -> {args.lang}")
    print(f"sentences              : {len(src_lines)}")
    print(f"BLEU                   : {bleu.score:.2f}")
    print(f"chrF2                  : {chrf.score:.2f}")
    print(f"signature              : {bleu_metric.get_signature()}")
    print(f"wall time              : {elapsed:.1f}s on {device}")
    print("=========================================")
    print("\nBLEU is 0-100, not a percentage of correctness.")
    print("Published M2M100-418M scores sit roughly in the teens to low thirties")
    print("depending on the language pair. Quote the number with the pair and")
    print("the test set, or it means nothing.")
    print("\nFirst three outputs, so you can eyeball that it is working:")
    for s, h, r in list(zip(src_lines, hyp, ref_lines))[:3]:
        print(f"  src: {s}\n  out: {h}\n  ref: {r}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("latency", help="measure seconds per line, batched vs not")
    pl.add_argument("--lang", default="fr", help="target language code (fr, es, de, hi, ...)")
    pl.add_argument("--n", type=int, default=30, help="how many lines to time")
    pl.add_argument("--batch", type=int, default=8, help="batch size for the batched run")
    pl.add_argument("--beams", type=int, default=4, help="match translator.py, which uses 4")
    pl.set_defaults(func=cmd_latency)

    pb = sub.add_parser("bleu", help="score translations against human references")
    pb.add_argument("--src", default="src.txt", help="file of source sentences, one per line")
    pb.add_argument("--ref", default="ref.txt", help="file of reference translations, one per line")
    pb.add_argument("--src-lang", default="en")
    pb.add_argument("--lang", default="fr", help="target language code")
    pb.add_argument("--n", type=int, default=100)
    pb.add_argument("--batch", type=int, default=8)
    pb.add_argument("--beams", type=int, default=4, help="match translator.py, which uses 4")
    pb.set_defaults(func=cmd_bleu)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
