#!/usr/bin/env python3
"""
Tests for the batching logic in translator.py.

Runs in milliseconds and needs no model download: the tokenizer and model
are replaced with stubs, so the REAL batching and line-reassembly code is
exercised, just without the neural network underneath.

What these actually protect against: batching is the kind of change that
silently reorders output or loses blank lines, and you would not notice
by eyeballing one song.

Run:  python test_translator.py
"""

import sys
import types

# translator.py imports torch and transformers at module level; stub them
# so this file runs on a machine with neither installed.
_torch = types.ModuleType("torch")
_torch.no_grad = lambda: __import__("contextlib").nullcontext()
sys.modules.setdefault("torch", _torch)
_tf = types.ModuleType("transformers")
_tf.M2M100ForConditionalGeneration = object
_tf.M2M100Tokenizer = object
sys.modules.setdefault("transformers", _tf)

import translator

BATCHES = []          # every chunk the model was actually asked to translate


class FakeIds:
    """Stands in for the input_ids tensor, carrying the texts along with it."""
    def __init__(self, texts):
        self.texts = texts
        self.shape = (len(texts), max((len(t) for t in texts), default=0))


class FakeTokenizer:
    def __call__(self, chunk, **kwargs):
        BATCHES.append(list(chunk))
        return {"input_ids": FakeIds(list(chunk))}

    def get_lang_id(self, code):
        return 0

    def batch_decode(self, outputs, **kwargs):
        return [f"<{t}>" for t in outputs.texts]


class FakeModel:
    def generate(self, input_ids=None, **kwargs):
        return input_ids          # echo the batch straight back


translator.get_translator = lambda: (FakeTokenizer(), FakeModel())


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return True
    print(f"  FAIL  {name}\n        got:  {got!r}\n        want: {want!r}")
    return False


def main():
    results = []

    # 1. Blank lines stay blank and everything stays in position.
    BATCHES.clear()
    lyrics = "first line\n\nsecond line\nthird line\n\n\nfourth line"
    results.append(check(
        "blank lines preserved exactly",
        translator.translate_lyrics(lyrics, "fr"),
        "<first line>\n\n<second line>\n<third line>\n\n\n<fourth line>",
    ))

    # 2. Blank lines are never sent to the model.
    results.append(check(
        "no blank lines sent to the model",
        [line for batch in BATCHES for line in batch],
        ["first line", "second line", "third line", "fourth line"],
    ))

    # 3. Order survives when the input spans several batches.
    BATCHES.clear()
    many = "\n".join(f"line {i}" for i in range(20))
    results.append(check(
        "order preserved across multiple batches",
        translator.translate_lyrics(many, "fr", batch_size=3),
        "\n".join(f"<line {i}>" for i in range(20)),
    ))

    # 4. The real chunking splits into batch_size groups.
    results.append(check(
        "input chunked to batch_size",
        [len(b) for b in BATCHES],
        [3, 3, 3, 3, 3, 3, 2],
    ))

    # 5. Whitespace-only lines count as blank.
    BATCHES.clear()
    results.append(check(
        "whitespace-only line treated as blank",
        translator.translate_lyrics("a\n   \nb", "fr"),
        "<a>\n\n<b>",
    ))

    # 6. Empty input short-circuits without touching the model.
    BATCHES.clear()
    results.append(check(
        "empty input returns empty and calls no model",
        (translator.translate_lyrics("", "fr"), BATCHES),
        ("", []),
    ))

    # 7. translate_line still works, as a batch of one.
    BATCHES.clear()
    results.append(check(
        "translate_line unchanged for callers",
        (translator.translate_line("hello", "fr"), BATCHES),
        ("<hello>", [["hello"]]),
    ))

    print()
    if all(results):
        print(f"All {len(results)} tests passed.")
        return 0
    print(f"{results.count(False)} of {len(results)} tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
