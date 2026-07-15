#!/usr/bin/env python3
"""
Minimal Flask API for the lyrics translator.

Exposes just the translation service — no database, auth, or unrelated
routes. The M2M100 model is loaded once at startup so requests are fast.

Endpoints
    GET  /api/health
    GET  /api/search_lyrics?artist=<a>&title=<t>
    POST /api/translate      body: {"lyrics": "...", "target_code": "es"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from translator import fetch_lyrics, translate_lyrics, get_translator

app = Flask(__name__)
CORS(app)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/search_lyrics")
def search_lyrics():
    artist = request.args.get("artist", type=str)
    title = request.args.get("title", type=str)
    if not artist or not title:
        return jsonify({"error": "artist and title are required"}), 400
    lyrics = fetch_lyrics(artist, title)
    if not lyrics:
        return jsonify({"error": "lyrics not found"}), 404
    return jsonify({"artist": artist, "title": title, "lyrics": lyrics})


@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    lyrics = data.get("lyrics")
    target_code = data.get("target_code")
    if not lyrics or not target_code:
        return jsonify({"error": "lyrics and target_code are required"}), 400
    translated = translate_lyrics(lyrics, target_code)
    return jsonify({"target_code": target_code, "lyrics": translated})


if __name__ == "__main__":
    # Warm the model on startup so the first request isn't slow.
    get_translator()
    app.run(host="0.0.0.0", port=5000, debug=False)
