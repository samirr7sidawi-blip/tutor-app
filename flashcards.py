"""Flashcards — generate Q/A cards from a study session and ship them to Anki.

The AI writes the card content (front/back) from the conversation; Anki is only
the review/scheduling host. Cards accumulate per subject in the session JSON.

Two ways into Anki, both dependency-free (stdlib only):
- ``send_to_anki`` — live push via the AnkiConnect add-on (HTTP on localhost:8765).
- ``to_csv`` — a CSV the user imports via Anki's File -> Import.
"""
import csv
import io
import json
import urllib.request

ANKI_CONNECT_URL = "http://127.0.0.1:8765"

FLASHCARD_PROMPT = """
You are creating study flashcards for the subject: '{subject_name}'.
Review the conversation so far (and the PDF knowledge base, if provided) and extract the most
important, testable facts the learner should remember.

Output RULES:
- Return ONLY a JSON array. No prose, no markdown, no code fences.
- Each element is an object: {{"front": "...", "back": "..."}}.
- "front" is a question or term; "back" is a concise, self-contained answer.
- Make each card atomic (one idea). Aim for the ~8 most important cards, fewer if the material
  is small.
- Do not create cards that duplicate the existing fronts listed below (if any).
"""

REGEN_PROMPT = """
You are revising ONE study flashcard for the subject: '{subject_name}', using the conversation
and the PDF knowledge base (if provided) for context.
Apply the user's change request to produce a single improved flashcard — keep it atomic and
accurate.
Return ONLY a JSON array containing exactly one object: [{{"front": "...", "back": "..."}}].
No prose, no code fences.
"""


def generate_cards(provider, subject, history, pdf_ctx, existing_fronts):
    """Ask the selected provider for new flashcards as a list of {front, back} dicts."""
    system = FLASHCARD_PROMPT.format(subject_name=subject)
    if existing_fronts:
        system += "\n\nEXISTING CARD FRONTS (do not duplicate these):\n- " + "\n- ".join(existing_fronts)
    if pdf_ctx:
        system += f"\n\n[PDF KNOWLEDGE BASE]\n{pdf_ctx}"

    user = "Create the flashcards from our session now. Return ONLY the JSON array."
    raw = "".join(provider.chat(system, history, user))
    return _parse_cards(raw)


def regenerate_card(provider, subject, history, pdf_ctx, card, feedback):
    """Revise one card per the user's change request. Returns a {front, back} dict or None."""
    system = REGEN_PROMPT.format(subject_name=subject)
    if pdf_ctx:
        system += f"\n\n[PDF KNOWLEDGE BASE]\n{pdf_ctx}"
    user = (
        "Current flashcard:\n"
        f"Front: {card['front']}\n"
        f"Back: {card['back']}\n\n"
        f"Change request: {feedback}\n\n"
        "Return the revised card as a JSON array with exactly one object."
    )
    raw = "".join(provider.chat(system, history, user))
    cards = _parse_cards(raw)
    return cards[0] if cards else None


def _parse_cards(raw):
    """Pull a JSON array of cards out of the model's reply, tolerating fences/prose."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON array of cards.")
    data = json.loads(raw[start:end + 1])

    cards, seen = [], set()
    for item in data:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        if front and back and front.lower() not in seen:
            seen.add(front.lower())
            cards.append({"front": front, "back": back})
    return cards


def to_csv(cards):
    """Front,Back CSV — one note per row, Anki-importable (no header)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for c in cards:
        writer.writerow([c["front"], c["back"]])
    return buf.getvalue()


def _invoke(action, **params):
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        ANKI_CONNECT_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("result")


def send_to_anki(subject, cards):
    """Push cards into Anki via AnkiConnect. Returns the count actually added.

    Raises (URLError/ConnectionError) if Anki isn't running with the AnkiConnect
    add-on — the caller should fall back to the CSV download.
    """
    deck = f"Tutor::{subject}"
    _invoke("createDeck", deck=deck)
    notes = [
        {
            "deckName": deck,
            "modelName": "Basic",
            "fields": {"Front": c["front"], "Back": c["back"]},
            "tags": ["tutor-app", subject.replace(" ", "_")],
            "options": {"allowDuplicate": False},
        }
        for c in cards
    ]
    result = _invoke("addNotes", notes=notes)
    # addNotes returns a note id per note, or null for skipped duplicates/failures.
    return sum(1 for r in result if r) if isinstance(result, list) else 0
