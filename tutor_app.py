import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import base64
import io
import json
import os
import re
import glob
import datetime

from providers.registry import PROVIDERS, DEFAULT_MODEL
from providers.base import MissingAPIKey
from modes import MODES, GUIDED_PHASES, PHASE_LABELS, build_system
import flashcards as fc
import imageutil

# Load ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY from .env (no sidebar pasting).
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Study Tutor", page_icon="🗂️")

# --- WARM BACKGROUND ---
# Drop a photo at assets/background.(png|jpg) and it appears, dimmed for readability.
@st.cache_data(show_spinner=False)
def _background_css(path, mtime):
    img = Image.open(path).convert("RGB")
    if max(img.size) > 1920:  # keep the inlined image light
        s = 1920 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"""
    <style>
    .stApp {{
        background-image:
          linear-gradient(rgba(23,18,13,0.86), rgba(23,18,13,0.90)),
          url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """

for _bg in ("assets/background.png", "assets/background.jpg", "assets/background.jpeg"):
    if os.path.exists(_bg):
        st.markdown(_background_css(_bg, os.path.getmtime(_bg)), unsafe_allow_html=True)
        break

# --- INITIALIZE SESSION STATE ---
if "current_subject" not in st.session_state:
    st.session_state.current_subject = "General"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "deep_thinking" not in st.session_state:
    st.session_state.deep_thinking = False
if "mode" not in st.session_state:
    st.session_state.mode = "Tutor"
if "phase" not in st.session_state:
    st.session_state.phase = "intro"
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# --- SETTINGS ---
SESSIONS_DIR = "sessions"

# Cap how much we send to the API each turn to avoid token-quota errors.
MAX_HISTORY_TURNS = 12          # last N messages kept verbatim
MAX_PDF_CHARS = 60_000          # ~15k tokens of PDF context, trimmed from the end

# --- FILE MANAGER ---
def _safe_subject(name):
    """Keep subject names to letters/digits/space/-/_ so they can't escape sessions/."""
    return re.sub(r"[^\w \-]", "", name or "").strip()

def _session_path(subject_name):
    return os.path.join(SESSIONS_DIR, f"session_{subject_name}.json")

def get_session_files():
    files = glob.glob(os.path.join(SESSIONS_DIR, "session_*.json"))
    return [os.path.basename(f).replace("session_", "").replace(".json", "") for f in files]

def save_session(subject_name):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    data = {
        "messages": st.session_state.messages,
        "pdf_context": st.session_state.pdf_context,
        "mode": st.session_state.mode,
        "phase": st.session_state.phase,
        "flashcards": st.session_state.flashcards,
        "last_active": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(_session_path(subject_name), "w") as f:
        json.dump(data, f)

def load_session(subject_name):
    path = _session_path(subject_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            st.session_state.messages = data.get("messages", [])
            st.session_state.pdf_context = data.get("pdf_context", "")
            # Old sessions lack mode/phase — default to Tutor so they load unchanged.
            st.session_state.mode = data.get("mode", "Tutor")
            st.session_state.phase = data.get("phase", "intro")
            st.session_state.flashcards = data.get("flashcards", [])
            return data.get("last_active", "Unknown")
    return None

def _show_usage(usage):
    """Small token/cache readout under a reply. Currently only Claude reports usage."""
    if usage is None:
        return
    inp = getattr(usage, "input_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    parts = [f"{inp:,} in"]
    if cached:
        parts.append(f"{cached:,} cached ⚡")
    if written:
        parts.append(f"{written:,} cache-write")
    parts.append(f"{out:,} out")
    st.caption("🧮 " + " · ".join(parts))

def _reset_card_widgets():
    """Drop per-card editor widget state so fields re-seed from the current cards
    after a save/redo/delete (Streamlit keeps widget state by key otherwise)."""
    for k in [k for k in st.session_state if k.startswith(("front_", "back_", "fb_"))]:
        del st.session_state[k]

# --- SIDEBAR ---
st.sidebar.title("🗂️ Subject Switcher")

model_options = list(PROVIDERS)
st.session_state.selected_model = st.sidebar.radio(
    "Model",
    model_options,
    index=model_options.index(st.session_state.selected_model),
)
st.session_state.deep_thinking = st.sidebar.checkbox(
    "🧠 Deep thinking",
    value=st.session_state.deep_thinking,
    help="Claude only — reasons harder before answering. Better for proofs/math, slower to start.",
)

chosen_mode = st.sidebar.radio(
    "Mode",
    MODES,
    index=MODES.index(st.session_state.mode),
    help="Tutor: Socratic Q&A · Quiz me: graded questions · Guided: Intro → Deep dive → Quiz",
)
if chosen_mode != st.session_state.mode:
    st.session_state.mode = chosen_mode
    # Entering Guided learning always starts at the introduction.
    if chosen_mode == "Guided learning":
        st.session_state.phase = "intro"
    save_session(st.session_state.current_subject)

st.sidebar.divider()

existing_subjects = get_session_files()
options = ["Create New..."] + existing_subjects
try:
    idx = options.index(st.session_state.current_subject)
except ValueError:
    idx = 0

selected_option = st.sidebar.radio("Select Subject:", options, index=idx)

if selected_option == "Create New...":
    new_name = st.sidebar.text_input("Name (e.g., Calculus):")
    if st.sidebar.button("Create & Switch"):
        clean_name = _safe_subject(new_name)
        if not clean_name:
            st.sidebar.warning("Use letters, numbers, spaces, - or _ in the name.")
        elif clean_name in existing_subjects:
            st.sidebar.warning(f"Subject '{clean_name}' already exists.")
        else:
            save_session(st.session_state.current_subject)
            st.session_state.current_subject = clean_name
            st.session_state.messages = []
            st.session_state.pdf_context = ""
            st.session_state.mode = "Tutor"
            st.session_state.phase = "intro"
            st.session_state.flashcards = []
            save_session(clean_name)
            st.rerun()
elif selected_option != st.session_state.current_subject:
    save_session(st.session_state.current_subject)
    st.session_state.current_subject = selected_option
    last_seen = load_session(selected_option)
    st.toast(f"Loaded {selected_option} (Last saved: {last_seen})", icon="📂")
    st.rerun()

st.sidebar.divider()
st.sidebar.write(f"**Current Subject:** {st.session_state.current_subject}")
uploaded_files = st.sidebar.file_uploader("Add Study Material", type=['pdf'], accept_multiple_files=True)

if uploaded_files and st.sidebar.button("Process & Update"):
    text = st.session_state.pdf_context
    for f in uploaded_files:
        reader = PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    st.session_state.pdf_context = text
    save_session(st.session_state.current_subject)
    st.sidebar.success(f"✅ Added to {st.session_state.current_subject}")

# Scan a photo's content into the subject's study material (vision -> text).
scan_img = st.sidebar.file_uploader("🖼 Scan image into notes", type=["png", "jpg", "jpeg"], key="scan_img")
if scan_img and st.sidebar.button("Scan & Add to notes"):
    provider = PROVIDERS[st.session_state.selected_model]
    if hasattr(provider, "thinking"):
        provider.thinking = False
    scan_system = (
        "You transcribe images into clean study notes. Extract ALL content from the image — "
        "text, equations, and the labeled parts of any diagram — as plain study text. No commentary."
    )
    try:
        img = imageutil.prepare_image(scan_img)
        with st.spinner("Scanning image…"):
            text = "".join(
                provider.chat(scan_system, [], "Extract everything from this image.", images=[img])
            )
        if text.strip():
            st.session_state.pdf_context += f"\n\n[SCANNED IMAGE]\n{text.strip()}"
            save_session(st.session_state.current_subject)
            st.sidebar.success("✅ Scanned into notes")
        else:
            st.sidebar.warning("No content extracted from the image.")
    except MissingAPIKey as e:
        st.sidebar.warning(f"🔑 No `{e}` in your `.env`.")
    except Exception as e:
        st.sidebar.error(f"Scan failed: {e}")

# Optional: clear context per subject
if st.sidebar.button("🗑️ Clear PDF context for this subject"):
    st.session_state.pdf_context = ""
    save_session(st.session_state.current_subject)
    st.rerun()

# --- MAIN CHAT AREA ---
st.title(f"Tutor: {st.session_state.current_subject}")
st.caption(f"Mode: `{st.session_state.mode}`  ·  Model: `{st.session_state.selected_model}`")

if st.session_state.pdf_context:
    st.caption(f"🟢 Context Loaded ({len(st.session_state.pdf_context):,} chars) | Memory Active")
else:
    st.caption("⚪ Empty Context — Upload PDFs to start")

# --- FLASHCARDS PANEL ---
_cards = st.session_state.flashcards
with st.expander(f"📇 Flashcards — {st.session_state.current_subject} ({len(_cards)})", expanded=bool(_cards)):
    if st.button("📇 Make flashcards from this session"):
        convo = st.session_state.messages[-40:]
        while convo and convo[0]["role"] != "user":
            convo = convo[1:]
        # Keep history ending on an assistant turn so the generate-instruction is the
        # only trailing user turn (avoids two consecutive user turns -> 400 on Claude).
        if convo and convo[-1]["role"] == "user":
            convo = convo[:-1]
        if not convo:
            st.warning("Have a study conversation first, then make flashcards from it.")
        else:
            provider = PROVIDERS[st.session_state.selected_model]
            if hasattr(provider, "thinking"):
                provider.thinking = False  # extraction doesn't need deep thinking
            pdf_ctx = st.session_state.pdf_context[-MAX_PDF_CHARS:]
            existing = [c["front"] for c in _cards]
            try:
                with st.spinner("Reviewing the session and writing flashcards…"):
                    new_cards = fc.generate_cards(
                        provider, st.session_state.current_subject, convo, pdf_ctx, existing
                    )
                if new_cards:
                    st.session_state.flashcards.extend(new_cards)
                    save_session(st.session_state.current_subject)
                    st.success(f"Added {len(new_cards)} new card(s).")
                    st.rerun()
                else:
                    st.info("No new cards — the deck already covers this session.")
            except MissingAPIKey as e:
                st.warning(f"🔑 No `{e}` in your `.env` — add it to use {st.session_state.selected_model}.")
            except Exception as e:
                st.error(f"Couldn't generate flashcards: {e}")

    if _cards:
        col_csv, col_anki = st.columns(2)
        with col_csv:
            st.download_button(
                "⬇ Download CSV",
                data=fc.to_csv(_cards),
                file_name=f"{st.session_state.current_subject}_flashcards.csv",
                mime="text/csv",
            )
        with col_anki:
            if st.button("📤 Send to Anki"):
                try:
                    added = fc.send_to_anki(st.session_state.current_subject, _cards)
                    st.success(f"Sent — {added} new card(s) in Anki deck `Tutor::{st.session_state.current_subject}`.")
                except Exception as e:
                    st.warning(
                        "Couldn't reach Anki. Open Anki (with the AnkiConnect add-on installed) "
                        f"and retry — or use the CSV download. [{e}]"
                    )
        st.divider()
        st.caption("Edit a card and **Save**, or describe a change and **Redo**. 🗑 deletes.")
        for i, c in enumerate(_cards):
            new_front = st.text_input(f"Front — card {i + 1}", value=c["front"], key=f"front_{i}")
            new_back = st.text_area(
                "Back", value=c["back"], key=f"back_{i}", height=80, label_visibility="collapsed"
            )
            fb = st.text_input(
                "Change request", key=f"fb_{i}", label_visibility="collapsed",
                placeholder="Ask to change this card — e.g. 'this is wrong, fix it' / 'ask about the proof instead' / 'make it simpler'",
            )
            b_save, b_redo, b_del = st.columns(3)
            if b_save.button("💾 Save", key=f"save_{i}"):
                st.session_state.flashcards[i] = {"front": new_front.strip(), "back": new_back.strip()}
                save_session(st.session_state.current_subject)
                _reset_card_widgets()
                st.rerun()
            if b_redo.button("🔁 Redo", key=f"redo_{i}"):
                provider = PROVIDERS[st.session_state.selected_model]
                if hasattr(provider, "thinking"):
                    provider.thinking = False
                convo = st.session_state.messages[-40:]
                while convo and convo[0]["role"] != "user":
                    convo = convo[1:]
                pdf_ctx = st.session_state.pdf_context[-MAX_PDF_CHARS:]
                try:
                    with st.spinner("Reworking that card…"):
                        revised = fc.regenerate_card(
                            provider, st.session_state.current_subject, convo, pdf_ctx, c,
                            fb.strip() or "Give a different, improved version of this card.",
                        )
                    if revised:
                        st.session_state.flashcards[i] = revised
                        save_session(st.session_state.current_subject)
                        _reset_card_widgets()
                        st.rerun()
                    else:
                        st.warning("Couldn't rework that card — try rephrasing your request.")
                except MissingAPIKey as e:
                    st.warning(f"🔑 No `{e}` in your `.env` — add it to use {st.session_state.selected_model}.")
                except Exception as e:
                    st.error(f"Redo failed: {e}")
            if b_del.button("🗑 Delete", key=f"del_{i}"):
                st.session_state.flashcards.pop(i)
                save_session(st.session_state.current_subject)
                _reset_card_widgets()
                st.rerun()
            st.divider()
    else:
        st.caption("No flashcards yet — study, then click the button above.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# A turn is driven by either the "Next →" button (Guided mode) or the chat input.
pending = None

if st.session_state.mode == "Guided learning":
    phase = st.session_state.phase
    st.caption(f"📚 Stage: **{PHASE_LABELS[phase]}**  (Introduction → Deep dive → Quiz)")
    has_reply = any(m["role"] == "assistant" for m in st.session_state.messages)
    if phase != "quiz" and has_reply:
        next_phase = GUIDED_PHASES[GUIDED_PHASES.index(phase) + 1]
        if st.button(f"Next → {PHASE_LABELS[next_phase]}"):
            st.session_state.phase = next_phase
            pending = "Let's go deeper." if next_phase == "deep_dive" else "Quiz me on what we just covered."

quiz_active = st.session_state.mode == "Quiz me" or (
    st.session_state.mode == "Guided learning" and st.session_state.phase == "quiz"
)
placeholder = "Type your answer..." if quiz_active else f"Continue studying {st.session_state.current_subject}..."
pending_files = []
chat_val = st.chat_input(placeholder, accept_file="multiple", file_type=["png", "jpg", "jpeg"])
if chat_val:
    pending = chat_val.text or ""
    pending_files = list(chat_val.files or [])
    if not pending and pending_files:
        pending = "Please look at the attached image and help me with it."

if pending:
    # Prepare any attached photos for the vision model (this turn only — not stored).
    images = None
    if pending_files:
        try:
            images = [imageutil.prepare_image(f) for f in pending_files]
        except Exception as e:
            st.error(f"Couldn't read the attached image: {e}")
            st.stop()
    user_content = f"{pending}  📷" if pending_files else pending
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(pending)
        for f in pending_files:
            f.seek(0)
            st.image(f, width=240)

    # History = prior turns (trimmed); the new user message is passed separately.
    history = st.session_state.messages[:-1][-MAX_HISTORY_TURNS:]
    # Providers (Claude especially) require the history to start with a user turn.
    while history and history[0]["role"] != "user":
        history = history[1:]

    # Trim PDF context (keep the tail — usually the most recently added material).
    pdf_ctx = st.session_state.pdf_context
    if len(pdf_ctx) > MAX_PDF_CHARS:
        pdf_ctx = "…[earlier content trimmed]…\n" + pdf_ctx[-MAX_PDF_CHARS:]

    # Mode/phase persona + PDF form one stable block (cached by the Claude provider).
    system = build_system(
        st.session_state.mode, st.session_state.phase, st.session_state.current_subject
    )
    if pdf_ctx:
        system += f"\n\n[PDF KNOWLEDGE BASE]\n{pdf_ctx}"

    model_name = st.session_state.selected_model
    provider = PROVIDERS[model_name]
    # Deep-thinking toggle applies only to Claude (the only provider with .thinking).
    if hasattr(provider, "thinking"):
        provider.thinking = st.session_state.deep_thinking

    turn_ok = False
    with st.chat_message("assistant"):
        try:
            full = st.write_stream(provider.chat(system, history, pending, images=images))
            st.session_state.messages.append(
                {"role": "assistant", "content": full, "model": model_name}
            )
            save_session(st.session_state.current_subject)
            _show_usage(getattr(provider, "last_usage", None))
            turn_ok = True
        except MissingAPIKey as e:
            st.warning(f"🔑 No `{e}` in your `.env` — add it to use {model_name}.")
            st.session_state.messages.pop()
        except Exception as e:
            st.error(f"Error from {model_name}: {e}")
            # Drop the un-answered user turn so the saved history stays consistent.
            st.session_state.messages.pop()

    # Guided mode: rerun so the "Next →" button (decided earlier in the script)
    # appears right after an intro/deep-dive reply, not on the next interaction.
    if turn_ok and st.session_state.mode == "Guided learning" and st.session_state.phase != "quiz":
        st.rerun()

