from __future__ import annotations

import json
import tempfile
import traceback
from collections import Counter
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from art_recognition.identity import DEFAULT_DINOV2_MODEL
from art_recognition.pipeline import (
    ArtRecognitionPipeline,
    preprocess_query_image_variants_from_bgr,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BUILD_REPORT_PATH = DATA_DIR / "build_report.json"
STYLE_PREDICTIONS_PATH = DATA_DIR / "armenian_style_predictions.csv"
INDEX_PATH = DATA_DIR / "faiss_index.idx"
MAPPING_PATH = DATA_DIR / "index_mapping.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CLASSIFIER_PATH = DATA_DIR / "style_classifier.pkl"
STREAMLIT_ERROR_LOG_PATH = DATA_DIR / "last_streamlit_error.log"
RECOGNITION_DEBUG_LOG_PATH = DATA_DIR / "recognition_debug_log.jsonl"
BIRTH_OF_VENUS_URL = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Botticelli_Venus.jpg"
MAX_UPLOAD_MB = 20
PUBLIC_MODEL_NAME = DEFAULT_DINOV2_MODEL
PUBLIC_TOP_K = 20


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _log_streamlit_exception(exc: Exception) -> None:
    STREAMLIT_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    STREAMLIT_ERROR_LOG_PATH.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


def page_css() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(211, 164, 78, 0.18), transparent 32rem),
                linear-gradient(180deg, #7f1827 0%, #8d2130 42%, #351316 100%);
            color: #2d1914;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        div.block-container {
            padding-top: 1rem;
            max-width: 1120px;
        }
        .hero {
            padding: 5.5rem 1.2rem 2.5rem;
            color: #fff8e8;
            background:
                linear-gradient(180deg, rgba(33, 13, 12, 0.28), rgba(44, 12, 17, 0.82)),
                linear-gradient(115deg, rgba(119, 24, 37, 0.54), rgba(30, 12, 10, 0.4)),
                url('"""
        + BIRTH_OF_VENUS_URL
        + """') center/cover no-repeat;
            min-height: 520px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            margin: -1rem -1rem 1.4rem;
            border-bottom: 1px solid rgba(238, 205, 132, 0.38);
        }
        .hero h1 {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: clamp(2.4rem, 8vw, 5.7rem);
            margin-bottom: 0.6rem;
            line-height: 0.98;
            letter-spacing: 0;
            max-width: 860px;
        }
        .hero p {
            font-size: 1.08rem;
            max-width: 760px;
            line-height: 1.6;
            color: #fff1d6;
        }
        .ornament {
            color: #e8c16d;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.6rem;
        }
        .intro-band, .analysis-band {
            background: #fff7e8;
            border: 1px solid rgba(232, 193, 109, 0.55);
            border-radius: 8px;
            padding: clamp(1rem, 3vw, 1.6rem);
            box-shadow: 0 20px 45px rgba(31, 11, 13, 0.24);
            margin-bottom: 1rem;
        }
        .intro-band h2, .analysis-band h2 {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: clamp(1.6rem, 5vw, 2.4rem);
            margin: 0 0 0.4rem;
            color: #671622;
            letter-spacing: 0;
        }
        .intro-band p {
            color: #4d382f;
            line-height: 1.65;
            margin-bottom: 0;
        }
        .result-panel {
            background: linear-gradient(180deg, #fffaf0, #f7ead2);
            border: 1px solid rgba(174, 125, 40, 0.42);
            border-radius: 8px;
            padding: 1.1rem;
            margin: 0.8rem 0 1rem;
        }
        .result-panel h3 {
            margin: 0 0 0.35rem;
            font-family: Georgia, 'Times New Roman', serif;
            color: #5c1320;
            font-size: 1.55rem;
        }
        .confidence-row {
            color: #6d5142;
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.9rem 0 1.1rem;
        }
        .insight-card {
            background: #fffaf0;
            border: 1px solid rgba(174, 125, 40, 0.35);
            border-radius: 8px;
            padding: 0.9rem;
            min-height: 112px;
        }
        .insight-card span {
            display: block;
            color: #7b5c45;
            font-size: 0.86rem;
            margin-bottom: 0.25rem;
        }
        .insight-card strong {
            display: block;
            color: #641724;
            font-family: Georgia, 'Times New Roman', serif;
            font-size: clamp(1.05rem, 2.5vw, 1.35rem);
            line-height: 1.18;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .insight-card small {
            display: block;
            color: #6d5142;
            margin-top: 0.35rem;
            line-height: 1.35;
        }
        .curator-panel {
            background: #fff2d3;
            border: 1px solid rgba(232, 193, 109, 0.48);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #2d1914;
        }
        .curator-panel h3 {
            margin: 0 0 0.35rem;
            font-family: Georgia, 'Times New Roman', serif;
            color: #641724;
            font-size: 1.45rem;
        }
        .reference-card {
            background: #fff7e8;
            border: 1px solid rgba(232, 193, 109, 0.42);
            border-radius: 8px;
            padding: 0.9rem;
            margin-bottom: 0.85rem;
        }
        div[data-testid="stChatMessage"] {
            background: #fff7e8;
            border: 1px solid rgba(100, 23, 36, 0.18);
            border-radius: 8px;
            color: #2d1914;
            padding: 0.4rem 0.75rem;
        }
        div[data-testid="stChatMessage"] p {
            color: #2d1914;
        }
        div[data-testid="stTextInput"] input {
            background: #fffaf0;
            color: #2d1914;
            border: 1px solid rgba(100, 23, 36, 0.4);
        }
        .reference-card strong {
            color: #641724;
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 1.05rem;
        }
        .small-note {
            color: #6d5142;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: #fffaf0;
            border: 1px solid rgba(174, 125, 40, 0.35);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        div[data-testid="stMetricValue"] {
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.15;
        }
        div[data-testid="stFileUploader"] {
            background: #fffaf0;
            border: 1px dashed rgba(127, 24, 39, 0.48);
            border-radius: 8px;
            padding: 0.9rem;
        }
        .stButton button, div[data-testid="stFileUploader"] button {
            background: #7f1827;
            color: #fff8e8;
            border: 1px solid #d8af55;
            border-radius: 6px;
        }
        @media (max-width: 640px) {
            .hero {
                min-height: 430px;
                padding: 3rem 1rem 1.5rem;
            }
            .intro-band, .analysis-band, .result-panel, .reference-card, .curator-panel {
                padding: 0.85rem;
            }
            .insight-grid {
                grid-template-columns: 1fr;
            }
            section.main > div {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_build_report(cache_key: float = 0.0) -> dict[str, object] | None:
    if not BUILD_REPORT_PATH.exists():
        return None
    return json.loads(BUILD_REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_style_predictions(cache_key: float = 0.0) -> pd.DataFrame:
    if not STYLE_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(STYLE_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_mapping(cache_key: float = 0.0) -> list[dict[str, object]]:
    if not MAPPING_PATH.exists():
        return []
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_embeddings(cache_key: float = 0.0) -> np.ndarray | None:
    if not EMBEDDINGS_PATH.exists():
        return None
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


@st.cache_resource(show_spinner=False)
def get_pipeline() -> ArtRecognitionPipeline:
    return ArtRecognitionPipeline(project_root=PROJECT_ROOT)


def render_intro_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="ornament">A quiet studio for painting discovery</div>
            <h1>Discover The Story In A Painting</h1>
            <p>
                Upload a photo of a painting and receive the clearest information the system can offer:
                a possible artwork match, artist and year when recognized, visual style, likely genre,
                and related paintings for comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-band">
            <h2>What You Can Learn</h2>
            <p>
                This website is made for curious viewers, students, collectors, and museum visitors.
                It can help identify a painting when the match is strong. When the exact painting is not found,
                it still studies the image and suggests the most likely artistic style, genre, and carefully chosen
                reference works that look visually related.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_page() -> None:
    report = load_build_report(_file_mtime(BUILD_REPORT_PATH))
    style_df = load_style_predictions(_file_mtime(STYLE_PREDICTIONS_PATH))
    mapping = load_mapping(_file_mtime(MAPPING_PATH))

    st.title("Project Progress")
    if report is None:
        st.warning("No build report found yet. Build the index first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Indexed", int(report.get("total_records", 0)))
    col2.metric("Armenian", int(report.get("armenian_records", 0)))
    col3.metric("WikiArt", int(report.get("wikiart_records", 0)))
    col4.metric("Style Classes", int(report.get("style_classes", 0)))

    st.json(report)

    if mapping:
        source_counts = Counter(record.get("source", "unknown") for record in mapping)
        st.subheader("Dataset Composition")
        st.bar_chart(pd.DataFrame({"count": source_counts}).T)

    if not style_df.empty:
        st.subheader("Armenian Style Predictions")
        st.dataframe(style_df.head(25), use_container_width=True, hide_index=True)

        style_counts = style_df["predicted_style"].value_counts().head(12)
        st.subheader("Most Common Predicted Styles")
        st.bar_chart(style_counts)

        st.subheader("High-Confidence Examples")
        top_conf = style_df.sort_values("predicted_style_confidence", ascending=False).head(12)
        st.dataframe(top_conf, use_container_width=True, hide_index=True)

    st.subheader("Saved Artifacts")
    artifacts = []
    for path in [INDEX_PATH, MAPPING_PATH, EMBEDDINGS_PATH, CLASSIFIER_PATH, STYLE_PREDICTIONS_PATH]:
        artifacts.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    st.dataframe(pd.DataFrame(artifacts), use_container_width=True, hide_index=True)


def _safe_open_image(uploaded_file) -> Image.Image:
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        converted = ImageOps.exif_transpose(image).convert("RGB")
        uploaded_file.seek(0)
        return converted
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def _array_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def _save_temp_rgb_image(image: Image.Image) -> Path:
    temp_dir = PROJECT_ROOT / "data" / "user_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
    image.save(temp_path)
    return temp_path


def _append_recognition_debug_log(
    upload_path: Path,
    result: dict[str, object],
    top_k: int,
) -> None:
    if result.get("is_recognized"):
        return
    top_matches = []
    for match in result.get("similar_paintings", [])[:top_k]:
        metadata = match.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        top_matches.append(
            {
                "rank": match.get("rank"),
                "score": match.get("score"),
                "predicted_label": metadata.get("image_path") or metadata.get("filename"),
                "title": metadata.get("title"),
                "artist": metadata.get("artist"),
                "style": metadata.get("style") or metadata.get("predicted_style"),
            }
        )
    row = {
        "query_image_id": str(upload_path),
        "ground_truth_label": None,
        "predicted_label": top_matches[0]["predicted_label"] if top_matches else None,
        "threshold_used": result.get("recognition_threshold"),
        "top_1_score": result.get("recognition_score"),
        "query_variant": result.get("query_variant"),
        "is_near_threshold": result.get("is_near_threshold"),
        "top_5": top_matches,
    }
    RECOGNITION_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECOGNITION_DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_query(
    upload_path: Path,
    model_name: str,
    top_k: int,
    details: dict[str, object],
) -> dict[str, object]:
    del details
    if not MAPPING_PATH.exists() or not EMBEDDINGS_PATH.exists() or not INDEX_PATH.exists():
        raise RuntimeError("The index is not available yet. Build the project first.")
    result = get_pipeline().query(
        upload_path,
        embedding_model=model_name,
        top_k=top_k,
    )
    _append_recognition_debug_log(upload_path, result, top_k)
    return result


def preprocess_query_image_with_details(image_path: str | Path) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError("Could not read the uploaded image.")

    query_variants = preprocess_query_image_variants_from_bgr(image)
    first_variant = query_variants[0]
    return {
        "processed_rgb": first_variant["processed_rgb"],
        "query_variants": query_variants,
        "annotated_rgb": first_variant["processed_rgb"],
        "candidate_count": 1,
        "message": f"Painting crop prepared with {first_variant['name']}.",
        "candidate_previews": [first_variant["processed_rgb"]],
    }


def _display_value(value: object, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "unknown"}:
        return fallback
    return _to_display_case(text)


def _to_display_case(value: str) -> str:
    text = value.replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Unknown"
    small_words = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    words = []
    for index, word in enumerate(text.split()):
        lower = word.lower()
        if index > 0 and lower in small_words:
            words.append(lower)
        else:
            words.append(lower[:1].upper() + lower[1:])
    return " ".join(words)


def _result_signature(result: dict[str, object]) -> str:
    return "|".join(
        [
            str(result.get("recognition_status")),
            str(result.get("recognized_painting")),
            str(result.get("predicted_style")),
            str(result.get("inferred_genre")),
            str(result.get("recognition_score")),
        ]
    )


def _best_story_metadata(result: dict[str, object]) -> dict[str, object]:
    if result.get("is_recognized"):
        return {
            "title": result.get("recognized_painting"),
            "artist": result.get("artist"),
            "year": result.get("year"),
            "style": result.get("predicted_style"),
            "genre": result.get("inferred_genre"),
        }

    matches = result.get("similar_paintings") or []
    if matches:
        metadata = matches[0].get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
    return {}


def _style_note(style: str) -> str:
    notes = {
        "Abstract Expressionism": "This style often values gesture, movement, and emotional force over precise representation.",
        "Art Nouveau Modern": "This style is often recognized by decorative rhythm, elegant lines, and ornamental surfaces.",
        "Baroque": "Baroque art usually feels dramatic, theatrical, and rich in light-shadow contrast.",
        "Color Field Painting": "Color Field works often use broad areas of color to create mood and visual atmosphere.",
        "Cubism": "Cubism breaks forms into geometric viewpoints, so the subject can feel seen from several angles at once.",
        "Expressionism": "Expressionism usually bends color and form toward emotion rather than realism.",
        "Impressionism": "Impressionism often focuses on light, atmosphere, and the feeling of a passing moment.",
        "Minimalism": "Minimalism reduces the image to essential forms, colors, or repeated structures.",
        "Naive Art Primitivism": "This style often uses direct forms, simplified perspective, and a deliberately plain visual language.",
        "Pop Art": "Pop Art borrows energy from popular culture, graphic design, advertising, and everyday images.",
        "Post Impressionism": "Post-Impressionism keeps vivid color but often adds stronger structure and personal expression.",
        "Realism": "Realism aims for believable everyday subjects, observed detail, and a grounded sense of life.",
        "Romanticism": "Romanticism often emphasizes drama, nature, emotion, memory, and the sublime.",
        "Symbolism": "Symbolism uses visible forms to suggest dreams, myths, inner states, or spiritual ideas.",
        "Ukiyo e": "Ukiyo-e is associated with Japanese print traditions, flattened space, clean contour, and elegant composition.",
    }
    return notes.get(style, "The predicted style gives a useful clue about the painting's visual language and historical family.")


def _genre_note(genre: str) -> str:
    notes = {
        "Cityscape": "A cityscape usually asks us to read architecture, streets, and public space as the main subject.",
        "Figure scene": "A figure scene usually places human presence, gesture, or social action at the center.",
        "Historical or religious scene": "This kind of image often points toward narrative, memory, ritual, or collective identity.",
        "Landscape": "A landscape directs attention to place, atmosphere, distance, and the character of nature.",
        "Portrait": "A portrait is usually about identity, presence, status, psychology, or remembrance.",
        "Seascape": "A seascape often uses water, horizon, weather, and light to create emotion and scale.",
        "Still life": "A still life turns objects into a careful study of color, texture, symbolism, and composition.",
    }
    return notes.get(genre, "The inferred genre describes the kind of subject the system sees most strongly.")


def _curator_answer(question: str, result: dict[str, object]) -> str:
    question_lower = question.lower()
    metadata = _best_story_metadata(result)
    title = _display_value(metadata.get("title") or result.get("recognized_painting"), "this painting")
    artist = _display_value(metadata.get("artist") or result.get("artist"), "an unknown painter")
    year = _display_value(metadata.get("year") or result.get("year"), "an unknown date")
    style = _display_value(result.get("predicted_style") or metadata.get("style"), "Unknown")
    genre = _display_value(result.get("inferred_genre") or metadata.get("genre"), "Unknown")

    if any(word in question_lower for word in ["artist", "painter", "who"]):
        if result.get("is_recognized"):
            return f"The strongest match is {title}, painted by {artist}. The recorded year is {year}. A good first reading is to compare the artist's subject choice with the predicted {style} style."
        possible_artist = result.get("possible_artist")
        if possible_artist:
            return f"I would treat the painter as a cautious possibility, not a fact: the similar works point toward {_display_value(possible_artist)}, but the exact painting was not recognized."
        return "I would not name a painter from this result. The visual matches are not strong enough, so the safer reading is to discuss style, genre, and related works."

    if any(word in question_lower for word in ["style", "movement"]):
        return f"The predicted style is {style}. {_style_note(style)}"

    if any(word in question_lower for word in ["genre", "subject", "scene"]):
        return f"The likely genre is {genre}. {_genre_note(genre)}"

    if any(word in question_lower for word in ["match", "recognized", "sure", "confidence"]):
        if result.get("is_recognized"):
            return f"This looks like a strong recognition: the match confidence is {float(result.get('recognition_score') or 0):.2f}. I would still present it as model-assisted identification, not museum authentication."
        return f"The exact painting was not found. The best visual match scored {float(result.get('recognition_score') or 0):.2f}, so the app gives descriptive analysis instead of claiming an identity."

    if any(word in question_lower for word in ["interesting", "tell", "story", "short", "about"]):
        if result.get("is_recognized"):
            return f"Short note: {title} by {artist} ({year}) is read here through a {style} lens. The most interesting thing to look for is how the composition creates mood before it gives details."
        return f"Short note: I cannot safely identify the exact painting, but its closest visual reading is {style} and {genre}. Look first at color, light, and composition; those are what guided the system's comparison."

    return f"I can help read this result. The key clues are: style {style}, genre {genre}, and {'a strong exact match' if result.get('is_recognized') else 'no exact match'}. Ask me about the artist, style, genre, or confidence."


def render_curator_chat(result: dict[str, object]) -> None:
    signature = _result_signature(result)
    if st.session_state.get("curator_signature") != signature:
        st.session_state.curator_signature = signature
        st.session_state.curator_messages = [
            {
                "role": "assistant",
                "content": _curator_answer("tell me something interesting", result),
            }
        ]

    st.markdown(
        """
        <div class="curator-panel">
            <h3>Curator Chat</h3>
            <p class="small-note">
                Ask a short question about the painting, painter, style, genre, or confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.curator_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.form("curator_chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask The Curator", placeholder="Ask about the painter, style, genre, or confidence...")
        submitted = st.form_submit_button("Send")
    if submitted and user_question.strip():
        question = user_question.strip()
        st.session_state.curator_messages.append({"role": "user", "content": question})
        answer = _curator_answer(question, result)
        st.session_state.curator_messages.append({"role": "assistant", "content": answer})
        st.rerun()


def render_prediction_page() -> None:
    if not INDEX_PATH.exists() or not EMBEDDINGS_PATH.exists() or not MAPPING_PATH.exists():
        st.error("The painting recognition service is not ready yet.")
        return

    st.markdown(
        """
        <div class="analysis-band">
            <h2>Upload A Painting</h2>
            <p class="small-note">
                Use a clear photo, scan, or screenshot. Cropped images usually give the best results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a painting image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
        return

    try:
        image = _safe_open_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_np = np.array(image)
    if image_np.shape[0] < 64 or image_np.shape[1] < 64:
        st.error("The uploaded image is too small. Please use a larger image.")
        return

    st.image(image_np, caption="Your uploaded painting", use_container_width=True)

    temp_path = _save_temp_rgb_image(image)
    with st.spinner("Studying the image..."):
        try:
            details = preprocess_query_image_with_details(temp_path)
            result = _predict_query(
                temp_path,
                model_name=PUBLIC_MODEL_NAME,
                top_k=PUBLIC_TOP_K,
                details=details,
            )
        except Exception as exc:
            _log_streamlit_exception(exc)
            st.error(f"Analysis failed: {exc}. Details were saved to data/last_streamlit_error.log.")
            return

    st.markdown('<div class="ornament">Analysis</div>', unsafe_allow_html=True)
    if result["is_recognized"]:
        st.markdown(
            f"""
            <div class="result-panel">
                <h3>{escape(_display_value(result.get("recognized_painting"), "Recognized Painting"))}</h3>
                <div>{escape(_display_value(result.get("artist"), "Unknown Artist"))} · {escape(_display_value(result.get("year"), "Unknown Year"))}</div>
                <div class="confidence-row">Match Confidence: {result["recognition_score"]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        possible_artist = result.get("possible_artist")
        artist_text = (
            f"{escape(_display_value(possible_artist))} ({result['possible_artist_confidence']:.2f})"
            if possible_artist
            else "No Reliable Artist Attribution"
        )
        st.markdown(
            f"""
            <div class="result-panel">
                <h3>Exact Painting Not Found</h3>
                <div>The Image Does Not Strongly Match A Known Artwork In The Collection.</div>
                <div class="confidence-row">Possible Painter: {artist_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result.get("is_near_threshold") and isinstance(result.get("near_match_candidate"), dict):
            candidate = result["near_match_candidate"]
            st.info(
                "Closest candidate is near the recognition threshold: "
                f"{_display_value(candidate.get('title'), 'Untitled')} "
                f"by {_display_value(candidate.get('artist'))} "
                f"with score {float(result.get('recognition_score') or 0):.2f}. "
                "It is shown as unrecognized to avoid a false claim."
            )

    style_text = _display_value(result.get("predicted_style"))
    style_confidence = float(result.get("predicted_style_confidence") or 0.0)
    genre_text = _display_value(result.get("inferred_genre"))
    genre_confidence = float(result.get("inferred_genre_confidence") or 0.0)
    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <span>Style</span>
                <strong>{escape(style_text)}</strong>
                <small>{style_confidence:.2f} confidence</small>
            </div>
            <div class="insight-card">
                <span>Genre</span>
                <strong>{escape(genre_text)}</strong>
                <small>{genre_confidence:.2f} confidence</small>
            </div>
            <div class="insight-card">
                <span>Result Type</span>
                <strong>{"Recognized" if result["is_recognized"] else "Visual analysis"}</strong>
                <small>{"The artwork match is strong." if result["is_recognized"] else "The exact artwork was not claimed."}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_curator_chat(result)

    st.markdown('<div class="ornament">Related Works</div>', unsafe_allow_html=True)
    for match in result["similar_paintings"]:
        metadata = match["metadata"]
        with st.container():
            st.markdown(
                f"""
                <div class="reference-card">
                    <strong>{escape(_display_value(metadata.get('title'), 'Untitled'))}</strong><br/>
                    Artist: {escape(_display_value(metadata.get('artist')))}<br/>
                    Year: {escape(_display_value(metadata.get('year')))}<br/>
                    Style: {escape(_display_value(metadata.get('style') or metadata.get('predicted_style')))}<br/>
                    Visual Closeness: {match['score']:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )
            image_path = metadata.get("processed_image_path")
            if image_path and Path(image_path).exists():
                st.image(str(image_path), caption=_display_value(metadata.get("title") or metadata.get("filename")), width=280)


def main() -> None:
    st.set_page_config(
        page_title="Painting Discovery",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    page_css()

    render_intro_page()
    render_prediction_page()


if __name__ == "__main__":
    main()
