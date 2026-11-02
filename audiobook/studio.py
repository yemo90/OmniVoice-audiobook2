#!/usr/bin/env python3
"""OmniVoice Studio — voice cloning, voice design and audiobook creation.

A thin Gradio front-end over ``omnivoice.OmniVoice`` that adds a third
"Audiobook" tab: drop a book file, pick a voice, and get a single audio file
named after the book.

Launched by the Pinokio ``start.js`` script:

    python ../audiobook/studio.py --ip 127.0.0.1 --port <port>
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

import audio_io
import text_io
import voice_store
from audio_io import ExportError
from text_io import ExtractionError
from voice_store import VoiceError

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

SYSTEMS = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

VOICE_MODES = ["Saved voice", "Reference audio", "Voice design", "Automatic"]

DESIGN_INFO = (
    "Comma-separated attributes, freely combined: gender (male/female), age "
    "(child, teenager, young adult, middle-aged, elderly), pitch (very low … very "
    "high), style (whisper), English accent (american, british, …), or a Chinese "
    "dialect (四川话, 陕西话 …). Example: `female, low pitch, british accent`"
)

MODEL: OmniVoice | None = None


def log(message: str) -> None:
    logging.info(message)


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


_PROMPT_CACHE: dict[str, object] = {}


def _language_value(language: str | None):
    return None if not language or language == "Auto" else language


def _saved_prompt(voice: dict):
    cache_key = voice["name"]
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    prompt = voice_store.load_prompt(voice, model=MODEL)
    if prompt is not None:
        _PROMPT_CACHE[cache_key] = prompt
    return prompt


def resolve_voice(mode: str, saved: str | None, ref_audio: str | None,
                  ref_text: str | None, instruct: str | None) -> dict:
    """Turn the UI selection into something ``model.generate`` understands."""
    if mode == "Saved voice":
        if not saved:
            raise VoiceError("Pick a saved voice, or choose another voice source.")
        voice = voice_store.get_voice(saved)
        if not voice:
            raise VoiceError(f"Saved voice '{saved}' no longer exists.")

        if voice["kind"] == "design":
            return {"kind": "design", "instruct": voice["instruct"]}

        prompt = _saved_prompt(voice)
        if prompt is None:
            raise VoiceError(
                f"The reference clip for '{saved}' is missing. Save the voice again."
            )
        return {"kind": "clone", "prompt": prompt}

    if mode == "Reference audio":
        if not ref_audio:
            raise VoiceError("Upload or record a reference clip first.")
        return {
            "kind": "clone",
            "prompt": MODEL.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=(ref_text or "").strip() or None
            ),
        }

    if mode == "Voice design":
        instruct = (instruct or "").strip()
        if not instruct:
            raise VoiceError("Describe the voice you want (e.g. 'male, british accent').")
        return {"kind": "design", "instruct": instruct}

    return {"kind": "auto"}


def synthesize(text: str, voice: dict, language: str | None, num_step: int,
               speed: float = 1.0) -> np.ndarray:
    """Generate one segment of speech."""
    kwargs = {
        "text": text,
        "language": _language_value(language),
        "generation_config": OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        ),
    }
    if voice["kind"] == "clone":
        kwargs["voice_clone_prompt"] = voice["prompt"]
    elif voice["kind"] == "design":
        kwargs["instruct"] = voice["instruct"]
    if speed and abs(float(speed) - 1.0) > 1e-3:
        kwargs["speed"] = float(speed)

    audio = MODEL.generate(**kwargs)
    return np.asarray(audio[0], dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------


def _voice_choices(selected: str | None = None):
    names = voice_store.voice_names()
    return gr.update(choices=names, value=selected if selected in names else (names[0] if names else None))


def refresh_voices():
    return _voice_choices()


def load_saved_voice_clip(saved: str | None):
    """Fill the reference audio + transcript boxes from a saved voice."""
    voice = voice_store.get_voice(saved) if saved else None
    if not voice:
        return gr.update(), gr.update()
    reference = voice_store.reference_path(voice)
    return (
        str(reference) if reference else None,
        voice.get("ref_text") or voice.get("instruct") or "",
    )


def save_voice_from_clip(name, ref_audio, ref_text, language):
    try:
        prompt = None
        if ref_audio:
            prompt = MODEL.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=(ref_text or "").strip() or None
            )
        voice = voice_store.save_clone_voice(name, ref_audio, ref_text, language, prompt)
        _PROMPT_CACHE.pop(voice["name"], None)
        return f"Saved voice **{voice['name']}** (reference clip).", _voice_choices(voice["name"])
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        return f"Could not save the voice: {exc}", _voice_choices()


def save_voice_from_design(name, instruct, language):
    try:
        voice = voice_store.save_design_voice(name, instruct, language)
        return f"Saved voice design **{voice['name']}**.", _voice_choices(voice["name"])
    except VoiceError as exc:
        return f"Could not save the voice: {exc}", _voice_choices()


def delete_saved_voice(saved):
    try:
        if not saved:
            return "Pick a voice to delete.", _voice_choices()
        voice_store.delete_voice(saved)
        _PROMPT_CACHE.pop(saved, None)
        return f"Deleted **{saved}**.", _voice_choices()
    except VoiceError as exc:
        return str(exc), _voice_choices()


def _save_single(audio: np.ndarray, stem: str, fmt: str) -> Path:
    out = audio_io.unique_path(OUTPUT_DIR, stem, fmt)
    return audio_io.export(audio, MODEL.sampling_rate, out, fmt), out


def generate_single(text, mode, saved, ref_audio, ref_text, instruct, language,
                    num_step, speed, fmt):
    if not text or not text.strip():
        return None, None, "Enter some text first."
    try:
        voice = resolve_voice(mode, saved, ref_audio, ref_text, instruct)
        log(f"Generating {len(text)} characters ({voice['kind']} voice)")
        audio = synthesize(text.strip(), voice, language, num_step, speed)
        path, _ = _save_single(audio, "speech", fmt)
        return str(path), str(path), f"Done — saved to `{path}`"
    except (VoiceError, ExportError) as exc:
        return None, None, f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log(traceback.format_exc())
        return None, None, f"Error: {type(exc).__name__}: {exc}"


def _book_paths(file_paths) -> list[str]:
    """Gradio hands back a list for multiple uploads, a single path for one."""
    if not file_paths:
        return []
    if isinstance(file_paths, (str, Path)):
        return [str(file_paths)]
    return [str(path) for path in file_paths if path]


def preview_book(file_paths, max_chars):
    paths = _book_paths(file_paths)
    if not paths:
        return "Drop one or more book files first."

    lines: list[str] = []
    first_segments = ""
    total_words = 0
    for file_path in paths:
        name = Path(file_path).name
        try:
            text = text_io.extract_text(file_path)
        except ExtractionError as exc:
            lines.append(f"- **{name}** — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- **{name}** — could not read it: {type(exc).__name__}: {exc}")
            continue

        segments = text_io.split_segments(text, max_chars)
        words = len(text.split())
        total_words += words
        lines.append(
            f"- **{name}** — {len(text):,} characters, ~{words:,} words, "
            f"{len(segments):,} segments, ~{words / 150:.0f} min of audio"
        )
        if len(paths) == 1:
            first_segments = "\n".join(segments[:3])

    heading = "**1 book**" if len(paths) == 1 else f"**{len(paths)} books**"
    report = (
        f"{heading} at {int(max_chars)} chars per segment — "
        f"~{total_words:,} words, ~{total_words / 150:.0f} min of audio in total.\n\n"
        + "\n".join(lines)
    )
    if first_segments:
        report += f"\n\nFirst segments:\n\n> {first_segments[:600].replace(chr(10), ' ')}"
    return report


def _render_book(file_path, voice, language, max_chars, num_step, speed, pause, fmt,
                 limit, tick=None):
    """Write one book's audio to disk and report how it went.

    Returns ``(path, summary, ok)``. Failures come back as a summary string
    rather than an exception, so one bad book cannot stop a whole batch.
    """
    try:
        text = text_io.extract_text(file_path)
    except ExtractionError as exc:
        return None, str(exc), False
    except Exception as exc:  # noqa: BLE001
        log(traceback.format_exc())
        return None, f"could not read it: {type(exc).__name__}: {exc}", False

    segments = text_io.split_segments(text, max_chars)
    if not segments:
        return None, "no speech segments were found in it", False

    limit = int(limit or 0)
    partial = limit > 0 and limit < len(segments)
    if partial:
        segments = segments[:limit]

    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", Path(file_path).stem).strip() or "audiobook"
    if partial:
        stem = f"{stem} (preview)"

    out_path = audio_io.unique_path(OUTPUT_DIR, stem, fmt)
    try:
        # Segments go straight to disk, so a long book never sits in memory.
        sink = audio_io.AudioSink(out_path, MODEL.sampling_rate, fmt, pause)
    except ExportError as exc:
        return None, str(exc), False

    failures: list[int] = []
    try:
        for index, segment in enumerate(segments):
            if tick:
                tick(index, len(segments), f"segment {index + 1}/{len(segments)}")
            for attempt in (1, 2):
                try:
                    sink.write(synthesize(segment, voice, language, num_step, speed))
                    break
                except Exception as exc:  # noqa: BLE001 - retry once per segment
                    log(f"Segment {index + 1} attempt {attempt} failed: {exc}")
                    if attempt == 2:
                        failures.append(index + 1)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log(traceback.format_exc())
        sink.discard()
        return None, f"generation stopped: {type(exc).__name__}: {exc}", False

    if not sink.segments:
        sink.discard()
        return None, "generation failed for every segment — see the terminal log", False

    if tick:
        tick(len(segments), len(segments), "finishing the audio file…")
    try:
        sink.finish()
    except ExportError as exc:
        sink.discard()
        return None, str(exc), False
    except Exception as exc:  # noqa: BLE001
        log(traceback.format_exc())
        sink.discard()
        return None, f"export failed: {type(exc).__name__}: {exc}", False

    summary = (
        f"{sink.segments:,} of {len(segments):,} segments · "
        f"~{sink.seconds / 60:.1f} min of audio · {fmt.upper()}"
    )
    if failures:
        summary += f" · skipped {len(failures)} segment(s): {failures[:20]}"
    return out_path, summary, True


def generate_audiobooks(file_paths, mode, saved, ref_audio, ref_text, instruct, language,
                        max_chars, num_step, speed, pause, fmt, limit,
                        progress=gr.Progress()):
    """Turn every dropped book into its own audio file, one book after another.

    The GPU works on one segment at a time, so the books run in sequence rather
    than in parallel, and a book that fails does not stop the rest of the batch.
    """
    paths = _book_paths(file_paths)
    if not paths:
        return None, None, "Drop one or more book files first."

    try:
        voice = resolve_voice(mode, saved, ref_audio, ref_text, instruct)
    except VoiceError as exc:
        return None, None, f"Error: {exc}"

    log(f"Audiobook batch: {len(paths)} book(s), voice={voice['kind']}, format={fmt}")

    produced: list[Path] = []
    rows: list[tuple[str, str, str]] = []

    for index, file_path in enumerate(paths):
        name = Path(file_path).name
        label = f"Book {index + 1}/{len(paths)}: {name}"

        def tick(done, total, desc, index=index, label=label):
            # One bar across the batch, not one that restarts for every book.
            share = (done / total) if total else 0.0
            progress((index + share) / len(paths), desc=f"{label} — {desc}")

        out_path, summary, ok = _render_book(
            file_path, voice, language, max_chars, num_step, speed, pause, fmt, limit, tick
        )
        log(f"{label}: {summary}")
        if ok:
            produced.append(out_path)
        rows.append((name, out_path.name if out_path else "—", summary))

    files = [str(path) for path in produced]
    first = files[0] if files else None

    if len(paths) == 1:
        name, audio_name, summary = rows[0]
        if produced:
            status = f"### Done — `{audio_name}`\n{summary}\n\nSaved to `{produced[0]}`"
        else:
            status = f"**{name}**: {summary}"
        return first, files, status

    status = f"### Done — {len(produced)} of {len(paths)} books"
    if len(produced) < len(paths):
        status += f" ({len(paths) - len(produced)} failed)"
    status += "\n\n| Book | Audio file | Detail |\n| --- | --- | --- |\n"
    for name, audio_name, summary in rows:
        status += f"| {name} | {audio_name} | {summary} |\n"
    status += f"\nSaved to `{OUTPUT_DIR}`"
    return first, files, status


def generate_audiobook(file_path, mode, saved, ref_audio, ref_text, instruct, language,
                       max_chars, num_step, speed, pause, fmt, limit,
                       progress=gr.Progress()):
    """Single-book entry point, kept for callers that pass one file."""
    audio, files, status = generate_audiobooks(
        file_path, mode, saved, ref_audio, ref_text, instruct, language,
        max_chars, num_step, speed, pause, fmt, limit, progress,
    )
    return audio, (files[0] if files else None), status


def open_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(OUTPUT_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])
        return f"Opened `{OUTPUT_DIR}`"
    except Exception as exc:  # noqa: BLE001
        return f"Could not open the folder ({exc}). It is at `{OUTPUT_DIR}`"


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


# Gradio 6 moved theme/css from the Blocks constructor to launch().
THEME = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
CSS = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.05em !important;}
"""


def build_ui() -> gr.Blocks:
    def language_dropdown(label="Language (optional)"):
        return gr.Dropdown(
            label=label,
            choices=SYSTEMS,
            value="Auto",
            allow_custom_value=False,
            info="Keep as Auto to auto-detect the language.",
        )

    def voice_source():
        with gr.Group():
            mode = gr.Radio(
                choices=VOICE_MODES,
                value="Saved voice",
                label="Voice source",
                info="Saved voices are reusable across tabs and sessions.",
            )
            saved = gr.Dropdown(
                label="Saved voice",
                choices=voice_store.voice_names(),
                value=(voice_store.voice_names() or [None])[0],
                interactive=True,
            )
            refresh = gr.Button("Refresh saved voices", size="sm")

            with gr.Group(visible=False) as ref_group:
                gr.Markdown(
                    "Upload or record a 3–10 second reference clip. Leave the "
                    "transcript empty to auto-transcribe it with Whisper."
                )
                ref_audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath",
                    label="Reference audio",
                )
                ref_text = gr.Textbox(
                    label="Reference transcript (optional)",
                    placeholder="Leave empty to let Whisper transcribe it",
                    lines=2,
                )

            with gr.Group(visible=False) as design_group:
                gr.Markdown(DESIGN_INFO)
                instruct = gr.Textbox(
                    label="Voice description", lines=2,
                    placeholder="female, low pitch, british accent",
                )
        return mode, saved, refresh, ref_audio, ref_text, instruct, ref_group, design_group

    def wire_voice_source(mode, saved, refresh, ref_group, design_group):
        def show_groups(choice):
            return (
                gr.update(visible=choice == "Reference audio"),
                gr.update(visible=choice == "Voice design"),
                gr.update(visible=choice == "Saved voice"),
            )

        mode.change(
            show_groups, inputs=mode, outputs=[ref_group, design_group, saved]
        )
        refresh.click(refresh_voices, outputs=saved)

    with gr.Blocks(title="OmniVoice Studio") as demo:
        gr.Markdown(
            """
# OmniVoice Studio

Zero-shot multilingual text-to-speech in 600+ languages. Clone a voice from a
clip, design one from a description, or turn a whole book into an audiobook.
"""
        )

        with gr.Tabs():
            # ---------------------------------------------------------- clone
            with gr.Tab("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_text = gr.Textbox(
                            label="Text to synthesize", lines=6,
                            placeholder="Type or paste the text you want spoken…",
                        )
                        clone_language = language_dropdown()
                        with gr.Accordion("Generation settings", open=False):
                            clone_steps = gr.Slider(
                                8, 64, value=32, step=1, label="Inference steps",
                                info="Lower = faster, higher = better quality.",
                            )
                            clone_speed = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="Speed",
                            )
                        clone_button = gr.Button("Generate speech", variant="primary")
                    with gr.Column(scale=2):
                        (
                            clone_mode, clone_saved, clone_refresh, clone_ref_audio,
                            clone_ref_text, clone_ref_instruct,
                            clone_ref_group, clone_design_group,
                        ) = voice_source()
                        wire_voice_source(
                            clone_mode, clone_saved, clone_refresh,
                            clone_ref_group, clone_design_group,
                        )
                        with gr.Accordion("Save this reference as a voice", open=False):
                            clone_voice_name = gr.Textbox(label="Voice name", lines=1)
                            clone_save = gr.Button("Save voice", size="sm")

                with gr.Row():
                    clone_audio = gr.Audio(label="Result", type="filepath")
                    clone_file = gr.File(label="Download")
                clone_status = gr.Markdown()

                clone_saved.change(
                    load_saved_voice_clip, inputs=clone_saved,
                    outputs=[clone_ref_audio, clone_ref_text],
                )
                clone_save.click(
                    save_voice_from_clip,
                    inputs=[clone_voice_name, clone_ref_audio, clone_ref_text, clone_language],
                    outputs=[clone_status, clone_saved],
                )
                clone_button.click(
                    generate_single,
                    inputs=[
                        clone_text, clone_mode, clone_saved, clone_ref_audio,
                        clone_ref_text, clone_ref_instruct, clone_language,
                        clone_steps, clone_speed, gr.State("wav"),
                    ],
                    outputs=[clone_audio, clone_file, clone_status],
                )

            # --------------------------------------------------------- design
            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=3):
                        design_text = gr.Textbox(
                            label="Text to synthesize", lines=6,
                            placeholder="Type or paste the text you want spoken…",
                        )
                        design_language = language_dropdown()
                        design_instruct = gr.Textbox(
                            label="Voice description", lines=2, value="female, low pitch, british accent",
                            info=DESIGN_INFO,
                        )
                        with gr.Accordion("Generation settings", open=False):
                            design_steps = gr.Slider(
                                8, 64, value=32, step=1, label="Inference steps"
                            )
                            design_speed = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="Speed",
                            )
                        design_button = gr.Button("Generate speech", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Accordion("Save this design as a voice", open=False):
                            design_name = gr.Textbox(label="Voice name", lines=1)
                            design_save = gr.Button("Save voice design", size="sm")
                        design_voice_status = gr.Markdown()
                        gr.Markdown(
                            "Voice design is trained on Chinese and English; cloning "
                            "a reference clip gives the most consistent results."
                        )

                with gr.Row():
                    design_audio = gr.Audio(label="Result", type="filepath")
                    design_file = gr.File(label="Download")
                design_status = gr.Markdown()

                design_save.click(
                    save_voice_from_design,
                    inputs=[design_name, design_instruct, design_language],
                    outputs=[design_voice_status, clone_saved],
                )
                design_button.click(
                    lambda text, instruct, language, steps, speed: generate_single(
                        text, "Voice design", None, None, None, instruct,
                        language, steps, speed, "wav",
                    ),
                    inputs=[design_text, design_instruct, design_language, design_steps, design_speed],
                    outputs=[design_audio, design_file, design_status],
                )

            # ------------------------------------------------------ audiobook
            with gr.Tab("Audiobook"):
                gr.Markdown(
                    "Drop one or more book files and pick a voice. Supported: "
                    "**TXT, MD, PDF, EPUB, DOC, DOCX, HTML**. Each book becomes its "
                    "own file, named after the book, in the `output` folder. Books in "
                    "a batch are read one after another."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        book_file = gr.File(
                            label="Book files (drop one or several here)",
                            file_count="multiple",
                            file_types=sorted(text_io.SUPPORTED_EXTS),
                            type="filepath",
                        )
                        book_preview = gr.Button("Check files", size="sm")
                        book_preview_md = gr.Markdown()

                        with gr.Accordion("Audiobook settings", open=True):
                            book_language = language_dropdown()
                            book_chars = gr.Slider(
                                120, 600, value=300, step=20,
                                label="Segment length (characters)",
                                info="Shorter segments are safer for long books; "
                                     "longer ones sound more natural.",
                            )
                            book_pause = gr.Slider(
                                0.0, 1.5, value=0.3, step=0.05,
                                label="Pause between segments (seconds)",
                            )
                            book_format = gr.Radio(
                                list(audio_io.FORMATS), value="mp3", label="Output format",
                            )
                            book_steps = gr.Slider(
                                8, 64, value=32, step=1, label="Inference steps",
                            )
                            book_speed = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="Speed",
                            )
                            book_limit = gr.Number(
                                value=0, precision=0, label="Limit segments (0 = full book)",
                                info="Use a small number to test the voice first.",
                            )
                        book_button = gr.Button("Create audiobooks", variant="primary")
                        open_folder = gr.Button("Open output folder", size="sm")

                    with gr.Column(scale=2):
                        (
                            book_mode, book_saved, book_refresh, book_ref_audio,
                            book_ref_text, book_ref_instruct,
                            book_ref_group, book_design_group,
                        ) = voice_source()
                        wire_voice_source(
                            book_mode, book_saved, book_refresh,
                            book_ref_group, book_design_group,
                        )
                        delete_target = gr.Dropdown(
                            label="Delete a saved voice",
                            choices=voice_store.voice_names(),
                            interactive=True,
                        )
                        delete_button = gr.Button("Delete voice", size="sm", variant="stop")
                        delete_status = gr.Markdown()

                book_audio = gr.Audio(
                    label="Audiobook (first file)", type="filepath",
                    interactive=False,
                )
                book_download = gr.File(
                    label="Download audiobooks", file_count="multiple",
                    interactive=False,
                )
                book_status = gr.Markdown()
                folder_status = gr.Markdown()

                book_saved.change(
                    load_saved_voice_clip, inputs=book_saved,
                    outputs=[book_ref_audio, book_ref_text],
                )
                book_preview.click(
                    preview_book, inputs=[book_file, book_chars], outputs=book_preview_md
                )
                book_button.click(
                    generate_audiobooks,
                    inputs=[
                        book_file, book_mode, book_saved, book_ref_audio, book_ref_text,
                        book_ref_instruct, book_language, book_chars, book_steps,
                        book_speed, book_pause, book_format, book_limit,
                    ],
                    outputs=[book_audio, book_download, book_status],
                )
                delete_button.click(
                    delete_saved_voice, inputs=delete_target,
                    outputs=[delete_status, book_saved],
                )
                delete_button.click(
                    refresh_voices, outputs=delete_target
                )
                open_folder.click(open_output_folder, outputs=folder_status)

        gr.Markdown(
            f"Outputs: `{OUTPUT_DIR}` · Voices: `{voice_store.VOICES_DIR}` · "
            "Powered by k2-fsa/OmniVoice"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVoice Studio (clone, design, audiobook).")
    parser.add_argument("--ip", default="127.0.0.1", help="Server IP (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    parser.add_argument("--model", default="k2-fsa/OmniVoice", help="Model checkpoint or local path.")
    parser.add_argument("--device", default=None, help="cuda, mps, xpu or cpu. Auto-detected by default.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--root-path", default=None, help="Gradio root_path for reverse proxies.")
    parser.add_argument("--no-asr", action="store_true", help="Skip loading Whisper ASR.")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    return parser


def main(argv=None) -> int:
    global MODEL

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    device = args.device or get_best_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    log(f"Loading model from {args.model}, device={device} …")
    MODEL = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voice_store.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print("Model loaded.")

    demo = build_ui()
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        theme=THEME,
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
