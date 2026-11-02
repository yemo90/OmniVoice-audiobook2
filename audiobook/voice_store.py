"""A tiny on-disk library for saved voices.

Each voice lives in ``voices/<slug>/``:

    meta.json     name, kind ("clone" or "design"), language, ref_text, instruct
    reference.*   the reference clip (clone voices only)
    prompt.pt     the encoded VoiceClonePrompt (clone voices only)
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


class VoiceError(RuntimeError):
    pass


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "voice"


def _read_meta(folder: Path) -> dict | None:
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta.setdefault("folder", str(folder))
    return meta


def list_voices() -> list[dict]:
    """All saved voices, newest first."""
    if not VOICES_DIR.exists():
        return []
    voices = []
    for folder in sorted(VOICES_DIR.iterdir()):
        if folder.is_dir():
            meta = _read_meta(folder)
            if meta:
                voices.append(meta)
    voices.sort(key=lambda m: m.get("created", ""), reverse=True)
    return voices


def voice_names() -> list[str]:
    return [voice["name"] for voice in list_voices()]


def get_voice(name: str) -> dict | None:
    for voice in list_voices():
        if voice["name"] == name:
            return voice
    return None


def _folder_for(name: str) -> Path:
    return VOICES_DIR / _slug(name)


def _unique_name(name: str) -> str:
    """If the slug is taken by a differently-named voice, keep both."""
    folder = _folder_for(name)
    existing = _read_meta(folder) if folder.exists() else None
    if existing and existing.get("name") != name:
        raise VoiceError(
            f"A different voice already uses the name '{existing.get('name')}' "
            f"(same folder). Pick another name."
        )
    return name


def save_clone_voice(
    name: str,
    ref_audio: str,
    ref_text: str | None,
    language: str | None,
    prompt=None,
) -> dict:
    name = name.strip()
    if not name:
        raise VoiceError("Give the voice a name first.")
    if not ref_audio:
        raise VoiceError("Record or upload a reference clip before saving.")

    _unique_name(name)
    folder = _folder_for(name)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    source = Path(ref_audio)
    suffix = source.suffix or ".wav"
    reference = folder / f"reference{suffix}"
    shutil.copyfile(source, reference)

    prompt_file = None
    if prompt is not None:
        prompt_path = folder / "prompt.pt"
        prompt.save(str(prompt_path))
        prompt_file = prompt_path.name

    meta = {
        "name": name,
        "kind": "clone",
        "language": language,
        "ref_text": (ref_text or "").strip(),
        "instruct": "",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ref_file": reference.name,
        "prompt_file": prompt_file,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["folder"] = str(folder)
    return meta


def save_design_voice(name: str, instruct: str, language: str | None) -> dict:
    name = name.strip()
    instruct = (instruct or "").strip()
    if not name:
        raise VoiceError("Give the voice a name first.")
    if not instruct:
        raise VoiceError("Describe the voice (e.g. 'female, low pitch, british accent').")

    _unique_name(name)
    folder = _folder_for(name)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": name,
        "kind": "design",
        "language": language,
        "ref_text": "",
        "instruct": instruct,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ref_file": None,
        "prompt_file": None,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["folder"] = str(folder)
    return meta


def reference_path(voice: dict) -> Path | None:
    if not voice.get("ref_file"):
        return None
    path = Path(voice["folder"]) / voice["ref_file"]
    return path if path.exists() else None


def load_prompt(voice: dict, model=None):
    """Return the saved VoiceClonePrompt, or encode one from the reference clip."""
    prompt_file = voice.get("prompt_file")
    if prompt_file:
        path = Path(voice["folder"]) / prompt_file
        if path.exists():
            try:
                from omnivoice import VoiceClonePrompt

                return VoiceClonePrompt.load(str(path))
            except Exception:
                pass  # fall through and re-encode

    reference = reference_path(voice)
    if reference is None or model is None:
        return None
    return model.create_voice_clone_prompt(
        ref_audio=str(reference), ref_text=voice.get("ref_text") or None
    )


def delete_voice(name: str) -> None:
    voice = get_voice(name)
    if not voice:
        raise VoiceError(f"No saved voice called '{name}'.")
    shutil.rmtree(voice["folder"], ignore_errors=True)
