# tests/download_fixtures.py
"""
Downloads short audio clips from FLEURS (Apache 2.0 / CC-BY-4.0) for E2E tests.
Run once: python tests/download_fixtures.py
Clips are cached in tests/fixtures/audio/.
"""
import io
import os
import soundfile as sf
import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "audio")

# FLEURS config names and expected ground-truth text fragments
CLIPS = [
    {
        "lang": "en",
        "fleurs_config": "en_us",
        "filename": "en_sample.wav",
        "expected_words": ["the", "a"],
    },
    {
        "lang": "zh",
        "fleurs_config": "cmn_hans_cn",
        "filename": "zh_sample.wav",
        "expected_words": ["的", "了"],
    },
    {
        "lang": "fr",
        "fleurs_config": "fr_fr",
        "filename": "fr_sample.wav",
        "expected_words": ["le", "la", "de"],
    },
    {
        "lang": "vi",
        "fleurs_config": "vi_vn",
        "filename": "vi_sample.wav",
        "expected_words": ["và", "của", "là"],
    },
]


def download_all():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    for clip in CLIPS:
        out_path = os.path.join(FIXTURES_DIR, clip["filename"])
        if os.path.exists(out_path):
            print(f"  already exists: {clip['filename']}")
            continue
        print(f"  downloading {clip['lang']} ({clip['fleurs_config']})...")
        ds = load_dataset(
            "google/fleurs",
            clip["fleurs_config"],
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        sample = next(iter(ds))
        arr = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        # trim to 8 seconds max
        arr = arr[: sr * 8]
        sf.write(out_path, arr, sr, format="WAV")
        print(f"  saved {clip['filename']} ({len(arr)/sr:.1f}s)")


def get_fixture_path(lang: str) -> str:
    for clip in CLIPS:
        if clip["lang"] == lang:
            path = os.path.join(FIXTURES_DIR, clip["filename"])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fixture for '{lang}' not found. Run: python tests/download_fixtures.py"
                )
            return path
    raise ValueError(f"No fixture configured for lang='{lang}'")


def get_expected_words(lang: str) -> list[str]:
    for clip in CLIPS:
        if clip["lang"] == lang:
            return clip["expected_words"]
    return []


if __name__ == "__main__":
    print("Downloading FLEURS audio fixtures...")
    download_all()
    print("Done.")
