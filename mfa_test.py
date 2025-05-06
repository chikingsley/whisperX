"""Test script for WhisperX transcription and MFA alignment."""
import subprocess
import os
import tempfile
import whisperx
# --------------------------------------------------------------------------- #
# Montreal Forced Aligner helper                                              #
# --------------------------------------------------------------------------- #
# !!! UPDATE THE DICTIONARY AND ACOUSTIC MODEL PATHS TO MATCH YOUR LOCAL FILES
DICT_PATHS = {
    "en": "/Users/simonpeacocks/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict",
    "fr": "/Users/simonpeacocks/Documents/MFA/pretrained_models/dictionary/french_mfa.dict"
}

ACOUSTIC_MODELS = {
    "en": "/Users/simonpeacocks/Documents/MFA/pretrained_models/acoustic/english_mfa.zip",
    "fr": "/Users/simonpeacocks/Documents/MFA/pretrained_models/acoustic/french_mfa.zip"
}


def run_mfa_alignment(audio_path: str, transcript: str, lang: str) -> str | None:
    """
    Align `audio_path` + `transcript` with MFA and return the resulting
    TextGrid path, or None if the language is not configured.

    MFA expects mono 16 kHz WAV with a matching .lab transcript file.
    """
    if lang not in DICT_PATHS:
        print(f"[MFA] No resources configured for language '{lang}', skipping.")
        return None

    dictionary_path = DICT_PATHS[lang]
    acoustic_model_path = ACOUSTIC_MODELS[lang]

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        lab_path = os.path.join(td, "audio.lab")
        # Convert to 16 kHz mono WAV required by MFA
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        out_dir = os.path.join(td, "aligned")
        subprocess.run(
            ["mamba",
             "run",
             "-n", "aligner",
             "--no-capture-output", # Prevent conda/mamba messing with stdio
             "mfa",
             "align", 
             td,
             dictionary_path,
             acoustic_model_path,
             out_dir,
             "--clean",
             "--overwrite"],
            check=True,
        )

        textgrid_path = os.path.join(out_dir, "audio.TextGrid")
        # Copy TextGrid next to original audio for convenience
        final_path = os.path.splitext(audio_path)[0] + ".TextGrid"
        subprocess.run(["cp", textgrid_path, final_path], check=True)
        return final_path
# --------------------------------------------------------------------------- #

device = "cpu" # pylint: disable=invalid-name
audio_file = "/Volumes/simons-enjoyment/GitHub/whisperX/French 1 - wav/French I - Lesson 01.wav" # pylint: disable=invalid-name
batch_size = 16 # reduce if low on GPU mem; pylint: disable=invalid-name
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy); pylint: disable=invalid-name

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model(
#    "large-v2", device, compute_type=compute_type, download_root=model_dir
# )

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# --------------------------------------------------------------------------- #
# 2. Align with Montreal Forced Aligner (optional)
transcript_text = " ".join(seg["text"] for seg in result["segments"]) # pylint: disable=invalid-name
mfa_grid = run_mfa_alignment(audio_file, transcript_text, result["language"])
if mfa_grid:
    print(f"[MFA] Alignment saved to: {mfa_grid}")
# --------------------------------------------------------------------------- #

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token="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", device=device
)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"])
# End-of-file (EOF)
