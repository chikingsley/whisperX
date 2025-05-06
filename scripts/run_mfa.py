#!/usr/bin/env python
"""
MFA Alignment Script - Aligns audio file with transcript using Montreal Forced Aligner.

This script takes an audio file and transcript, then runs Montreal Forced Aligner
to generate phoneme-level alignments.

Requirements:
- Montreal Forced Aligner installed in a conda/mamba environment named 'aligner'
- Pre-downloaded acoustic models and dictionaries
"""

import argparse
import os
import subprocess
import tempfile
import shutil
import sys

# Default paths - update these to match your local setup
DEFAULT_DICT_PATHS = {
    "en": "/Users/simonpeacocks/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict",
    "fr": "/Users/simonpeacocks/Documents/MFA/pretrained_models/dictionary/french_mfa.dict"
}

DEFAULT_ACOUSTIC_MODELS = {
    "en": "/Users/simonpeacocks/Documents/MFA/pretrained_models/acoustic/english_mfa.zip",
    "fr": "/Users/simonpeacocks/Documents/MFA/pretrained_models/acoustic/french_mfa.zip"
}

def run_mfa(audio_path, transcript_text, lang="en", output_dir=None,
            dict_path=None, acoustic_model=None):
    """
    Run MFA alignment on an audio file with provided transcript.
    
    Args:
        audio_path: Path to the audio file (WAV format recommended)
        transcript_text: Text transcript of the audio
        lang: Language code ("en" or "fr")
        output_dir: Where to save the output TextGrid (defaults to same dir as audio)
        dict_path: Path to pronunciation dictionary (optional)
        acoustic_model: Path to acoustic model (optional)
    
    Returns:
        Path to the generated TextGrid file
    """
    # Use default paths if not provided
    if dict_path is None:
        dict_path = DEFAULT_DICT_PATHS.get(lang)
        if not dict_path or not os.path.exists(dict_path):
            sys.exit(f"Error: Dictionary for language '{lang}' not found at {dict_path}")
    if acoustic_model is None:
        acoustic_model = DEFAULT_ACOUSTIC_MODELS.get(lang)
        if not acoustic_model or not os.path.exists(acoustic_model):
            sys.exit(f"Error: Acoustic model for language '{lang}' not found at {acoustic_model}")
    # Set default output directory to same as audio file
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(audio_path))
    # Create temporary directory for MFA files
    with tempfile.TemporaryDirectory() as td:
        # Create subdirectories
        corpus_dir = os.path.join(td, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        out_dir = os.path.join(td, "aligned")
        os.makedirs(out_dir, exist_ok=True)
        # Copy audio file to corpus dir with simple name
        audio_ext = os.path.splitext(audio_path)[1]
        temp_audio_path = os.path.join(corpus_dir, f"audio{audio_ext}")
        shutil.copy2(audio_path, temp_audio_path)
        # Create transcript file
        with open(os.path.join(corpus_dir, "audio.lab"), "w", encoding="utf-8") as f:
            f.write(transcript_text)
        # Run MFA alignment
        print(f"Running MFA alignment for {os.path.basename(audio_path)}...")
        try:
            _result = subprocess.run(
                ["mamba",
                 "run", 
                 "-n", "aligner", 
                 "mfa", 
                 "align", 
                 corpus_dir,
                 dict_path,
                 acoustic_model,
                 out_dir,
                 "-j", "1",
                 "--clean",
                ],
                check=True,
                capture_output=False,
            )
            # Copy TextGrid to output directory
            textgrid_path = os.path.join(out_dir, "audio.TextGrid")
            if not os.path.exists(textgrid_path):
                sys.exit("Error: MFA did not generate a TextGrid file")
            # Generate output path with same name as audio but .TextGrid extension
            final_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(audio_path))[0] + ".TextGrid"
            )
            # Copy the TextGrid to the final location
            shutil.copy2(textgrid_path, final_path)
            print(f"Alignment successful! TextGrid saved to: {final_path}")
            return final_path
        except subprocess.CalledProcessError as e:
            sys.exit(f"Error running MFA: {e}")
        except Exception as e: # pylint: disable=broad-exception-caught
            sys.exit(f"Unexpected error: {e}")

def main():
    """Command-line interface for running MFA alignment on audio files."""
    parser = argparse.ArgumentParser(description="Run MFA alignment on audio with transcript")
    parser.add_argument("audio_path", help="Path to audio file (WAV format preferred)")
    parser.add_argument("--transcript", help="Transcript text (or path to text file)")
    parser.add_argument("--transcript_file", help="Path to transcript text file")
    parser.add_argument("--lang", default="en", choices=["en", "fr"],
                        help="Language of audio (en or fr)")
    parser.add_argument("--output_dir", help="Directory to save TextGrid output")
    parser.add_argument("--dict_path", help="Path to pronunciation dictionary")
    parser.add_argument("--acoustic_model", help="Path to acoustic model")
    args = parser.parse_args()
    # Validate audio file
    if not os.path.exists(args.audio_path):
        sys.exit(f"Error: Audio file not found: {args.audio_path}")
    # Get transcript text
    transcript_text = None
    # First check if transcript was directly provided
    if args.transcript:
        transcript_text = args.transcript
    # Then check if a transcript file was provided
    elif args.transcript_file:
        if not os.path.exists(args.transcript_file):
            sys.exit(f"Error: Transcript file not found: {args.transcript_file}")
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
    # Error if no transcript
    if not transcript_text:
        sys.exit("Error: No transcript provided. Use --transcript or --transcript_file")
    # Run the alignment
    run_mfa(
        args.audio_path,
        transcript_text,
        lang=args.lang,
        output_dir=args.output_dir,
        dict_path=args.dict_path,
        acoustic_model=args.acoustic_model
    )

if __name__ == "__main__":
    main()
 # End-of-file (EOF)
