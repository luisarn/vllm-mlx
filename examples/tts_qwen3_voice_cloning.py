#!/usr/bin/env python3
"""
Qwen3-TTS Voice Cloning Example

This example demonstrates voice cloning using Qwen3-TTS model.
The model clones a voice from a reference audio file and synthesizes
speech with that voice.

Requirements:
    pip install mlx-audio

Usage:
    python tts_qwen3_voice_cloning.py "Hello, this is my cloned voice!" --ref-audio my_voice.wav --play

    python tts_qwen3_voice_cloning.py "Hello world!" --ref-audio my_voice.wav --output output.wav
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Cloning Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone voice and play immediately
  python tts_qwen3_voice_cloning.py "Hello world!" --ref-audio my_voice.wav --play

  # Clone voice and save to file
  python tts_qwen3_voice_cloning.py "Hello world!" --ref-audio my_voice.wav --output output.wav

  # Use local model
  python tts_qwen3_voice_cloning.py "Hello!" --ref-audio my_voice.wav \\
      --model ~/.Qwen3-TTS-12Hz-0.6B-Base-bf16 --play
        """,
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Path to reference audio file to clone voice from (WAV format)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        help="Qwen3-TTS model to use (default: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (WAV format)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio immediately after generation",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (0.5 to 2.0, default: 1.0)",
    )

    args = parser.parse_args()

    # Check if ref_audio exists
    ref_path = Path(args.ref_audio)
    if not ref_path.exists():
        print(f"Error: Reference audio file not found: {args.ref_audio}")
        sys.exit(1)

    try:
        from vllm_mlx.audio import TTSEngine
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install vllm-mlx with audio support:")
        print("  pip install mlx-audio")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Text: '{args.text}'")
    print()

    try:
        # Initialize and load the model
        engine = TTSEngine(model_name=args.model)
        engine.load()

        print(f"Model family: {engine._model_family}")
        print("Generating speech...")

        # Generate speech with voice cloning
        audio = engine.generate(
            text=args.text,
            ref_audio=str(ref_path),
            speed=args.speed,
        )

        print(f"Generated audio: {audio.duration:.2f}s at {audio.sample_rate}Hz")

        # Save to file if requested
        if args.output:
            engine.save(audio, args.output)
            print(f"Saved to: {args.output}")

        # Play audio if requested
        if args.play:
            print("Playing audio...")
            try:
                import sounddevice as sd

                sd.play(audio.audio, audio.sample_rate)
                sd.wait()
            except ImportError:
                print("Warning: sounddevice not installed. Cannot play audio.")
                print("Install with: pip install sounddevice")

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
