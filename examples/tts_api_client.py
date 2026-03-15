#!/usr/bin/env python3
"""
TTS API Client Example

Demonstrates using the vllm-mlx server's /v1/audio/speech endpoint
with voice cloning via Qwen3-TTS.

Usage:
    # Start the server first:
    vllm-mlx serve <model> --port 8000

    # Standard TTS (Kokoro):
    python tts_api_client.py "Hello world!" --output output.wav

    # Voice cloning with Qwen3-TTS:
    python tts_api_client.py "Hello world!" \
        --model qwen3-tts \
        --ref-audio my_voice.wav \
        --output cloned.wav

    # With API key:
    python tts_api_client.py "Hello!" --api-key your-key --output output.wav
"""

import argparse
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(
        description="TTS API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard TTS
  python tts_api_client.py "Hello world!" --output output.wav

  # Voice cloning
  python tts_api_client.py "Hello world!" --model qwen3-tts \\
      --ref-audio my_voice.wav --output cloned.wav

  # List available voices
  python tts_api_client.py --list-voices
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument(
        "--server", default="http://localhost:8000", help="Server URL"
    )
    parser.add_argument(
        "--model", default="kokoro", help="TTS model (kokoro, qwen3-tts, etc.)"
    )
    parser.add_argument(
        "--voice", default="af_heart", help="Voice ID (for non-cloning models)"
    )
    parser.add_argument(
        "--ref-audio", help="Reference audio file for voice cloning"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output file path (WAV)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (0.5 to 2.0)"
    )
    parser.add_argument(
        "--api-key", help="API key for authentication"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices for the model",
    )

    args = parser.parse_args()

    # List voices mode
    if args.list_voices:
        url = f"{args.server}/v1/audio/voices"
        params = {"model": args.model}
        headers = {}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            print(f"Available voices for {args.model}:")
            for voice in data.get("voices", []):
                print(f"  - {voice}")
            if data.get("requires_ref_audio"):
                print("\nNote: This model requires a reference audio file for voice cloning.")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Validate text is provided
    if not args.text:
        print("Error: text is required (unless using --list-voices)")
        parser.print_help()
        sys.exit(1)

    # Build request
    url = f"{args.server}/v1/audio/speech"
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    data = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "speed": args.speed,
        "response_format": "wav",
    }

    files = {}
    ref_path = None

    # Handle reference audio for voice cloning
    if args.ref_audio:
        ref_path = Path(args.ref_audio)
        if not ref_path.exists():
            print(f"Error: Reference audio file not found: {args.ref_audio}")
            sys.exit(1)
        files["ref_audio"] = open(ref_path, "rb")

    try:
        print(f"Server: {args.server}")
        print(f"Model: {args.model}")
        print(f"Text: '{args.text}'")
        if args.ref_audio:
            print(f"Reference audio: {args.ref_audio}")
        print("Generating speech...")

        # Make request
        if files:
            # Multipart form for file upload
            response = requests.post(
                url, data=data, files=files, headers=headers, timeout=120
            )
        else:
            # Regular form data
            response = requests.post(
                url, data=data, headers=headers, timeout=60
            )

        response.raise_for_status()

        # Save audio
        output_path = Path(args.output)
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Saved to: {args.output}")
        print(f"Size: {len(response.content)} bytes")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")
        sys.exit(1)
    finally:
        # Close file handles
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
