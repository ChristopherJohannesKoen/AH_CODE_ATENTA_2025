# Run.py
import os
from typing import Optional
from df.enhance import enhance, init_df, load_audio, save_audio

def denoise(input_file: str, output_dir: Optional[str] = None) -> str:
    """
    Denoise an audio file using DeepFilterNet.

    Parameters
    ----------
    input_file : str
        Absolute path to the noisy input audio file (e.g., saved by Flask).
    output_dir : Optional[str]
        Directory to write the enhanced file. If None, defaults to:
        <backend>/Model/Input/Audio

    Returns
    -------
    str
        Absolute path to the enhanced output file.

    Raises
    ------
    FileNotFoundError
        If input_file doesn't exist.
    RuntimeError
        If model load or processing fails.
    """
    input_file = os.path.abspath(input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Resolve <backend> as the parent of this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(base_dir)

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(backend_dir, "Model", "Input", "Audio")

    os.makedirs(output_dir, exist_ok=True)

    # Output name: enhanced_<originalname>.wav
    in_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"enhanced_{in_base}.wav")

    try:
        print("[denoise] Loading DeepFilterNet model...")
        model, df_state, _ = init_df()
        print("[denoise] Model loaded.")

        print(f"[denoise] Loading audio: {input_file}")
        audio, _ = load_audio(input_file, sr=df_state.sr())

        print("[denoise] Enhancing...")
        enhanced = enhance(model, df_state, audio)

        print(f"[denoise] Saving to: {output_file}")
        save_audio(output_file, enhanced, df_state.sr())

        print("[denoise] Done.")
        return os.path.abspath(output_file)

    except Exception as e:
        # Let caller decide how to handle; include context
        raise RuntimeError(f"denoise failed for '{input_file}': {e}") from e


if __name__ == "__main__":
    # Optional: manual quick test (edit path as needed)
    # Example: python Run.py "/absolute/path/to/noisy_recording.wav"
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Run.py /absolute/path/to/noisy_recording.wav [optional_output_dir]")
        sys.exit(1)
    inp = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    out_path = denoise(inp, out_dir)
    print("Enhanced file at:", out_path)
