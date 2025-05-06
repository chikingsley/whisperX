import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

def select_files():
    """Opens a dialog to select multiple MP3 files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select MP3 files to convert",
        filetypes=(("MP3 files", "*.mp3"), ("All files", "*.*"))
    )
    root.destroy()
    return file_paths

def select_output_directory():
    """Opens a dialog to select the output directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory_path = filedialog.askdirectory(
        title="Select Output Directory for WAV files"
    )
    root.destroy()
    return directory_path

def convert_files(input_files, output_dir):
    """Converts selected MP3 files to WAV using ffmpeg."""
    if not input_files:
        messagebox.showwarning("No Files", "No input files selected.")
        return

    if not output_dir:
        messagebox.showwarning("No Directory", "No output directory selected.")
        return

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    total_files = len(input_files)
    converted_count = 0
    errors = []

    print(f"Starting conversion of {total_files} files...")

    for i, input_file in enumerate(input_files):
        base_name = os.path.basename(input_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.wav")

        print(f"\n[{i+1}/{total_files}] Converting '{base_name}' to '{os.path.basename(output_file)}'...")

        # Check if ffmpeg exists (simple check)
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            messagebox.showerror("Error", "ffmpeg not found or not executable. Please ensure ffmpeg is installed and in your system's PATH.")
            return # Stop processing if ffmpeg isn't found on the first file


        # Construct the ffmpeg command
        # Using -acodec pcm_s16le is more explicit than -sample_fmt s16 for WAV
        command = [
            "ffmpeg",
            "-y",          # Overwrite output file if it exists
            "-i", input_file,
            "-ar", "16000", # Sample rate 16 kHz
            "-ac", "1",     # Mono audio
            "-acodec", "pcm_s16le", # Signed 16-bit PCM for WAV
            output_file
        ]

        try:
            process = subprocess.run(
                command,
                check=True,          # Raise an error if ffmpeg fails
                capture_output=True, # Capture stdout/stderr
                text=True,           # Decode output as text
                encoding=sys.getdefaultencoding() # Use system encoding
            )
            print(f"Successfully converted '{base_name}'.")
            # print("ffmpeg output:", process.stderr) # Print ffmpeg's usual progress output
            converted_count += 1
        except subprocess.CalledProcessError as e:
            error_message = f"Failed to convert '{base_name}'. Error:\n{e.stderr}"
            print(error_message, file=sys.stderr)
            errors.append(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred converting '{base_name}':\n{e}"
            print(error_message, file=sys.stderr)
            errors.append(error_message)


    # --- Summary ---
    summary_message = f"\n--- Conversion Summary ---\n"
    summary_message += f"Total files selected: {total_files}\n"
    summary_message += f"Successfully converted: {converted_count}\n"
    summary_message += f"Failed: {len(errors)}\n"

    if errors:
        summary_message += "\nErrors encountered:\n" + "\n".join([f"- {err.splitlines()[0]}" for err in errors]) # Show first line of each error

    print(summary_message)
    # Display summary message box
    if errors:
        messagebox.showerror("Conversion Complete with Errors", summary_message)
    else:
        messagebox.showinfo("Conversion Complete", summary_message)


if __name__ == "__main__":
    # Ensure Tkinter can initialize properly
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
    except tk.TclError as e:
        print("Could not initialize Tkinter GUI. Ensure you have a display environment.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    selected_files = select_files()
    if selected_files: # Proceed only if files were selected
        output_directory = select_output_directory()
        if output_directory: # Proceed only if output directory was selected
            convert_files(selected_files, output_directory)
        else:
            print("Output directory selection cancelled.")
    else:
        print("File selection cancelled.")