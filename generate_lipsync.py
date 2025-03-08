import os
import subprocess
import pandas as pd
import whisper
import tempfile
import argparse
from PIL import Image
import torch
import shutil

# Set up folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MOUTH_IMAGES_DIR = os.path.join(ASSETS_DIR, "mouth_images")
AUDIO_FILE = os.path.join(ASSETS_DIR, "audio.wav")
TRANSCRIPT_FILE = os.path.join(ASSETS_DIR, "transcript.txt")
TEMP_IMAGES_DIR = os.path.join(BASE_DIR, "temp_images")

# CLI setup
parser = argparse.ArgumentParser(
    description="Generation of a video with lipsync and Whisper.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Usage examples:
  python3 generate_concat.py                         # Generate MP4 with white background
  python3 generate_concat.py --background yellow     # Generate MP4 with yellow background
  python3 generate_concat.py --format mov            # Generate MOV with transparency
  python3 generate_concat.py --model small --cpu     # Use small Whisper model on CPU
  python3 generate_concat.py --help                  # Show this help message

Available colors:
  white, black, red, green, blue, yellow, cyan, magenta, purple,
  orange, pink, brown, gold, silver, navy, teal, olive, maroon,
  gray/grey, lime

You can also specify hexadecimal colors:
  python3 generate_concat.py --background 00FF00     # Green (hexadecimal)
  python3 generate_concat.py --background #FF0000    # Red (hexadecimal with #)
"""
)
parser.add_argument(
    "--format",
    choices=["mp4", "mov"],
    default="mp4",
    help="Video format (mp4 for colored background, mov for transparency)",
)
parser.add_argument(
    "--background",
    type=str,
    default="white",
    help="Background color for MP4 (e.g. 'green', 'blue', '00FF00'). Ignored if --format=mov",
)
parser.add_argument(
    "--model",
    choices=["tiny", "base", "small", "medium", "large"],
    default="large",
    help="Whisper model to use",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force Whisper execution on CPU",
)
parser.add_argument(
    "--keep-tmp",
    action="store_true",
    help="Keep temporary files sync.tsv and concat.txt",
)
args = parser.parse_args()

# Check files and folders
if not os.path.exists(ASSETS_DIR):
    print("❌ Error: The 'assets' folder does not exist. Create it and insert the required files.")
    exit(1)

if not os.path.exists(MOUTH_IMAGES_DIR):
    print("❌ Error: The 'mouth_images' folder does not exist inside 'assets'.")
    exit(1)

if not os.path.exists(AUDIO_FILE):
    print("❌ Error: The 'audio.wav' file is not present in 'assets'.")
    exit(1)

# Find Rhubarb executable
RHUBARB_EXEC = os.path.join(BASE_DIR, "rhubarb")
if not os.path.exists(RHUBARB_EXEC):
    print("❌ Error: The Rhubarb executable file ('rhubarb') is not in the main folder.")
    exit(1)

# Check and determine image resolutions
print("🔍 Checking mouth images resolution...")
mouth_shapes = ["A", "B", "C", "D", "E", "F", "G", "X"]
found_images = []
resolutions = []

for mouth_shape in mouth_shapes:
    img_path = os.path.join(MOUTH_IMAGES_DIR, f"mouth_{mouth_shape}.png")
    if os.path.exists(img_path):
        found_images.append(img_path)
        img = Image.open(img_path)
        resolutions.append((img.width, img.height))

if not found_images:
    print("❌ Error: No mouth images found in the mouth_images folder.")
    exit(1)

# Check if all images have the same resolution
if len(set(resolutions)) > 1:
    print("❌ Error: The mouth images have different resolutions. All images must have the same resolution.")
    print("Found resolutions:")
    for img_path, resolution in zip(found_images, resolutions):
        print(f"  - {os.path.basename(img_path)}: {resolution[0]}x{resolution[1]}")
    exit(1)

# Set final resolution based on the mouth images
FINAL_WIDTH, FINAL_HEIGHT = resolutions[0]
print(f"✅ All mouth images have the same resolution: {FINAL_WIDTH}x{FINAL_HEIGHT}")

# Create temporary files
sync_file = tempfile.NamedTemporaryFile(
    delete=not args.keep_tmp, suffix=".tsv", dir=BASE_DIR
)
concat_file = tempfile.NamedTemporaryFile(
    delete=not args.keep_tmp, suffix=".txt", dir=BASE_DIR
)
OUTPUT_VIDEO = os.path.join(BASE_DIR, f"lipsync.{args.format}")

# ---------------------------------------------------------
# 1) Whisper: Use transcript.txt if present, otherwise generate it
# ---------------------------------------------------------
if os.path.exists(TRANSCRIPT_FILE):
    print("✅ Transcription found, using transcript.txt...")
    transcript_file = TRANSCRIPT_FILE
else:
    print(f"⏳ No transcription found, generating with Whisper {args.model}...")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using {device.upper()} for Whisper...")

    model = whisper.load_model(args.model, device=device)
    transcription_result = model.transcribe(AUDIO_FILE)

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_transcript:
        transcript_file = temp_transcript.name
        temp_transcript.write(transcription_result["text"])
    print(f"✅ Transcription completed! Temporary file: {transcript_file}")

# ---------------------------------------------------------
# 2) Run Rhubarb: generate the .tsv synchronization file
# ---------------------------------------------------------
rhubarb_cmd = [
    RHUBARB_EXEC, "-f", "tsv", "-o", sync_file.name, "-d", transcript_file, AUDIO_FILE
]

print("⏳ Generating synchronization file with Rhubarb...")
subprocess.run(rhubarb_cmd, check=True)
print("✅ Synchronization completed!")

# If the transcription was automatically generated, delete the temporary file
if transcript_file != TRANSCRIPT_FILE:
    os.remove(transcript_file)
    print("🗑️ Temporary transcription file deleted.")

# ---------------------------------------------------------
# 3) Prepare FFmpeg with possible GPU acceleration
# ---------------------------------------------------------
ffmpeg_gpu_args = []
try:
    ffmpeg_check = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True)
    if "cuda" in ffmpeg_check.stdout.lower():
        print("✅ FFmpeg supports CUDA, enabling GPU acceleration.")
        ffmpeg_gpu_args = ["-hwaccel", "cuda"]
    else:
        print("⚠️ FFmpeg does not support CUDA, proceeding without hardware acceleration.")
except FileNotFoundError:
    print("❌ FFmpeg not found, make sure it is installed.")
    exit(1)

# ---------------------------------------------------------
# 4) Read the synchronization .tsv and build the timeline
# ---------------------------------------------------------
print("⏳ Preparing concatenation file for FFmpeg...")
try:
    # Read the Rhubarb table
    sync_data = pd.read_csv(sync_file.name, sep="\t", header=None, names=["timestamp", "viseme"])
    
    # start / end for each interval
    sync_data["start"] = sync_data["timestamp"].shift().fillna(0)
    sync_data["end"] = sync_data["timestamp"]
    
    # Total audio duration via ffprobe
    ffprobe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", AUDIO_FILE
    ]
    duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
    
    # Add last line to cover until the end of audio
    last_row = pd.DataFrame({
        "timestamp": [duration],
        "viseme": [sync_data.iloc[-1]["viseme"]],
        "start": [sync_data.iloc[-1]["timestamp"]],
        "end": [duration]
    })
    sync_data = pd.concat([sync_data, last_row], ignore_index=True)

    # -----------------------------------------------------
    # If the user wants MP4 => compose images on colored background
    # If the user wants MOV => copy PNGs "as is"
    # -----------------------------------------------------
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

    def parse_color_hex(c_str):
        """Converts e.g. '00FF00' or '#00FF00' to (0,255,0)."""
        c_str = c_str.strip()
        # If c_str starts with '#', remove it
        if c_str.startswith('#'):
            c_str = c_str[1:]
        # Now c_str = "00FF00"
        r = int(c_str[0:2], 16)
        g = int(c_str[2:4], 16)
        b = int(c_str[4:6], 16)
        return (r, g, b)

    if args.format == "mp4":
        # Convert the user string to (R,G,B)
        # If the user wrote "green", "blue", "white", etc., map them to hex.
        color_mapping = {
            "green": "#00FF00",
            "lime": "#00FF00",
            "red": "#FF0000",
            "blue": "#0000FF",
            "white": "#FFFFFF",
            "black": "#000000",
            "gray": "#808080",
            "grey": "#808080",
            "yellow": "#FFFF00",
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "purple": "#800080",
            "orange": "#FFA500",
            "pink": "#FFC0CB",
            "brown": "#A52A2A",
            "gold": "#FFD700",
            "silver": "#C0C0C0",
            "navy": "#000080",
            "teal": "#008080",
            "olive": "#808000",
            "maroon": "#800000",
        }
        raw_color = args.background.lower()
        if raw_color in color_mapping:
            chosen_hex = color_mapping[raw_color]
        else:
            # If it's not a known name, treat it as hex (with or without #)
            if raw_color.startswith("#"):
                chosen_hex = raw_color
            else:
                # Add # if missing
                if len(raw_color) == 6:
                    chosen_hex = "#" + raw_color
                else:
                    # fallback to white
                    chosen_hex = "#FFFFFF"
        # Now parse_color_hex
        bg_r, bg_g, bg_b = parse_color_hex(chosen_hex)

        print(f"🎨 Composing mouths on colored background {chosen_hex} (R={bg_r},G={bg_g},B={bg_b})")

        for mouth_shape in ["A", "B", "C", "D", "E", "F", "G", "X"]:
            src_file = os.path.join(MOUTH_IMAGES_DIR, f"mouth_{mouth_shape}.png")
            dst_file = os.path.join(TEMP_IMAGES_DIR, f"mouth_{mouth_shape}.png")

            if os.path.exists(src_file):
                # Load the mouth
                mouth_img = Image.open(src_file).convert("RGBA")
                # Create a colored background with the same dimensions as mouth images
                bg_img = Image.new("RGBA", (FINAL_WIDTH, FINAL_HEIGHT), (bg_r, bg_g, bg_b, 255))
                
                # (Optional) If the mouth is smaller, center it
                offx = (FINAL_WIDTH - mouth_img.width) // 2
                offy = (FINAL_HEIGHT - mouth_img.height) // 2
                
                bg_img.alpha_composite(mouth_img, (offx, offy))
                # Save the result
                bg_img.save(dst_file)
            else:
                print(f"⚠️ Image {src_file} not found.")
    else:
        # MOV => transparency. Copy PNGs as they are
        print("🎥 MOV format requested: maintaining transparency, copying PNGs as they are.")
        for mouth_shape in ["A", "B", "C", "D", "E", "F", "G", "X"]:
            src_file = os.path.join(MOUTH_IMAGES_DIR, f"mouth_{mouth_shape}.png")
            dst_file = os.path.join(TEMP_IMAGES_DIR, f"mouth_{mouth_shape}.png")
            if os.path.exists(src_file):
                Image.open(src_file).save(dst_file)
            else:
                print(f"⚠️ Image {src_file} not found.")

    # -----------------------------------------------------
    # Create the concatenation file (concat.txt)
    # -----------------------------------------------------
    with open(concat_file.name, "w") as f:
        prev_end = 0.0

        for i, row in sync_data.iterrows():
            start_time = float(row["start"])
            end_time = float(row["end"])
            mouth_shape = row["viseme"]

            # Gap
            if start_time > prev_end and i > 0:
                f.write(f"file '{os.path.join(TEMP_IMAGES_DIR, 'mouth_X.png')}'\n")
                f.write(f"duration {start_time - prev_end}\n")

            # Current image
            f.write(f"file '{os.path.join(TEMP_IMAGES_DIR, f'mouth_{mouth_shape}.png')}'\n")

            # Specify duration
            if i < len(sync_data) - 1:
                f.write(f"duration {end_time - start_time}\n")
            else:
                # last image
                f.write(f"duration {end_time - start_time}\n")

            prev_end = end_time

    print("✅ Concatenation file prepared!")
except Exception as e:
    print(f"❌ Error in preparing the concatenation file: {str(e)}")
    exit(1)

# ---------------------------------------------------------
# 5) Invoke FFmpeg to create the final video
# ---------------------------------------------------------
# If MP4 → PNGs already have the colored background. Just concatenate them + audio
# If MOV → PNGs have transparency and correspond to the entire frame. Just concatenate them + audio
#
# The main difference is the codec/pix_fmt: for .mov we use ProRes 4444 to maintain alpha
# For .mp4 we use libx264 (yuv420p)
# ---------------------------------------------------------

if args.format == "mov":
    # MOV with transparency
    ffmpeg_cmd = [
        "ffmpeg",
        *ffmpeg_gpu_args,
        "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file.name,
        "-i", AUDIO_FILE,
        "-vsync", "vfr",
        "-pix_fmt", "yuva444p10le",
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-c:a", "aac",
        "-strict", "experimental",
        "-shortest",
        OUTPUT_VIDEO
    ]
else:
    # MP4 with colored background already in the PNGs
    ffmpeg_cmd = [
        "ffmpeg",
        *ffmpeg_gpu_args,
        "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file.name,
        "-i", AUDIO_FILE,
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-c:a", "aac",
        "-strict", "experimental",
        "-shortest",
        OUTPUT_VIDEO
    ]

print(f"⏳ Generating the {args.format.upper()} video...")
subprocess.run(ffmpeg_cmd, check=True)
print(f"✅ {args.format.upper()} video generated!")
if not args.keep_tmp:
    shutil.rmtree(TEMP_IMAGES_DIR, ignore_errors=True)
    print(f"🗑️ Temporary folder '{TEMP_IMAGES_DIR}' automatically removed.")
print("🎬 Processes completed! Check the generated files. 🚀")