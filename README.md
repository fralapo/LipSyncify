# 🎬 Lipsync Generator: Rhubarb & Whisper AI 🎙️

Quickly generate videos with lip synchronization by combining voice audio and custom mouth images.

- **Rhubarb Lip Sync** automatically calculates the timing of lip movements.
- **Whisper AI** automatically generates audio transcription, if needed.
- **FFmpeg** exports the final result in `.mp4` video with custom background or transparent `.mov`.

---
![Lipsync Generator Logo](logo.jpeg)
---

## 🛠️ Installation

### 1️⃣ Download Rhubarb Lip Sync
Download the latest release from the official repository:
- 👉 [Rhubarb Lip Sync Releases](https://github.com/DanielSWolf/rhubarb-lip-sync/releases)

Select the file compatible with your operating system:
- **Windows:** `Rhubarb-Lip-Sync-<version>-Windows.zip`
- **Linux:** `Rhubarb-Lip-Sync-<version>-Linux.zip`
- **macOS:** `Rhubarb-Lip-Sync-<version>-macOS.zip`

📌 Extract the contents into a dedicated folder.

### 2️⃣ Clone this repository
Open a terminal in the newly extracted folder and run:
```bash
cd Rhubarb-Lip-Sync-<version>-<system>  # Change with the correct name of your folder
git clone https://github.com/fralapo/LipSyncify.git
```

### 3️⃣ Install Python dependencies
Make sure Python is installed (recommended version: 3.10 or higher):
```bash
pip install -r requirements.txt
```

📌 For GPU support with Whisper (optional but recommended):
```bash
pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4️⃣ Install FFmpeg
Check if FFmpeg is already installed:
```bash
ffmpeg -version
```
If needed, download it from [here](https://ffmpeg.org/download.html).

---

## 📂 Project structure
```
📁 Rhubarb-Lip-Sync/
├── 📁 assets/
│   ├── 📁 mouth_images/      # Transparent PNGs for mouths
│   ├── 🎵 audio.wav          # Audio file to synchronize
│   └── 📜 transcript.txt     # (Optional) Audio transcription
├── 📝 generate_lipsync.py    # Main script
├── 📜 requirements.txt       # Python dependencies
└── 🛠 rhubarb                 # Rhubarb Lip Sync executable
```

📌 If the `transcript.txt` file is missing, Whisper will generate it automatically.

---

## 🚀 How to use it

Open the terminal in the project folder and run:

Usage examples:
```bash
python3 generate_lipsync.py                         # Generate MP4 with white background
```
```bash
python3 generate_lipsync.py --background yellow     # Generate MP4 with yellow background
```
```bash
python3 generate_lipsync.py --format mov            # Generate MOV with transparency
```
```bash
python3 generate_lipsync.py --model small --cpu     # Use small Whisper model on CPU
```
```bash
python3 generate_lipsync.py --help                  # Show this help message
```

Available colors:
```
white, black, red, green, blue, yellow, cyan, magenta, purple,
orange, pink, brown, gold, silver, navy, teal, olive, maroon,
gray/grey, lime
```

You can also specify hexadecimal colors:
```bash
python3 generate_lipsync.py --background 00FF00     # Green (hexadecimal)
```
```bash
python3 generate_lipsync.py --background #FF0000    # Red (hexadecimal with #)
```

---

## 🎨 Customizing mouth images

You can change the images by inserting your PNG files in the folder:
```
assets/mouth_images/
```

| Viseme | Description | PNG file name |
|--------|-------------|-------------------|
| A | Closed mouth for M, B, P | `mouth_A.png` |
| B | Semi-open mouth for K, S, T | `mouth_B.png` |
| C | Open mouth for E, AE | `mouth_C.png` |
| D | Wide mouth for AA, O | `mouth_D.png` |
| E | Round mouth for O, ER | `mouth_E.png` |
| F | Narrow mouth for UW, W | `mouth_F.png` |
| G | Teeth over lip for F, V | `mouth_G.png` |
| H | Visible tongue for L | `mouth_H.png` |
| X | Resting mouth | `mouth_X.png` |

---

## ❓ Frequently Asked Questions

- **How to check if FFmpeg is installed?**
  ```bash
  ffmpeg -version
  ```

- **Transparent background not working?**
  - Use a compatible video player (e.g., QuickTime, Adobe Premiere, DaVinci Resolve).

- **Problems with Whisper and GPU?**
  - Use the `--cpu` option or install PyTorch compatible with your GPU.

---

## 🙏 **Acknowledgments**

Special thanks to the following open-source projects that made this repository possible:

- [**OpenAI Whisper**](https://github.com/openai/whisper) – for powerful and accurate audio transcription.
- [**Rhubarb Lip Sync**](https://github.com/DanielSWolf/rhubarb-lip-sync) – a robust tool for automated lip synchronization.

---

🎉 **Happy lipsync!** 🚀
