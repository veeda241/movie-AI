# movie-AI

Python backend for a multiagent movie scene creator that uses Hugging Face text-generation models for scene planning and a routed video pipeline with a local MP4 fallback for scene playback.

## Project Layout

```text
movie_pipeline/
├── main.py
├── agents/
│   ├── director.py
│   ├── screenwriter.py
│   ├── cinematographer.py
│   ├── editor.py
│   └── video_organizer.py
├── pipeline/
│   ├── orchestrator.py
│   └── scene_packet.py
├── video/
│   └── motif_client.py
├── output/
└── streamlit_app.py
```

## Setup

1. Clone the repository.
2. On Windows, activate the repository virtual environment before running commands:
	```powershell
	.\.venv\Scripts\Activate.ps1
	```

   If you prefer not to activate it, use `.\.venv\Scripts\python.exe` in the commands below.
3. Install dependencies from the repository root:

	```powershell
	python -m pip install -r requirements.txt
	```

4. Set the required environment variables in PowerShell:

	```powershell
	$env:HF_TOKEN = "your-hugging-face-token"
	$env:HF_TEXT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
	```

Make sure `HF_TOKEN` is set in the same shell session that launches the app. If you open a new terminal, export it again before running Streamlit or the CLI.

`HF_TEXT_MODEL` is optional. If you do not set it, the code defaults to `meta-llama/Meta-Llama-3-8B-Instruct`.

Optional video settings:

	```powershell
	$env:HF_VIDEO_PROVIDER = "fal-ai"
	$env:HF_VIDEO_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
	```

If Hugging Face provider credits are unavailable or the routed model cannot return a video, the app automatically writes a local cinematic MP4 fallback so the Streamlit UI still shows playable video.
That fallback now streams frames through the bundled ffmpeg binary from `imageio-ffmpeg`, so no separate system ffmpeg install is required.

## Run

1. Start the CLI pipeline from the repository root:

	```powershell
	python movie_pipeline/main.py
	```

2. Start the Streamlit dashboard from the repository root:

	```powershell
	python -m streamlit run streamlit_app.py
	```

If the shell is using the wrong interpreter, run the dashboard through the repository environment directly:

	```powershell
	.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
	```

The dashboard shows scene cards, the organizer manifest, a live Agent Processing panel, raw packet JSON, and inline MP4 previews when video files are available.

Generated scene packets and video stubs are written to `movie_pipeline/output`.
