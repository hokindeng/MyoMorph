# MotionGPT Setup and Bug Fixes

This document contains all the fixes applied to make MotionGPT work properly.

## Fixed Issues

### 1. NumPy Compatibility Issue
**Problem:** NumPy 2.x incompatible with OpenCV compiled against NumPy 1.x
**Solution:** Downgrade NumPy to 1.26.4
```bash
pip install "numpy<2.0"
```

### 2. MoviePy Editor Module Missing
**Problem:** `ModuleNotFoundError: No module named 'moviepy.editor'`
**Solution:** Downgrade to MoviePy 1.0.3
```bash
pip install "moviepy==1.0.3"
```

### 3. Missing Shapely Package
**Problem:** `ModuleNotFoundError: No module named 'shapely'`
**Solution:** Install shapely
```bash
pip install shapely
```

### 4. Wrong Whisper Model Path
**Problem:** Invalid path `deps/whisper-large-v2` causing repository not found error
**Solution:** Changed to official Hugging Face model path in `configs/assets.yaml`:
```yaml
model:
  whisper_path: openai/whisper-large-v2
```

### 5. Gradio API Compatibility
**Problem:** `TypeError: Audio.__init__() got an unexpected keyword argument 'source'`
**Solution:** Updated to modern Gradio 5.x API in `app.py`:
```python
# Old (deprecated)
aud = gr.Audio(source="microphone", ...)

# New (working)
aud = gr.Audio(sources=["microphone"], ...)
```

## Quick Setup Instructions

1. **Activate your conda environment:**
   ```bash
   conda activate mgpt
   ```

2. **Install fixed dependencies:**
   ```bash
   pip install "numpy<2.0"
   pip install "moviepy==1.0.3"
   pip install shapely
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **For public access, the app launches with share=True enabled**
   - Local URL: http://localhost:8889
   - Public URL: Will be displayed in terminal (e.g., https://xxxxxxxx.gradio.live)

## Environment
- Python: 3.10
- NumPy: 1.26.4 (critical - do not upgrade to 2.x)
- MoviePy: 1.0.3
- Gradio: 5.34.0
- Shapely: Latest version

## Files Modified
- `app.py`: Fixed Gradio Audio API and added public sharing
- `configs/assets.yaml`: Fixed whisper model path
- `.gitignore`: Added comprehensive ignore patterns
- `requirements_fixed.txt`: Documented working package versions 