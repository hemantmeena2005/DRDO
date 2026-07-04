# Aerial Object Detection System

A full-stack web application that detects **AirPlane**, **Drone**, and
**Helicopter** in uploaded images using a trained YOLOv8 model.

- **Backend:** Python, Flask, Ultralytics YOLOv8, OpenCV, Pillow
- **Frontend:** HTML, CSS, vanilla JavaScript (no frameworks)

---

## Folder structure

```
aerial-object-detector/
│
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── best.pt                # <-- place your trained YOLOv8 weights here
│
├── uploads/                # Uploaded images (created automatically)
├── outputs/                 # Annotated prediction images (created automatically)
│
├── templates/
│   └── index.html          # Main page
│
├── static/
│   ├── style.css            # Styling
│   └── script.js             # Frontend logic
│
└── README.md
```

---

## 1. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` includes:

```
flask
ultralytics
opencv-python-headless
pillow
werkzeug
```

---

## 2. Add your trained model

Copy your trained YOLOv8 weights file into the project root and make sure
it is named exactly `best.pt`:

```
aerial-object-detector/
└── best.pt
```

The model must have been trained with these 3 classes, in this order:

```
0: AirPlane
1: Drone
2: Helicopter
```

If your model's class order is different, update the `CLASS_NAMES`
dictionary at the top of `app.py` to match.

---

## 3. Run the app

```bash
python app.py
```

The server starts at:

```
http://localhost:5000
```

Open that URL in your browser.

---

## 4. Using the app

1. Drag and drop an image onto the upload area, or click to browse.
2. Click **Run Detection**.
3. View the original image, the annotated detection image, the list of
   detected classes with confidence scores, the total object count, and
   the inference time.
4. Click **Download Result** to save the annotated image.
5. Click **Try Another Image** (or **Clear**) to start over.

---

## Configuration

A few settings can be adjusted at the top of `app.py`:

| Setting               | Default | Description                                  |
|-----------------------|---------|-----------------------------------------------|
| `CONFIDENCE_THRESHOLD`| `0.4`   | Minimum confidence for a detection to be kept |
| `MAX_CONTENT_LENGTH`  | `16 MB` | Maximum upload size                           |
| `ALLOWED_EXTENSIONS`  | png, jpg, jpeg, bmp, webp | Accepted image types        |

---

## API reference

### `GET /`
Returns the web page.

### `POST /predict`
Accepts a multipart form upload with an `image` field.

**Success response (200):**
```json
{
  "success": true,
  "original_image": "/uploads/photo_ab12cd34ef.jpg",
  "image": "/outputs/result_photo_ab12cd34ef.jpg",
  "detections": [
    { "class": "Drone", "confidence": 0.95 }
  ],
  "count": 1,
  "inference_time": "35 ms"
}
```

**Error response (4xx/5xx):**
```json
{
  "success": false,
  "error": "Description of what went wrong."
}
```

---

## Troubleshooting

- **"Model is not loaded on the server"** — make sure `best.pt` exists in
  the project root and is a valid Ultralytics YOLOv8 weights file.
- **Server unavailable in the browser** — confirm `python app.py` is
  still running and that nothing else is using port 5000.
- **Slow inference on CPU** — YOLOv8 runs faster with a CUDA-enabled GPU
  and the matching PyTorch build; on CPU-only machines, expect higher
  inference times, especially on larger images.
# DRDO
