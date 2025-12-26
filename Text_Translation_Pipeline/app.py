import gradio as gr
import cv2
import numpy as np
import torch
from deep_translator import GoogleTranslator
import logging
from ultralytics import YOLO
import os
import shutil
import time

# --- Import TextRecognition ---
try:
    from paddleocr import TextRecognition
except ImportError:
    print("⚠️ 'TextRecognition' class not found. Fallback to PaddleOCR.")
    from paddleocr import PaddleOCR as TextRecognition 

logging.getLogger("ppocr").setLevel(logging.ERROR)

# --- 1. Configurations (Using Mobile Models) ---
CLASS_MAP = {
    0: {"name": "Arabic",   "model": "arabic_PP-OCRv3_mobile_rec",     "trans": "ar"},
    1: {"name": "Latin",    "model": "en_PP-OCRv3_mobile_rec",         "trans": "auto"},
    2: {"name": "Chinese",  "model": "ch_PP-OCRv4_mobile_rec",         "trans": "zh-CN"},
    3: {"name": "Korean",   "model": "korean_PP-OCRv3_mobile_rec",     "trans": "ko"},
    4: {"name": "Japanese", "model": "japan_PP-OCRv3_mobile_rec",      "trans": "ja"},
    5: {"name": "Bangla",   "model": "bangla_PP-OCRv3_mobile_rec",     "trans": "bn"}, 
    6: {"name": "Hindi",    "model": "devanagari_PP-OCRv3_mobile_rec", "trans": "hi"},
    7: {"name": "Other",    "model": "en_PP-OCRv3_mobile_rec",         "trans": "auto"}
}

print("Loading Custom YOLO OBB Model...")
model = YOLO('best.pt') 

rec_engines = {}

def get_rec_model(model_name):
    if model_name not in rec_engines:
        print(f"Initializing Recognition Model: {model_name}...")
        try:
            rec_engines[model_name] = TextRecognition(model_name=model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            rec_engines[model_name] = None
    return rec_engines[model_name]

# --- 2. SIMPLE CROP (No Warp) ---
def get_simple_crop(image, xyxyxyxy):
    pts = np.array(xyxyxyxy, dtype=int)
    x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    
    h, w = image.shape[:2]
    # Add 2px padding to ensure we don't cut off edges
    crop = image[max(0, y_min-2):min(h, y_max+2), max(0, x_min-2):min(w, x_max+2)]
    return crop

# --- 3. Pipeline ---
def pipeline(input_image, target_lang_code):
    # Setup Debug Directory
    if os.path.exists("debug_crops"):
        shutil.rmtree("debug_crops") # Clean old crops
    os.makedirs("debug_crops", exist_ok=True)

    img_np = np.array(input_image)
    
    # 1. Detection
    results = model.predict(img_np, task='obb', conf=0.25)
    
    annotated_img = img_np.copy()
    output_data = []
    gallery_images = [] # For Gradio display

    temp_crop_path = "temp_crop.png"

    for result in results:
        if result.obb is not None:
            obb_coords = result.obb.xyxyxyxy.cpu().numpy().astype(int)
            class_ids = result.obb.cls.cpu().numpy().astype(int)
            
            for i, box in enumerate(obb_coords):
                cls_id = class_ids[i]
                lang_info = CLASS_MAP.get(cls_id, CLASS_MAP[7])
                
                # 2. CROP
                cropped_roi = get_simple_crop(img_np, box)
                if cropped_roi.size == 0: continue

                # SAVE CROP FOR DEBUGGING
                # Save with timestamp to avoid overwriting
                timestamp = int(time.time() * 1000)
                debug_filename = f"debug_crops/{lang_info['name']}_{i}_{timestamp}.png"
                # Convert RGB to BGR for OpenCV saving
                cv2.imwrite(debug_filename, cv2.cvtColor(cropped_roi, cv2.COLOR_RGB2BGR))
                
                # Add to Gallery (Tuple: Image, Label)
                gallery_images.append((cropped_roi, f"{lang_info['name']}"))

                # 3. RECOGNITION
                detected_text = ""
                try:
                    rec_model = get_rec_model(lang_info['model'])
                    if rec_model:
                        # We use the FILE PATH we just saved to ensure compatibility
                        rec_results = rec_model.predict(input=debug_filename)
                        
                        # DEBUG PRINT
                        print(f"[DEBUG] File: {debug_filename} | Raw Output: {rec_results}")
                        
                        # Parse
                        texts = []
                        iterable_res = rec_results if isinstance(rec_results, list) else [rec_results]
                        
                        for res in iterable_res:
                            if hasattr(res, 'rec_text'):
                                texts.append(res.rec_text)
                            elif isinstance(res, dict) and 'rec_text' in res:
                                texts.append(res['rec_text'])
                            elif isinstance(res, tuple) and len(res) > 0:
                                texts.append(str(res[0]))
                        
                        detected_text = " ".join(texts)
                        
                except Exception as e:
                    print(f"[ERROR] Rec failed for {lang_info['name']}: {e}")
                    detected_text = "Error"

                # 4. TRANSLATE
                translated_text = ""
                if detected_text.strip() and detected_text != "Error":
                    try:
                        translator = GoogleTranslator(source=lang_info['trans'], target=target_lang_code)
                        translated_text = translator.translate(detected_text)
                    except:
                        translated_text = "Trans Fail"

                output_data.append([lang_info['name'], detected_text, translated_text])
                
                # Visuals
                pts = box.reshape((-1, 1, 2))
                color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)
                cv2.polylines(annotated_img, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(annotated_img, f"{lang_info['name']}", (box[0][0], box[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_img, output_data, gallery_images


# --- Interface ---
language_options = [("English", "en"), ("Hindi", "hi"), ("Telugu", "te"), ("Tamil", "ta"), ("Spanish", "es")]

# 1. Define your list of example files (filenames only!)
example_images = [
    ["arabic.jpg", "en"],   # [Image File, Default Language]
    ["korean.jpg", "en"],
]

iface = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=[code for name, code in language_options], value="en", label="Destination Language")
    ],
    outputs=[
        gr.Image(type="numpy", label="Annotated Output"),
        gr.Dataframe(headers=["YOLO Class", "Detected Text", "Translated Text"], label="Results"),
        gr.Gallery(label="Debug Crops (Saved to 'debug_crops' folder)")
    ],
    title="End to End Text Translation Pipeline",
    description="Click one of the examples below to test quickly.",
    examples=example_images 
)

if __name__ == "__main__":
    iface.launch()