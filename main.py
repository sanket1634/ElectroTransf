import os
import tempfile
import subprocess
import glob
import re
import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. THE VECTORIZATION ENGINE (For JPEG/PNG)
@app.post("/api/image-to-vector")
async def image_to_vector(file: UploadFile = File(...), max_colors: int = Form(5)):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
        
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pixels = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, max_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    coverages = {}
    visual_previews = []
    total_pixels = labels.shape[0]
    trap_kernel = np.ones((3,3), np.uint8) 
    
    for i in range(max_colors):
        color = centers[i]
        
        # Skip white background
        if color[0] > 240 and color[1] > 240 and color[2] > 240:
            continue 
            
        mask = (labels.flatten() == i).reshape(img.shape[:2]).astype(np.uint8)
        cov_pct = (np.sum(mask) / total_pixels) * 100
        
        if cov_pct > 0.1:
            trapped_mask = cv2.dilate(mask, trap_kernel, iterations=1)
            
            bmp_path = os.path.join(temp_dir, f"plate_{i}.bmp")
            svg_path = os.path.join(temp_dir, f"plate_{i}.svg")
            
            binary_img = np.ones_like(trapped_mask, dtype=np.uint8) * 255
            binary_img[trapped_mask == 1] = 0 
            cv2.imwrite(bmp_path, binary_img)
            
            subprocess.run(["potrace", bmp_path, "-s", "-o", svg_path])
            
            with open(svg_path, "rb") as f:
                svg_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            plate_name = f"SPOT-{hex_color.upper()}"
            
            coverages[plate_name] = round(cov_pct, 2)
            visual_previews.append({
                "name": plate_name,
                "hex": hex_color,
                "coverage": round(cov_pct, 2),
                "svg_base64": f"data:image/svg+xml;base64,{svg_b64}"
            })
            
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {"filename": file.filename, "plates": coverages, "previews": visual_previews}

# 2. THE GHOSTSCRIPT RIP ENGINE (For PDFs)
@app.post("/api/rip-studio")
async def rip_studio(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(prefix="flexo_rip_")
    pdf_path = os.path.join(temp_dir, file.filename)
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    base_out = os.path.join(temp_dir, "plate.tif")
    
    cmd = [
        "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=tiffsep", 
        "-dSimulateOverprint=true", "-r300", f"-sOutputFile={base_out}", pdf_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    coverages = {}
    visual_previews = []
    sep_files = glob.glob(os.path.join(temp_dir, "plate(*.tif"))
    
    for tif in sep_files:
        m = re.search(r"plate\((.+)\)\.tif$", os.path.basename(tif))
        if not m: continue
        channel_name = m.group(1)
        
        if any(ignore in channel_name.lower() for ignore in ["dieline", "die", "cut"]):
            continue
            
        with Image.open(tif) as img:
            arr = np.asarray(img, dtype=np.float32)
            ink_matrix = (255.0 - arr) / 255.0
            mean_cov = float(np.mean(ink_matrix))
            
        if mean_cov > 0.001: 
            cov_pct = round(mean_cov * 100, 2)
            coverages[channel_name] = cov_pct
            
            pil_plate = Image.fromarray(arr.astype(np.uint8), mode='L')
            buffered = io.BytesIO()
            pil_plate.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            visual_previews.append({
                "name": channel_name,
                "coverage": cov_pct,
                "base64": f"data:image/png;base64,{img_str}"
            })
            
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {"filename": file.filename, "plates": coverages, "previews": visual_previews}
