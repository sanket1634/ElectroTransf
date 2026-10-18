import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/auto-prepress")
async def auto_prepress(file: UploadFile = File(...), max_colors: int = Form(5)):
    # Load the uploaded design
    img = Image.open(file.file).convert('RGB')
    
    # Standardize size for fast processing
    img.thumbnail((1200, 1200))
    
    # AI Color Quantization: Reduce image to its dominant flexo spot colors
    q_img = img.quantize(colors=max_colors)
    palette = q_img.getpalette()
    
    q_data = np.array(q_img)
    
    plates_coverage = {}
    visual_previews = []
    
    for i in range(max_colors):
        r, g, b = palette[i*3 : i*3+3]
        
        # Ignore paper-white background (not printed)
        if r > 240 and g > 240 and b > 240:
            continue
            
        # Locate all pixels belonging to this specific ink color
        mask = (q_data == i)
        coverage_pct = float(np.mean(mask) * 100)
        
        if coverage_pct > 0.05: # Only create a plate if the color actually exists
            # Create the Flexo Negative Plate (Black = Ink, White = No Ink)
            plate_img = np.ones_like(q_data, dtype=np.uint8) * 255
            plate_img[mask] = 0 
            
            pil_plate = Image.fromarray(plate_img, mode='L')
            
            # Convert negative to base64 so the browser can download it
            buffered = io.BytesIO()
            pil_plate.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Generate a standard Spot Color Name based on HEX code
            color_name = f"SPOT-HEX-{r:02x}{g:02x}{b:02x}".upper()
            plates_coverage[color_name] = round(coverage_pct, 2)
            
            visual_previews.append({
                "name": color_name,
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "coverage": round(coverage_pct, 2),
                "base64": f"data:image/png;base64,{img_str}"
            })
            
    return {
        "filename": file.filename, 
        "plates": plates_coverage, 
        "previews": visual_previews
    }

@app.get("/")
async def root():
    return {"status": "ElectroTransf AI Prepress Engine Active."}
