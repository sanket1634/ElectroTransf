import os
import tempfile
import subprocess
import glob
import re
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/rip-pdf")
async def rip_pdf(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(prefix="flexo_rip_")
    pdf_path = os.path.join(temp_dir, file.filename)
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    base_out = os.path.join(temp_dir, "plate.tif")
    
    # Run Ghostscript to extract true pixel densities and spot channels
    cmd = [
        "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=tiffsep", 
        "-dSimulateOverprint=true", "-r72", f"-sOutputFile={base_out}", pdf_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    coverages = {}
    sep_files = glob.glob(os.path.join(temp_dir, "plate(*.tif"))
    
    for tif in sep_files:
        m = re.search(r"plate\((.+)\)\.tif$", os.path.basename(tif))
        if not m: continue
        channel_name = m.group(1)
        
        # Ignore CAD cutting lines
        if any(ignore in channel_name.lower() for ignore in ["dieline", "die", "cut"]):
            continue
            
        with Image.open(tif) as img:
            arr = np.asarray(img, dtype=np.float32)
            # 255 = paper white (0% ink), 0 = solid ink (100%)
            ink_matrix = (255.0 - arr) / 255.0
            mean_cov = float(np.mean(ink_matrix))
            
        if mean_cov > 0.001:
            coverages[channel_name] = round(mean_cov * 100, 2)
            
    # Wipe the heavy PDF from the server to save space
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {"filename": file.filename, "plates": coverages}