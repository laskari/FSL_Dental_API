import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form, HTTPException
import sys
from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form
import torch
import json, os

from src.extraction_util import run_ada_pipeline
from src.logger import log_message, setup_logger
from config import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger, formatter = setup_logger(LOGFILE_DIR)
app = FastAPI()

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/ada_extraction")
async def ml_extraction(data: dict):
    try:
        # Log start of the extraction
        XELP_process_request = 'XELP_process_request'
        formatter.start_timing(XELP_process_request)

        log_message(logger, "Started ml_extraction", level="INFO")
        # Get the image path from the payload
        image_file_path = data.get('FilePath')

        if not image_file_path:
            log_message(logger, "FilePath field is required", level="ERROR")
            raise HTTPException(status_code=400, detail="FilePath field is required")

        if not os.path.exists(image_file_path):
            log_message(logger, f"File not found: {image_file_path}", level="ERROR")
            raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

        log_message(logger, f"File found: {image_file_path}. Running pipeline...", level="INFO")

        # Run the pipeline and capture result and error
        result, error = run_ada_pipeline(file_name = image_file_path, logger = logger, formatter = formatter)

        if error:
            # Log the error before raising HTTPException
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        # log_message(logger, "Pipeline ran successfully", level="INFO")
        # If there's no error, return the result with file path
        response_data = {"version": VERSION, "file_path": data.get('FilePath'), "result": result['result']}
        
        # log_message(logger, f"Extraction result: {response_data}", level="INFO")
        overall_elapsed_time = formatter.stop_timing(XELP_process_request)

        log_message(logger, f"The pipeline process completed with Data extraction and ROI prediction", level="DEBUG", elapsed_time=overall_elapsed_time)
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )
    
@app.post("/ada_extraction_streaming")
async def ml_extraction(file: UploadFile = File(...)):
    try:
        # Log start of the extraction
        XELP_process_request = 'XELP_process_request'
        formatter.start_timing(XELP_process_request)

        log_message(logger, "Started ml_extraction", level="INFO")
        # Get the image path from the payload
        # image_file_path = data.get('FilePath')

        # Read the contents of the uploaded file
        contents = await file.read()

        # Get file name
        file_name = file.filename
        
        log_message(logger, f"Running pipeline...", level="INFO")

        # Run the pipeline and capture result and error
        result, error = run_ada_pipeline(content = contents, logger = logger, formatter = formatter)

        if error:
            # Log the error before raising HTTPException
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        # log_message(logger, "Pipeline ran successfully", level="INFO")
        # If there's no error, return the result with file path
        response_data = {"version": VERSION, "file_name": file_name, "result": result['result']}
        
        # log_message(logger, f"Extraction result: {response_data}", level="INFO")
        overall_elapsed_time = formatter.stop_timing(XELP_process_request)

        log_message(logger, f"The pipeline process completed with Data extraction and ROI prediction", level="DEBUG", elapsed_time=overall_elapsed_time)
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )


if __name__ == '__main__':
    port = int(sys.argv[1]) if  len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port)

