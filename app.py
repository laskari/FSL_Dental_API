import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form, HTTPException

from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form
import torch
import json, os

from src.extraction_util import run_ada_pipeline
from src.logger import log_message, setup_logger
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger(LOGFILE_DIR)
app = FastAPI()

@app.get("/")
async def root_route():
    return "Application working"

# @app.post("/ada_extraction")
# async def ml_extraction(data: dict):
#     try:

#         # Get the image path from the payload
#         # image_file_path = data["ClipData"][0]['FilePath']

#         # image_file_path = data.get('FilePath')

#         image_file_path = data.get('FilePath')

#         if not image_file_path:
#             raise HTTPException(status_code=400, detail="FilePath field is required")

#         if not os.path.exists(image_file_path):
#             raise HTTPException(status_code=400, detail=f"File not found: {image_file_path}")

#         result, error = run_ada_pipeline(image_file_path)

#         if error:
#             # If there's an error, raise HTTPException with status code 500 (Internal Server Error)
#             raise HTTPException(status_code=500, detail=error)
        

#         # If there's no error, return the result with file path
#         response_data = {"file_path": data.get('FilePath'), "result": result['result']}
#         return JSONResponse(content=response_data)

#     except Exception as e:
#         print(f"Error occured while processing {e} >>> ")
#         return JSONResponse(
#             status_code=500,
#             content=f"Error while processing Extraction {e}"
#         )

# @app.post("/ada_extraction")
# async def ml_extraction(file: UploadFile = File(...)):
#     try:
#         # Read the contents of the uploaded file
#         contents = await file.read()

#         file_name = file.filename

#         # Call the function to process the image file contents
#         result, error = run_ada_pipeline(contents, file_name)

#         if error:
#             # If there's an error, raise HTTPException with status code 500 (Internal Server Error)
#             raise HTTPException(status_code=500, detail=error)

#         # print({"result": result['result']})
#         # If there's no error, return the result
#         response_data= {"result": result['result']}
#         return JSONResponse(content=response_data)

#     except Exception as e:
#         print(f"Error occurred while processing {e}")
#         # Return an error response
#         return JSONResponse(
#             status_code=500,
#             content=f"Error while processing Extraction {e}"
#         )

@app.post("/ada_extraction")
async def ml_extraction(data: dict):
    try:
        # Log start of the extraction
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
        result, error = run_ada_pipeline(image_file_path)

        if error:
            # Log the error before raising HTTPException
            log_message(logger, f"Error in pipeline: {error}", level="ERROR")
            raise HTTPException(status_code=500, detail=error)

        log_message(logger, "Pipeline ran successfully", level="INFO")

        # If there's no error, return the result with file path
        response_data = {"file_path": data.get('FilePath'), "result": result['result']}
        
        # Log the successful result
        log_message(logger, f"Extraction result: {response_data}", level="INFO")
        return JSONResponse(content=response_data)

    except Exception as e:
        log_message(logger, f"Error occurred: {e}", level="ERROR")
        return JSONResponse(
            status_code=500,
            content=f"Error while processing Extraction {e}"
        )

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8002)
