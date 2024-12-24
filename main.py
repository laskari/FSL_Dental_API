import os, glob

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, HTMLResponse

app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Log directories for different services
log_dirs = {
    "service1": r"D:\project\FSL\new_codebase\FSL_Dental_API\logs",
    "service2": r"D:\project\FSL\new_codebase\FSL_HCFA_API\logs",
    "service3": r"D:\project\FSL\new_codebase\FSL_UB_API\logs",
}
def get_latest_log_file(service: str):
    """
    Get the latest log file for the given service.
    
    :param service: The name of the service
    :return: The path to the latest log file, or None if no log files are found
    """
    log_dir = log_dirs.get(service)
    if not log_dir:
        return None

    # Search for log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.log*"))
    if not log_files:
        return None

    # Sort files by modification time (latest first)
    log_files.sort(key=os.path.getmtime, reverse=True)

    # Return the latest log file
    return log_files[0]

# Optional: Redirect root to the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=open("frontend/test.html").read())

@app.get("/read-error-log")
async def read_error_log(service: str = Query(..., description="Service name")):
    log_file = get_latest_log_file(service)
    print(log_file)
    if not log_file:
        raise HTTPException(status_code=404, detail="Service log file not found.")

    elif log_file is None:
        raise HTTPException(status_code=404, detail="No Log file.")
    try:
        with open(log_file, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    # port = int(sys.argv[1]) if  len(sys.argv) > 1 else 5000
    uvicorn.run("main:app", host="localhost", port=5002)