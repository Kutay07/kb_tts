from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from funcs import initialize_model, synthesize_chunks, synthesize_to_file, synthesize_to_json, stream_chunks, list_speakers

# FastAPI application
app = FastAPI()

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "tr"
    speaker: str = "Dionisio Schuyler"

@app.post("/synthesize")
async def synthesize_api(request: SynthesizeRequest, background_tasks: BackgroundTasks):
    text = request.text
    language = request.language
    speaker = request.speaker

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    background_tasks.add_task(synthesize_chunks, text, language, speaker)

    return JSONResponse(content={"message": "Voice synthesis has started."})

@app.post("/synthesize_json")
async def synthesize_json_api(request: SynthesizeRequest):
    text = request.text
    language = request.language
    speaker = request.speaker

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    wav_tensor = await synthesize_to_json(text, language, speaker)
    return JSONResponse(content={"wav": wav_tensor})

@app.post("/synthesize_file")
async def synthesize_file_api(request: SynthesizeRequest):
    text = request.text
    language = request.language
    speaker = request.speaker

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    file_path = await synthesize_to_file(text, language, speaker)
    return FileResponse(file_path)

@app.get("/stream_audio")
async def stream_audio():
    return StreamingResponse(stream_chunks(), media_type="audio/wav")

if __name__ == "__main__":
    initialize_model()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
