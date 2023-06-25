from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import uvicorn
from TTS.api import TTS



app = FastAPI()

# Create an instance of TTS
G = False
clone = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=G)
VC = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=G)

# Dictionary to store voice samples
voice_samples = {}


class VoiceCloningRequest(BaseModel):
    username: str

# YourTTS with one-shot cloning function
def Zero_Shot_CloneTTS(voice, language, text):
    clone.tts_to_file(text=text, speaker_wav=voice, language=language, file_path="output.wav")




@app.get("/set_language/{language}/{username}")
async def set_language(language: str, username: str):
    # Check if the language is supported
    if language not in ["en", "fr-fr"]:
        return {"error": f"Unsupported language: {language}"}

    # Update the language for the user
    voice_samples[username] = language

    return {"message": f"Language set to {language} for user {username}"}



@app.post("/set_sample/{username}")
async def set_sample(username: str, audio_file: UploadFile = File(...)):
    # Save the audio sample file
    voice_sample_path = f"/home/ubuntu/{username}.wav"
    with open(voice_sample_path, "wb") as file:
        file.write(await audio_file.read())

    # Update the voice_samples dictionary
    voice_samples[username] = voice_sample_path

    return {"message": "Voice sample uploaded successfully"}


@app.get("/tts")
async def generate_tts(text: str, username: str):
    # Check if voice sample exists for the given username
    if username not in voice_samples:
        return {"error": f"No voice sample found for user {username}"}

    # Perform voice cloning with the saved voice sample
    voice_sample_path = voice_samples[username]
    language = voice_samples[username]

    cloned_audio_path = f"/home/ubuntu/{username}.wav"
    Zero_Shot_CloneTTS(voice_sample_path, language=language, text=text)
    shutil.move("output.wav", cloned_audio_path)

    # Return the cloned audio file as a response
    return FileResponse(cloned_audio_path, media_type="audio/wav")



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
