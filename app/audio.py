import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from gtts import gTTS

from .config import settings

router = APIRouter()


@router.post('/speech-to-text')
async def speech_to_text(file: UploadFile = File(...)):
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail='OPENAI_API_KEY is required for server-side STT')

    try:
        import openai
        openai.api_key = settings.OPENAI_API_KEY
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name

        # Use Whisper via OpenAI
        with open(tmp_path, 'rb') as audio_file:
            resp = openai.Audio.transcribe('whisper-1', audio_file)
        os.unlink(tmp_path)
        return {"transcript": resp.get('text')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/text-to-speech')
async def text_to_speech(text: str):
    try:
        tts = gTTS(text=text, lang='en')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tmp.close()
        tts.save(tmp.name)

        def iterfile():
            with open(tmp.name, 'rb') as f:
                yield from f
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        return StreamingResponse(iterfile(), media_type='audio/mpeg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
