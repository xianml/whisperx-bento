from __future__ import annotations
import bentoml
import whisperx
import numpy as np
from typing import Any
from bentoml.io import File, Text, JSON
from transformers.pipelines.audio_utils import ffmpeg_read
import torch
import io

MODEL_TAG = "large-v2"
SAMPLE_RATE = 16000

class Whisper(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = whisperx.load_model(MODEL_TAG, self.device)

    @bentoml.Runnable.method(batchable=False)
    def decode(self, data: bytes):
        audio_nparray = ffmpeg_read(data, SAMPLE_RATE)

        #1. Transcribe with original whisper (batched)
        result = self.model.transcribe(audio_nparray)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_nparray, self.device, return_char_alignments=False)
        return result["segments"]

runner = bentoml.Runner(
    Whisper,
    name="whisper",
)

svc = bentoml.Service(
    "bentoml-whisperx",
    runners=[runner],
)

@svc.api(input=File(), output=JSON())
async def decode(audio: io.BytesIO[Any]) -> dict[str, Any]:
    try:
      return await runner.decode.async_run(audio.read())
    except:
      return {"error_msg":"Ensure that the soundfile has a valid audio file extension (e.g. wav, flac or mp3)"}