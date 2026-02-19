# TTS audio generation (edge_tts or Google Cloud TTS, async/sync batch).

import asyncio
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import edge_tts

import constants
from config import get_config
from errors import ProgressCallback

logger = logging.getLogger(__name__)

_GOOGLE_CLIENT = None


def _get_google_client():
    """Lazy init Google TTS client. Requires GOOGLE_APPLICATION_CREDENTIALS."""
    global _GOOGLE_CLIENT
    if _GOOGLE_CLIENT is not None:
        return _GOOGLE_CLIENT
    try:
        from google.cloud import texttospeech
        cfg = get_config()
        creds_path = cfg.get("google_application_credentials", "")
        if creds_path and os.path.isfile(creds_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        _GOOGLE_CLIENT = texttospeech.TextToSpeechClient()
        return _GOOGLE_CLIENT
    except Exception as e:
        logger.error("Google TTS client init failed: %s", e)
        return None


def _synthesize_google_api_key(text: str, voice_name: str, output_path: str, api_key: str) -> bool:
    """Synthesize via Google TTS REST API with API key. Returns True on success."""
    import base64
    import json
    import urllib.request
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    lang = voice_name[:5] if len(voice_name) >= 5 else "en-US"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang, "name": voice_name},
        "audioConfig": {"audioEncoding": "MP3"},
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        audio_b64 = data.get("audioContent")
        if audio_b64:
            with open(output_path, "wb") as out:
                out.write(base64.b64decode(audio_b64))
            return os.path.exists(output_path) and os.path.getsize(output_path) > constants.MIN_AUDIO_FILE_SIZE
        return False
    except Exception as e:
        logger.error("Google TTS API failed for %s: %s", text[:30], e)
        return False


def _synthesize_google(text: str, voice_name: str, output_path: str) -> bool:
    """Synthesize one file with Google Cloud TTS. Prefers API key, else SDK (service account)."""
    cfg = get_config()
    api_key = cfg.get("google_tts_api_key", "").strip()
    if api_key:
        return _synthesize_google_api_key(text, voice_name, output_path, api_key)
    client = _get_google_client()
    if not client:
        return False
    try:
        from google.cloud import texttospeech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(name=voice_name, language_code=voice_name[:5])
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return os.path.exists(output_path) and os.path.getsize(output_path) > constants.MIN_AUDIO_FILE_SIZE
    except Exception as e:
        logger.error("Google TTS failed for %s: %s", text[:30], e)
        return False


def _run_google_batch(
    tasks: List[Dict[str, str]],
    concurrency: int,
    progress_callback: Optional[ProgressCallback] = None
) -> None:
    """Run Google Cloud TTS batch with ThreadPoolExecutor."""
    total = len(tasks)
    completed = [0]
    lock = threading.Lock()

    def do_one(task: Dict[str, str]) -> bool:
        import time
        success = False
        if not os.path.exists(task["path"]):
            for attempt in range(constants.TTS_RETRY_ATTEMPTS):
                success = _synthesize_google(task["text"], task["voice"], task["path"])
                if success:
                    break
                time.sleep(1.5 * (attempt + 1))
        else:
            success = True
        with lock:
            completed[0] += 1
            c = completed[0]
        if progress_callback:
            progress_callback(c / total, f"正在生成 ({c}/{total})")
        return success

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(do_one, tasks))


async def _generate_audio_batch_edge(
    tasks: List[Dict[str, str]],
    concurrency: int = constants.TTS_CONCURRENCY,
    progress_callback: Optional[ProgressCallback] = None
) -> None:
    """Generate audio files concurrently with retry logic."""
    semaphore = asyncio.Semaphore(concurrency)
    total_files = len(tasks)
    completed_files = 0

    async def worker(task: Dict[str, str]) -> None:
        nonlocal completed_files
        async with semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.8))

            success = False
            error_msg = ""

            for attempt in range(constants.TTS_RETRY_ATTEMPTS):
                try:
                    if not os.path.exists(task['path']):
                        comm = edge_tts.Communicate(task['text'], task['voice'])
                        await comm.save(task['path'])

                        if os.path.exists(task['path']) and os.path.getsize(task['path']) > constants.MIN_AUDIO_FILE_SIZE:
                            success = True
                            break
                        else:
                            if os.path.exists(task['path']):
                                os.remove(task['path'])
                            raise Exception("File size too small")
                    else:
                        success = True
                        break
                except Exception as e:
                    error_msg = str(e)
                    await asyncio.sleep(1.5 * (attempt + 1))

            if not success:
                logger.error("TTS failed for: %s | Error: %s", task['text'], error_msg)

            completed_files += 1
            if progress_callback:
                progress_callback(
                    completed_files / total_files,
                    f"正在生成 ({completed_files}/{total_files})"
                )

    jobs = [worker(task) for task in tasks]
    await asyncio.gather(*jobs, return_exceptions=True)


def run_async_batch(
    tasks: List[Dict[str, str]],
    concurrency: int = constants.TTS_CONCURRENCY,
    progress_callback: Optional[ProgressCallback] = None,
    tts_provider: str = "edge"
) -> None:
    """Run audio generation batch. Supports edge-tts (async) and Google Cloud TTS (sync)."""
    if not tasks:
        return

    if tts_provider == "google":
        _run_google_batch(tasks, concurrency, progress_callback)
        return

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate_audio_batch_edge(tasks, concurrency, progress_callback))
        loop.close()
    except Exception as e:
        logger.error("Async loop error: %s", e)
