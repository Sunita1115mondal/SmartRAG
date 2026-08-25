"""
Audio processor for speech-to-text conversion and audio content extraction.

Whisper is loaded lazily only when an audio file needs transcription.
By default, Whisper runs on CPU to avoid CUDA out-of-memory conflicts
with Ollama/Qwen on GPUs with limited VRAM.
"""

import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

try:
    import whisper
except ImportError:
    whisper = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import librosa
except ImportError:
    librosa = None

from ..base import BaseProcessor, ProcessingResult

logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor):
    """Processor for audio files with speech-to-text capabilities."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.supported_extensions = [
            '.mp3',
            '.wav',
            '.m4a',
            '.ogg',
            '.flac',
            '.aac'
        ]

        self.processor_type = "audio"

        self.chunk_size = config.get(
            'processing', {}
        ).get('chunk_size', 1000)

        self.chunk_overlap = config.get(
            'processing', {}
        ).get('chunk_overlap', 200)

        # Whisper configuration
        self.whisper_model_name = config.get(
            'models', {}
        ).get('whisper_model', 'base')

        self.max_audio_duration = config.get(
            'processing', {}
        ).get('max_audio_duration', 300)

        self.sample_rate = config.get(
            'processing', {}
        ).get('audio_sample_rate', 16000)

        # Run Whisper on CPU by default.
        #
        # This is intentional because Ollama/Qwen is using the RTX 3050 GPU.
        # Loading both Ollama and Whisper onto a 6 GB GPU can cause CUDA OOM.
        self.whisper_device = config.get(
            'models', {}
        ).get('whisper_device', 'cpu')

        # Check library availability
        self.whisper_available = whisper is not None
        self.pydub_available = AudioSegment is not None
        self.speech_recognition_available = sr is not None
        self.librosa_available = librosa is not None

        if not self.whisper_available:
            logger.warning(
                "Whisper not available. "
                "Install with: pip install openai-whisper"
            )

        if not self.pydub_available:
            logger.warning(
                "pydub not available. "
                "Install with: pip install pydub"
            )

        if not self.speech_recognition_available:
            logger.info(
                "SpeechRecognition not available. "
                "Whisper will be used if available."
            )

        if not self.librosa_available:
            logger.info(
                "librosa not available. "
                "pydub will be used for audio metadata if available."
            )

        # IMPORTANT:
        # Do NOT load Whisper here.
        #
        # Whisper will only be loaded when an audio file is actually
        # uploaded and transcription is required.
        self.whisper_model = None

        logger.info(
            "AudioProcessor initialized. "
            f"Whisper model '{self.whisper_model_name}' "
            f"will be loaded on demand using device: "
            f"{self.whisper_device}"
        )

        # Initialize SpeechRecognition
        self.speech_recognizer = None

        if self.speech_recognition_available:
            try:
                self.speech_recognizer = sr.Recognizer()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize SpeechRecognition: {str(e)}"
                )
                self.speech_recognizer = None

    def _load_whisper_model(self):
        """
        Load Whisper only when audio transcription is requested.

        Lazy loading prevents Whisper from consuming memory when the
        SmartRAG application starts.
        """

        # Model already loaded
        if self.whisper_model is not None:
            return

        if not self.whisper_available:
            raise RuntimeError(
                "Whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )

        try:
            logger.info(
                f"Loading Whisper model '{self.whisper_model_name}' "
                f"on {self.whisper_device}..."
            )

            self.whisper_model = whisper.load_model(
                self.whisper_model_name,
                device=self.whisper_device
            )

            logger.info(
                f"Whisper model '{self.whisper_model_name}' "
                f"loaded successfully on {self.whisper_device}"
            )

        except Exception as e:
            self.whisper_model = None

            logger.error(
                f"Failed to load Whisper model "
                f"'{self.whisper_model_name}': {str(e)}"
            )

            raise

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the audio file."""

        path = Path(file_path)

        return (
            path.suffix.lower() in self.supported_extensions
            and (
                self.whisper_available
                or self.speech_recognition_available
            )
        )

    def extract_content(
        self,
        file_path: Union[str, Path]
    ) -> ProcessingResult:
        """Extract transcript and metadata from an audio file."""

        start_time = time.time()

        try:
            path = Path(file_path)

            if not path.exists():
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=f"Audio file not found: {path}"
                )

            if not self.can_process(path):
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=(
                        f"Unsupported or unavailable audio format: "
                        f"{path.suffix}"
                    )
                )

            logger.info(f"Processing audio file: {path}")

            # Get common file metadata
            metadata = self._get_file_metadata(path)

            # Get audio metadata
            audio_info = self._get_audio_info(path)
            metadata.update(audio_info)

            # Check maximum allowed duration
            duration = audio_info.get('duration', 0)

            if duration and duration > self.max_audio_duration:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=(
                        f"Audio too long: {duration:.2f}s "
                        f"> {self.max_audio_duration}s"
                    )
                )

            transcript = None
            confidence = None

            # Prefer Whisper
            if self.whisper_available:

                logger.info("Using Whisper for transcription")

                transcript, confidence = (
                    self._transcribe_with_whisper(path)
                )

            # Fallback to SpeechRecognition
            elif self.speech_recognition_available:

                logger.info(
                    "Whisper unavailable. "
                    "Using SpeechRecognition."
                )

                transcript, confidence = (
                    self._transcribe_with_speech_recognition(path)
                )

            if not transcript or not transcript.strip():

                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=(
                        "No speech could be detected or transcribed "
                        "from audio"
                    )
                )

            transcript = transcript.strip()

            # Add transcription metadata
            metadata['transcript_length'] = len(transcript)
            metadata['word_count'] = len(transcript.split())

            if confidence is not None:
                metadata['transcription_confidence'] = confidence

            # Create RAG chunks from transcript
            chunks = self._create_chunks(
                transcript,
                metadata,
                self.chunk_size,
                self.chunk_overlap
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Audio processing completed. "
                f"Created {len(chunks)} chunks."
            )

            return ProcessingResult(
                chunks=chunks,
                success=True,
                processing_time=processing_time,
                metadata={
                    'chunks_created': len(chunks),
                    'transcription_method': (
                        'whisper'
                        if self.whisper_available
                        else 'speech_recognition'
                    )
                }
            )

        except Exception as e:

            logger.error(
                f"Error processing audio file {file_path}: {str(e)}"
            )

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process audio: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _get_audio_info(
        self,
        path: Path
    ) -> Dict[str, Any]:
        """Extract audio metadata."""

        audio_info = {}

        try:
            # Preferred method: pydub
            if self.pydub_available:

                audio = AudioSegment.from_file(path)

                audio_info.update({
                    'duration': len(audio) / 1000.0,
                    'sample_rate': audio.frame_rate,
                    'channels': audio.channels,
                    'sample_width': audio.sample_width,
                    'frame_count': audio.frame_count()
                })

                return audio_info

            # Fallback method: librosa
            if self.librosa_available:

                duration = librosa.get_duration(
                    path=str(path)
                )

                audio_info['duration'] = duration

        except Exception as e:

            logger.warning(
                f"Failed to extract audio information: {str(e)}"
            )

        return audio_info

    def _transcribe_with_whisper(
        self,
        path: Path
    ) -> tuple[Optional[str], Optional[float]]:
        """Transcribe audio using Whisper."""

        try:

            # Lazy-load the Whisper model only now.
            if self.whisper_model is None:

                self._load_whisper_model()

            logger.info(
                f"Starting Whisper transcription for: {path.name}"
            )

            # fp16=False is required/recommended for CPU inference.
            # If whisper_device is later changed to CUDA, Whisper can
            # automatically use fp16 when appropriate.
            use_fp16 = self.whisper_device == "cuda"

            result = self.whisper_model.transcribe(
                str(path),
                fp16=use_fp16
            )

            transcript = result.get('text', '').strip()

            # Calculate average segment confidence
            segments = result.get('segments', [])

            if segments:

                confidences = [
                    segment.get('avg_logprob')
                    for segment in segments
                    if segment.get('avg_logprob') is not None
                ]

                if confidences:

                    avg_confidence = (
                        sum(confidences) / len(confidences)
                    )

                else:

                    avg_confidence = None

            else:

                avg_confidence = None

            return transcript, avg_confidence

        except Exception as e:

            logger.error(
                f"Whisper transcription failed: {str(e)}"
            )

            return None, None

    def _transcribe_with_speech_recognition(
        self,
        path: Path
    ) -> tuple[Optional[str], Optional[float]]:
        """Transcribe audio using SpeechRecognition."""

        temp_wav_path = None

        try:

            if self.speech_recognizer is None:
                raise RuntimeError(
                    "SpeechRecognition is not initialized"
                )

            # Convert non-WAV audio to WAV
            if (
                self.pydub_available
                and path.suffix.lower() != '.wav'
            ):

                audio = AudioSegment.from_file(path)

                temp_wav_path = path.with_name(
                    f"{path.stem}_temp.wav"
                )

                audio.export(
                    temp_wav_path,
                    format="wav"
                )

                wav_path = temp_wav_path

            else:

                wav_path = path

            # Read audio
            with sr.AudioFile(str(wav_path)) as source:

                audio_data = (
                    self.speech_recognizer.record(source)
                )

            transcript = None
            confidence = None

            # Google Speech Recognition
            try:

                transcript = (
                    self.speech_recognizer.recognize_google(
                        audio_data
                    )
                )

            except sr.RequestError:

                logger.warning(
                    "Google Speech Recognition unavailable. "
                    "Trying offline Sphinx recognition."
                )

                try:

                    transcript = (
                        self.speech_recognizer.recognize_sphinx(
                            audio_data
                        )
                    )

                except (
                    sr.RequestError,
                    sr.UnknownValueError
                ):

                    transcript = None

            except sr.UnknownValueError:

                logger.warning(
                    "SpeechRecognition could not understand the audio."
                )

                transcript = None

            return transcript, confidence

        except Exception as e:

            logger.error(
                f"Speech recognition failed: {str(e)}"
            )

            return None, None

        finally:

            # Remove temporary WAV file
            if (
                temp_wav_path is not None
                and temp_wav_path.exists()
            ):

                try:
                    temp_wav_path.unlink()
                except Exception as e:
                    logger.warning(
                        f"Could not delete temporary audio file: {str(e)}"
                    )

    def _segment_long_audio(
        self,
        path: Path,
        segment_duration: int = 30
    ) -> List[Path]:
        """
        Split long audio into smaller segments.

        This method is currently available for future use.
        """

        if not self.pydub_available:

            return [path]

        try:

            audio = AudioSegment.from_file(path)

            segment_length_ms = (
                segment_duration * 1000
            )

            segments = []

            for index, start_ms in enumerate(
                range(
                    0,
                    len(audio),
                    segment_length_ms
                )
            ):

                end_ms = min(
                    start_ms + segment_length_ms,
                    len(audio)
                )

                segment = audio[start_ms:end_ms]

                segment_path = path.with_name(
                    f"{path.stem}_segment_{index}"
                    f"{path.suffix}"
                )

                segment.export(
                    segment_path,
                    format=path.suffix[1:]
                )

                segments.append(segment_path)

            return segments

        except Exception as e:

            logger.warning(
                f"Audio segmentation failed: {str(e)}"
            )

            return [path]

    def unload_whisper_model(self):
        """
        Unload Whisper from memory.

        Useful after processing audio if the application needs to
        release RAM/GPU resources.
        """

        if self.whisper_model is not None:

            logger.info("Unloading Whisper model")

            self.whisper_model = None

            # Clear CUDA cache only if CUDA was actually used.
            if self.whisper_device == "cuda":

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    pass


class AudioProcessorManager:
    """Manager for audio processing operations."""

    def __init__(self, config: Dict[str, Any]):

        self.config = config

        self.processor = AudioProcessor(config)

    def process_file(
        self,
        file_path: Union[str, Path]
    ) -> ProcessingResult:
        """Process an audio file."""

        return self.extract_content(file_path)

    def extract_content(
        self,
        file_path: Union[str, Path]
    ) -> ProcessingResult:
        """Extract content from an audio file."""

        if not self.processor.can_process(file_path):

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=(
                    f"Cannot process audio file: {file_path}"
                )
            )

        return self.processor.extract_content(file_path)

    def get_supported_extensions(self) -> List[str]:
        """Get all supported audio extensions."""

        return self.processor.supported_extensions

    def is_available(self) -> bool:
        """Check whether audio processing is available."""

        return (
            self.processor.whisper_available
            or self.processor.speech_recognition_available
        )

    def can_process(
        self,
        file_path: Union[str, Path]
    ) -> bool:
        """Check whether this manager can process the file."""

        return self.processor.can_process(file_path)

    def unload_models(self):
        """Release loaded audio models from memory."""

        self.processor.unload_whisper_model()