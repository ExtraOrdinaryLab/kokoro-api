import io
import random
import base64
from typing import Dict

import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from rich.console import Console
from kokoro import KModel, KPipeline

console = Console()


class KokoroTTSServer:
    """
    Server-based Kokoro TTS processor that keeps models loaded in memory.
    """
    def __init__(self, voice_options: Dict[str, str], use_gpu: bool = True):
        """
        Initialize Kokoro TTS server with models and pipelines for all voices.
        
        Args:
            voice_options (dict): Dictionary of available voices
            use_gpu (bool): Whether to use GPU if available (default: True)
        """
        self.voice_options = voice_options
        self.cuda_available = torch.cuda.is_available()
        self.device = 'cuda' if (use_gpu and self.cuda_available) else 'cpu'
        self.use_gpu = use_gpu
        
        console.log(f"[green]Initializing Kokoro TTS Server on {self.device}[/green]")
        
        # Load models once at startup
        self.model = KModel(repo_id='hexgrad/Kokoro-82M').to(self.device).eval()
        self.model_cpu = None  # Will be loaded only if needed for fallback
        
        # Initialize pipelines for both American and British English
        self.pipelines = {}
        self.voice_packs = {}
        
        for lang_code in ['a', 'b']:  # 'a' for American English, 'b' for British English
            pipeline = KPipeline(lang_code=lang_code, model=False)
            
            # Add custom pronunciation for 'kokoro' if needed
            if lang_code == 'a':
                pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            elif lang_code == 'b':
                pipeline.g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
            
            self.pipelines[lang_code] = pipeline
        
        # Load voice packs for all voices
        console.log("[yellow]Loading voice packs...[/yellow]")
        for voice_id in voice_options.keys():
            lang_code = voice_id[0]
            pack = self.pipelines[lang_code].load_voice(voice_id)
            self.voice_packs[voice_id] = pack
        
        console.log(f"[green]Server initialized with {len(voice_options)} voices[/green]")
        console.log(f"Available voices: {', '.join(voice_options.keys())}")
    
    def text_to_audio(self, text: str, voice: str, speed: float = 1.0) -> torch.Tensor:
        """
        Convert text to audio waveform using pre-loaded model and pipeline.
        
        Args:
            text (str): The text to convert to speech
            voice (str): Voice ID to use
            speed (float): Speech speed multiplier (default: 1.0)
        
        Returns:
            torch.Tensor: Audio waveform tensor (mono, 24kHz)
        """
        if voice not in self.voice_options:
            raise ValueError(f"Voice '{voice}' not available. Available voices: {list(self.voice_options.keys())}")
        
        lang_code = voice[0]
        pipeline = self.pipelines[lang_code]
        pack = self.voice_packs[voice]
        
        # Process text through pipeline
        for _, ps, _ in pipeline(text, voice, speed):
            # Get reference audio for the voice
            ref_s = pack[len(ps)-1]
            
            try:
                # Generate audio
                audio = self.model(ps, ref_s, speed)
                return audio
                
            except Exception as e:
                # Fallback to CPU if GPU fails
                if self.use_gpu and self.cuda_available:
                    console.log(f"[yellow]GPU generation failed: {e}[/yellow]")
                    console.log("[yellow]Retrying with CPU...[/yellow]")
                    
                    # Load CPU model only when needed
                    if self.model_cpu is None:
                        self.model_cpu = KModel().to('cpu').eval()
                    
                    audio = self.model_cpu(ps, ref_s, speed)
                    return audio
                else:
                    raise e
        
        # Return empty audio if no text processed
        return torch.zeros(1)
    
    def get_random_voice(self) -> str:
        """Get a random voice from available options."""
        return random.choice(list(self.voice_options.keys()))
    
    def resample_to_16khz(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resample audio from 24kHz to 16kHz.
        
        Args:
            audio_tensor (torch.Tensor): Input audio tensor at 24kHz
            
        Returns:
            torch.Tensor: Resampled audio tensor at 16kHz
        """
        # Ensure audio is 2D (channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample from 24000 to 16000 Hz
        resampler = torchaudio.transforms.Resample(
            orig_freq=24000, 
            new_freq=16000
        )
        resampled_audio = resampler(audio_tensor)
        
        # Return as 1D tensor (mono)
        return resampled_audio.squeeze(0)
    
    def get_audio_duration(self, audio_tensor: torch.Tensor, sample_rate: int = 16000) -> float:
        """
        Calculate audio duration in seconds.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor
            sample_rate (int): Sample rate of the audio
            
        Returns:
            float: Duration in seconds
        """
        return len(audio_tensor) / sample_rate


# Initialize the TTS server globally
tts_server = None


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "device": tts_server.device,
            "cuda_available": tts_server.cuda_available,
            "voices_loaded": len(tts_server.voice_options)
        })
    
    @app.route('/voices', methods=['GET'])
    def get_voices():
        """Get available voices."""
        return jsonify({
            "voices": tts_server.voice_options,
            "count": len(tts_server.voice_options)
        })
    
    @app.route('/tts', methods=['POST'])
    def text_to_speech():
        """
        Convert text to speech.
        
        Expected JSON payload:
        {
            "text": "Hello world",
            "voice": "af_heart",  # optional, random if not specified
            "speed": 1.0,         # optional, default 1.0
            "format": "wav",      # optional, default "wav"
            "return_audio": true  # optional, return base64 audio data
        }
        """
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({"error": "Missing 'text' field"}), 400
            
            text = data['text']
            voice = data.get('voice', tts_server.get_random_voice())
            speed = data.get('speed', 1.0)
            return_audio = data.get('return_audio', False)
            
            # Validate inputs
            if not isinstance(text, str) or len(text.strip()) == 0:
                return jsonify({"error": "Text must be a non-empty string"}), 400
            
            if voice not in tts_server.voice_options:
                return jsonify({
                    "error": f"Voice '{voice}' not available",
                    "available_voices": list(tts_server.voice_options.keys())
                }), 400
            
            if not isinstance(speed, (int, float)) or speed <= 0:
                return jsonify({"error": "Speed must be a positive number"}), 400
            
            # Generate audio
            audio_waveform = tts_server.text_to_audio(text, voice, speed=speed)
            audio_16k = tts_server.resample_to_16khz(audio_waveform)
            duration = tts_server.get_audio_duration(audio_16k, sample_rate=16000)
            
            response_data = {
                "success": True,
                "text": text,
                "voice": voice,
                "voice_description": tts_server.voice_options[voice],
                "speed": speed,
                "duration": round(duration, 2),
                "sample_rate": 16000
            }
            
            if return_audio:
                # Convert audio to base64
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    audio_16k.unsqueeze(0),
                    sample_rate=16000,
                    format="wav"
                )
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                response_data["audio_data"] = audio_base64
            
            return jsonify(response_data)
            
        except Exception as e:
            console.log(f"[red]Error in TTS processing: {e}[/red]")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/tts/file', methods=['POST'])
    def text_to_speech_file():
        """
        Convert text to speech and return audio file.
        
        Expected JSON payload:
        {
            "text": "Hello world",
            "voice": "af_heart",  # optional, random if not specified
            "speed": 1.0          # optional, default 1.0
        }
        """
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({"error": "Missing 'text' field"}), 400
            
            text = data['text']
            voice = data.get('voice', tts_server.get_random_voice())
            speed = data.get('speed', 1.0)
            
            # Generate audio
            audio_waveform = tts_server.text_to_speech(text, voice, speed=speed)
            audio_16k = tts_server.resample_to_16khz(audio_waveform)
            
            # Create in-memory file
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                audio_16k.unsqueeze(0),
                sample_rate=16000,
                format="wav"
            )
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'tts_{voice}_{hash(text) % 10000}.wav'
            )
            
        except Exception as e:
            console.log(f"[red]Error in TTS file processing: {e}[/red]")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch', methods=['POST'])
    def batch_process():
        """
        Process multiple texts in batch.
        
        Expected JSON payload:
        {
            "texts": ["Hello", "World"],
            "voice": "af_heart",  # optional, random if not specified
            "speed": 1.0,         # optional, default 1.0
            "return_audio": false # optional, return base64 audio data for each
        }
        """
        try:
            data = request.get_json()
            
            if not data or 'texts' not in data:
                return jsonify({"error": "Missing 'texts' field"}), 400
            
            texts = data['texts']
            voice = data.get('voice', tts_server.get_random_voice())
            speed = data.get('speed', 1.0)
            return_audio = data.get('return_audio', False)
            
            if not isinstance(texts, list):
                return jsonify({"error": "texts must be a list"}), 400
            
            results = []
            
            for i, text in enumerate(texts):
                try:
                    # Generate audio
                    audio_waveform = tts_server.text_to_audio(text, voice, speed=speed)
                    audio_16k = tts_server.resample_to_16khz(audio_waveform)
                    duration = tts_server.get_audio_duration(audio_16k, sample_rate=16000)
                    
                    result = {
                        "index": i,
                        "text": text,
                        "voice": voice,
                        "duration": round(duration, 2),
                        "success": True
                    }
                    
                    if return_audio:
                        # Convert audio to base64
                        buffer = io.BytesIO()
                        torchaudio.save(
                            buffer,
                            audio_16k.unsqueeze(0),
                            sample_rate=16000,
                            format="wav"
                        )
                        buffer.seek(0)
                        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                        result["audio_data"] = audio_base64
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "index": i,
                        "text": text,
                        "success": False,
                        "error": str(e)
                    })
            
            return jsonify({
                "success": True,
                "processed": len(results),
                "results": results
            })
            
        except Exception as e:
            console.log(f"[red]Error in batch processing: {e}[/red]")
            return jsonify({"error": str(e)}), 500
    
    return app


def main():
    """Main function to start the server."""
    # Voice options
    voice_options = {
        'af_heart': '🇺🇸 🚺 Heart ❤️',
        'af_bella': '🇺🇸 🚺 Bella 🔥', 
        'af_nicole': '🇺🇸 🚺 Nicole 🎧',
        'af_sarah': '🇺🇸 🚺 Sarah',
        'af_sky': '🇺🇸 🚺 Sky',
        'am_michael': '🇺🇸 🚹 Michael',
        'am_adam': '🇺🇸 🚹 Adam',
        'am_liam': '🇺🇸 🚹 Liam',
        'bf_emma': '🇬🇧 🚺 Emma',
        'bf_alice': '🇬🇧 🚺 Alice',
        'bm_george': '🇬🇧 🚹 George',
        'bm_lewis': '🇬🇧 🚹 Lewis'
    }
    
    # Initialize the global TTS server
    global tts_server
    tts_server = KokoroTTSServer(voice_options=voice_options, use_gpu=True)
    
    # Create Flask app
    app = create_app()
    
    console.log("[green]🎤 Kokoro TTS Server starting...[/green]")
    console.log("Available endpoints:")
    console.log("  GET  /health     - Health check")
    console.log("  GET  /voices     - List available voices")
    console.log("  POST /tts        - Generate speech (JSON response)")
    console.log("  POST /tts/file   - Generate speech (WAV file)")
    console.log("  POST /batch      - Batch processing")
    console.log(f"[green]Server running on http://localhost:1996[/green]")
    
    # Start the server
    app.run(host='0.0.0.0', port=1996, debug=False, threaded=True)


if __name__ == "__main__":
    main()
