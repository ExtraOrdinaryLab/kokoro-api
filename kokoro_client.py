import json
import time
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional

from rich.progress import track
from rich.console import Console

console = Console()


class KokoroTTSClient:
    """
    Client for interacting with Kokoro TTS Server.
    """
    def __init__(self, server_url: str = "http://localhost:1996"):
        """
        Initialize the TTS client.
        
        Args:
            server_url (str): URL of the TTS server
        """
        self.server_url = server_url.rstrip('/')
        
    def health_check(self) -> Dict:
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {e}")
    
    def get_voices(self) -> Dict:
        """Get available voices from the server."""
        try:
            response = requests.get(f"{self.server_url}/voices", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Failed to get voices: {e}")
    
    def text_to_speech(
        self, 
        text: str, 
        voice: Optional[str] = None, 
        speed: float = 1.0,
        return_audio: bool = False
    ) -> Dict:
        """
        Convert text to speech.
        
        Args:
            text (str): Text to convert
            voice (str, optional): Voice to use (random if None)
            speed (float): Speech speed multiplier
            return_audio (bool): Whether to return base64 audio data
            
        Returns:
            Dict: Response from server
        """
        payload = {
            "text": text,
            "speed": speed,
            "return_audio": return_audio
        }
        
        if voice:
            payload["voice"] = voice
        
        try:
            response = requests.post(
                f"{self.server_url}/tts", 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"TTS request failed: {e}")
    
    def text_to_speech_file(
        self, 
        text: str, 
        output_path: str,
        voice: Optional[str] = None, 
        speed: float = 1.0
    ) -> str:
        """
        Convert text to speech and save as file.
        
        Args:
            text (str): Text to convert
            output_path (str): Path to save the audio file
            voice (str, optional): Voice to use (random if None)
            speed (float): Speech speed multiplier
            
        Returns:
            str: Path to saved file
        """
        payload = {
            "text": text,
            "speed": speed
        }
        
        if voice:
            payload["voice"] = voice
        
        try:
            response = requests.post(
                f"{self.server_url}/tts/file", 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
        except Exception as e:
            raise RuntimeError(f"TTS file request failed: {e}")
    
    def batch_process(
        self, 
        texts: List[str], 
        voice: Optional[str] = None, 
        speed: float = 1.0,
        return_audio: bool = False
    ) -> Dict:
        """
        Process multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to convert
            voice (str, optional): Voice to use (random if None)
            speed (float): Speech speed multiplier
            return_audio (bool): Whether to return base64 audio data
            
        Returns:
            Dict: Batch processing results
        """
        payload = {
            "texts": texts,
            "speed": speed,
            "return_audio": return_audio
        }
        
        if voice:
            payload["voice"] = voice
        
        try:
            response = requests.post(
                f"{self.server_url}/batch", 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Batch processing failed: {e}")
    
    def save_audio_from_base64(self, audio_base64: str, output_path: str) -> str:
        """
        Save base64 audio data to file.
        
        Args:
            audio_base64 (str): Base64 encoded audio data
            output_path (str): Path to save the file
            
        Returns:
            str: Path to saved file
        """
        audio_bytes = base64.b64decode(audio_base64)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        return output_path
    
    def process_dataset_fast(
        self,
        input_jsonl_path: str,
        output_dir: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        batch_size: int = 10
    ) -> Dict:
        """
        Process entire dataset using the server for faster processing.
        
        Args:
            input_jsonl_path (str): Path to input JSONL file
            output_dir (str): Output directory for audio files
            voice (str, optional): Voice to use (random if None)
            speed (float): Speech speed multiplier
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            Dict: Processing statistics
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read input data
        console.log(f"Reading input file: {input_jsonl_path}")
        data_points = []
        with open(input_jsonl_path, 'r') as f:
            for line in f:
                data_points.append(json.loads(line))
        
        console.log(f"Processing {len(data_points)} data points...")
        
        # Process in batches
        processed_count = 0
        failed_count = 0
        manifest_data = []
        
        for i in track(range(0, len(data_points), batch_size), description="Processing batches"):
            batch = data_points[i:i+batch_size]
            batch_texts = [dp['sentence'] for dp in batch]
            
            try:
                # Process batch
                start_time = time.time()
                result = self.batch_process(
                    texts=batch_texts,
                    voice=voice,
                    speed=speed,
                    return_audio=True
                )
                batch_time = time.time() - start_time
                
                # Save audio files and create manifest entries
                for j, res in enumerate(result['results']):
                    if res['success']:
                        original_idx = i + j
                        data_point = batch[j]
                        
                        # Generate filename
                        if 'audio' in data_point and 'path' in data_point['audio']:
                            original_name = Path(data_point['audio']['path']).stem
                            output_filename = f"{original_name}_kokoro_{res['voice']}.wav"
                        else:
                            output_filename = f"sample_{original_idx:06d}_kokoro_{res['voice']}.wav"
                        
                        output_filepath = output_path / output_filename
                        
                        # Skip if file already exists
                        if output_filepath.exists():
                            console.log(f"[dim]Skipping existing file: {output_filename}[/dim]")
                            continue
                        
                        # Save audio file from base64
                        self.save_audio_from_base64(res['audio_data'], str(output_filepath))
                        
                        # Create manifest entry
                        manifest_entry = {
                            "audio": {"path": str(output_filepath)},
                            "sentence": res['text'],
                            "sentences": [],
                            "duration": res['duration'],
                            "voice": res['voice']
                        }
                        manifest_data.append(manifest_entry)
                        processed_count += 1
                    else:
                        console.log(f"[red]Failed to process: {res.get('error', 'Unknown error')}[/red]")
                        failed_count += 1
                
                console.log(f"Batch {i//batch_size + 1}: {len(batch)} items in {batch_time:.2f}s ({len(batch)/batch_time:.1f} items/s)")
                
            except Exception as e:
                console.log(f"[red]Batch processing failed: {e}[/red]")
                failed_count += len(batch)
                continue
        
        # Save manifest file
        manifest_path = output_path / "manifest.jsonl"
        with open(manifest_path, 'w') as f:
            for entry in manifest_data:
                f.write(json.dumps(entry) + '\n')
        
        stats = {
            "total_items": len(data_points),
            "processed": processed_count,
            "failed": failed_count,
            "output_dir": str(output_path),
            "manifest_file": str(manifest_path)
        }
        
        console.log(f"[green]Processing complete![/green]")
        console.log(f"Successfully processed: {processed_count}")
        console.log(f"Failed: {failed_count}")
        console.log(f"Output directory: {output_path}")
        console.log(f"Manifest file: {manifest_path}")
        
        return stats


def process_dataset_example(input_jsonl, output_directory):
    """Example of processing a dataset using the server."""
    client = KokoroTTSClient()
    
    if Path(input_jsonl).exists():
        console.log(f"[green]Processing dataset: {input_jsonl}[/green]")
        
        stats = client.process_dataset_fast(
            input_jsonl_path=input_jsonl,
            output_dir=output_directory,
            voice=None,  # Use random voices
            speed=1.0,
            batch_size=100  # Process 20 items at once
        )
        
        console.log(f"[green]Dataset processing completed![/green]")
        console.log(f"Stats: {stats}")
    else:
        console.log(f"[yellow]Dataset file not found: {input_jsonl}[/yellow]")
        console.log("[yellow]Update the path in the example to use this function[/yellow]")


if __name__ == "__main__":
    console.log("[green]🎤 Kokoro TTS Client Demo[/green]")
    
    # Run dataset processing example
    # Example dataset processing (adjust paths as needed)
    input_jsonl = '/home/jovyan/workspace/aura/corpus/audio/SAP/train.v2.jsonl'
    output_directory = '/home/jovyan/workspace/aura/corpus/audio/kokoro-sap/train'
    process_dataset_example(input_jsonl, output_directory)
