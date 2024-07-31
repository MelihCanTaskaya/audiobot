import sys
sys.path.append('C:/Users/melih/BigVGAN')
import os
import torch
import bigvgan
import librosa
import numpy as np
from scipy.io.wavfile import write
from meldataset import get_mel_spectrogram, MAX_WAV_VALUE

def load_model(model_name='nvidia/bigvgan_base_24khz_100band', use_cuda_kernel=False):
    """Load and prepare the BigVGAN model."""
    model = bigvgan.BigVGAN.from_pretrained(model_name, use_cuda_kernel=use_cuda_kernel)
    model.remove_weight_norm()
    return model.eval()

def process_audio(wav_path, model, device):
    """Load and preprocess the audio file."""
    wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)
    wav = np.clip(wav, -1, 1)
    wav = torch.FloatTensor(wav).unsqueeze(0).to(device)
    return wav, sr

def generate_waveform(model, wav, device):
    """Generate a waveform from the given audio tensor."""
    with torch.no_grad():
        mel = get_mel_spectrogram(wav, model.h).to(device)
        print(f"Mel spectrogram shape: {mel.shape}")
        wav_gen = model(mel)
        print(f"Generated waveform shape (before squeeze): {wav_gen.shape}")
    wav_gen = wav_gen.squeeze(0).squeeze(0).cpu()  # Correctly squeeze the tensor to get the shape [T_time]
    print(f"Generated waveform shape (after squeeze): {wav_gen.shape}")
    return wav_gen

def save_waveform(wav_gen, sampling_rate, output_path):
    """Save the generated waveform to a file."""
    wav_gen_int16 = (wav_gen * MAX_WAV_VALUE).numpy().astype('int16')
    print(f"Generated waveform shape (int16): {wav_gen_int16.shape}")

    # Validate output path and sampling rate
    if not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")

    if wav_gen_int16.ndim != 1:
        raise ValueError(f"Invalid waveform shape: {wav_gen_int16.shape}")

    try:
        write(output_path, sampling_rate, wav_gen_int16)
        print(f"Generated audio saved to {output_path}")
    except Exception as e:
        print(f"Failed to save the audio file: {e}")

def main(wav_path, output_directory, output_file, model_name='nvidia/bigvgan_base_24khz_100band', use_cuda_kernel=False):
    """Main function to orchestrate audio processing and saving."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, output_file)
    
    try:
        model = load_model(model_name, use_cuda_kernel).to(device)
        wav, sr = process_audio(wav_path, model, device)
        wav_gen = generate_waveform(model, wav, device)
        save_waveform(wav_gen, sr, output_path)
    except FileNotFoundError:
        print(f"Error: File not found - {wav_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    wav_path = r"C:\Users\melih\OneDrive\Desktop\Files\Music\Winter Haze 61.5bpm @cobalii.wav"
    output_directory = r'C:\Users\melih\OneDrive\Desktop\Files\Music'
    output_file = 'generated_audio.wav'
    
    main(wav_path, output_directory, output_file)
