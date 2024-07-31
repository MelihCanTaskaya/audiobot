import librosa
import numpy as np
from scipy.io.wavfile import write 
import torch
import os
 

def process_audio(wav_path, model, device):
    """Load and preprocess the audio file."""
    wav, sr_native = librosa.load(wav_path, sr=None, mono=True)

    if sr_native != model.h.sampling_rate:
        wav = librosa.resample(wav, sr_native, model.h.sampling_rate)
        sr = model.h.sampling_rate
    else:
        sr = sr_native
    
    wav = np.clip(wav, -1, 1)
    wav = torch.FloatTensor(wav).unsqueeze(0).to(device)
    return wav, sr

def save_waveform(wav_gen, sampling_rate, output_path):
    """Save the generated waveform to a file."""
    wav_gen_int16 = (wav_gen * MAX_WAV_VALUE).numpy().astype('int16')
    
    if not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")

    if wav_gen_int16.ndim not in [1, 2]:
        raise ValueError(f"Invalid waveform shape: {wav_gen_int16.shape}")

    try:
        if wav_gen_int16.ndim == 1:
            write(output_path, sampling_rate, wav_gen_int16)
        elif wav_gen_int16.ndim == 2:
            if wav_gen_int16.shape[1] == 2:
                write(output_path, sampling_rate, wav_gen_int16)
            else:
                raise ValueError(f"Unsupported number of channels: {wav_gen_int16.shape[1]}")
    except Exception as e:
        print(f"Failed to save the audio file: {e}")

def main(wav_path, output_directory, output_file, model_name='nvidia/bigvgan_base_24khz_100band', use_cuda_kernel=False):
    """Main function to orchestrate audio processing and saving."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, output_file)
    
    try:
        model = load_model(model_name, use_cuda_kernel)
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
    wav_path = r'C:\Users\melih\OneDrive\Desktop\Files\Music\test_audio.wav'
    output_directory = r'C:\Users\melih\OneDrive\Desktop\Files\Music'
    output_file = 'generated_audio.wav'
    
    main(wav_path, output_directory, output_file)
