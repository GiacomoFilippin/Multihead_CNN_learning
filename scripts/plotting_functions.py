import matplotlib.pyplot as plt
import librosa.display

def make_stft_spectrogram_plot(spectrogram, sr, title="STFT Power Spectrogram", hop_length=512, n_fft=2048, fmax=8192):
    plt.figure(figsize=(12, 6))
    # Crea il plot dello spettrogramma
    librosa.display.specshow(spectrogram, 
                            sr=sr,
                            x_axis='time',
                            y_axis="linear",
                            hop_length=hop_length,
                            n_fft=n_fft,
                            fmax=sr//2,
                            cmap='viridis')

    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Power Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 5000)  # Regola questo in base all'interesse frequenziale
    plt.tight_layout()
    return plt

def make_mel_spectrogram_plot(spectrogram, sr, title="Mel Power Spectrogram", fmax=8192):
    plt.figure(figsize=(12, 6))
    # Crea il plot dello spettrogramma
    librosa.display.specshow(spectrogram, 
                            sr=sr,
                            x_axis='time',
                            y_axis='mel',
                            fmax=fmax,
                            cmap='viridis')

    # Aggiungi barra del colore e labels
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    
    return plt