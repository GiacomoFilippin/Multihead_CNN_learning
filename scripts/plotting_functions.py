import matplotlib.pyplot as plt
import librosa.display

def make_stft_spectrogram_plot(spectrogram,
                               sr,
                               title="STFT Power Spectrogram",
                               hop_length=512,
                               n_fft=2048,
                               fmax=5000):
    plt.figure(figsize=(12, 6))
    # usa y_axis='hz' e passa fmax qui
    librosa.display.specshow(spectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             n_fft=n_fft,
                             x_axis='time',
                             y_axis='hz',
                             fmax=fmax,
                             cmap='viridis')

    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # in teoria non serve, ma assicura il ritaglio
    plt.ylim(0, fmax)
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
    plt.ylim(0, fmax)
    plt.tight_layout()
    
    return plt