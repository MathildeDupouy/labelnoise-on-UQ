import numpy as np
import torch

from src.Utils.spectrogram import estimate_inf_sup_db, stft_signal


class Constructor:
    """Abstract class for constructors."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, item):
        raise NotImplementedError

    def construct_from_path(self, path):
        raise NotImplementedError

class ECG_Audio_Constructor(Constructor):
    """Constructor for PTB audio."""

    def __init__(self):
        super().__init__()

    def __call__(self, signal):
        return torch.unsqueeze(torch.Tensor(signal), 0)


class ECG_Spectrogram_Constructor(Constructor):
    """Constructor for PTB spectrogram."""

    def __init__(self, n_fft=32, n_overlap=4, window="blackman", sample_rate = 125):
        """Initialize Spectrogram constructor.
        This constructor computes the spectrogram of the input signal using STFT.
        The spectrogram is then converted to a log scale and filtered to remove
        values outside the estimated min and max dB values.
        The spectrogram is then converted to a PyTorch tensor.

        Args:
            n_fft (int, optional): _description_. Defaults to 32.
            n_overlap (int, optional): _description_. Defaults to 4.
            window (str, optional): _description_. Defaults to 'blackman'.
            sample_rate (int, optional): _description_. Defaults to 125.
        """
        super().__init__()

        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.window = window
        self.sample_rate = sample_rate  # Hz

    def __call__(self, signal):
        stft_waveform, f, t = stft_signal(
            signal,
            self.n_fft,
            sample_rate=self.sample_rate,
            noverlap=self.n_overlap,
            window=self.window,
        )
        spectrogram = np.absolute(stft_waveform) ** 2
        log_spec = 20 * np.log10(spectrogram)
        # Estimating the min and max db values of the spectrogram in log scale for the colobar
        x, y = 0, 9
        min_db, max_db = estimate_inf_sup_db(spectrogram, x, y)
        # Filtering the spectrogram
        log_spec = 20 * np.log10(spectrogram)
        log_spec[log_spec < min_db] = min_db
        log_spec[log_spec > max_db] = max_db
        feature = log_spec
        
        return torch.unsqueeze(torch.from_numpy(feature.copy()).float(), 0)

class EEG_Audio_Constructor(Constructor):
    """Constructor for ESR audio."""

    def __init__(self):
        super().__init__()

    def __call__(self, signal):
        return torch.unsqueeze(torch.Tensor(signal), 0)
    def __init__(self):
        super().__init__()

    def __call__(self, signal):
        return torch.unsqueeze(torch.Tensor(signal), 0)

class EEG_Spectrogram_Constructor(Constructor):
    """Constructor for ESR time-frequency representation."""

    def __init__(self, n_fft=32, n_overlap=4, window="blackman", sample_rate = 178):
        """Initialize Spectrogram constructor.
        This constructor computes the spectrogram of the input signal using STFT.
        The spectrogram is then converted to a log scale and filtered to remove
        values outside the estimated min and max dB values.
        The spectrogram is then converted to a PyTorch tensor.

        Args:
            n_fft (int, optional): _description_. Defaults to 32.
            n_overlap (int, optional): _description_. Defaults to 4.
            window (str, optional): _description_. Defaults to 'blackman'.
            sample_rate (int, optional): _description_. Defaults to 178.
        """
        super().__init__()

        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.window = window
        self.sample_rate = sample_rate  # Hz

    def __call__(self, signal):
        stft_waveform, f, t = stft_signal(
            signal,
            self.n_fft,
            sample_rate=self.sample_rate,
            noverlap=self.n_overlap,
            window=self.window,
        )
        spectrogram = np.absolute(stft_waveform) ** 2
        log_spec = 20 * np.log10(spectrogram)
        # Estimating the min and max db values of the spectrogram in log scale for the colobar
        x, y = 0, 9
        min_db, max_db = estimate_inf_sup_db(spectrogram, x, y)
        # Filtering the spectrogram
        log_spec = 20 * np.log10(spectrogram)
        log_spec[log_spec < min_db] = min_db
        log_spec[log_spec > max_db] = max_db
        feature = log_spec
        
        return torch.unsqueeze(torch.from_numpy(feature.copy()).float(), 0)


#----- Test -----#
if __name__ == "__main__":
    # Test EEG constructors
    print("---- Testing EEG constructors ----")
    signal = np.random.rand(32, 178 * 10)  # 32 channels, 10 seconds at 178 Hz
    audio_constructor = EEG_Audio_Constructor()
    spectro_constructor = EEG_Spectrogram_Constructor()
    audio_feature = audio_constructor(signal)
    spectro_feature = spectro_constructor(signal)
    print(f"Audio feature shape: {audio_feature.shape}")
    print(f"Spectrogram feature shape: {spectro_feature.shape}")
