import numpy as np
import torch
import librosa
import soxr
import os
from soundfile import read as read_audio
from subprocess import CalledProcessError, run
from silero_vad import get_speech_timestamps, load_silero_vad
from math import ceil


class SampleProcessor:
    def __init__(
        self,
        filename: str,
        use_vad: bool = True,
        sr: int = 16000,
        mono: bool = True,
        top_db: int = 30,
        hop: int = 160,
        n_mels: int = 80,
        n_fft: int = 400,
        window_size: int = 192,
        stride: int = 1,
        slice: int = 60,
        strategy: str = "pad",  # pad or tile
    ):
        self.ratio = sr // hop
        assert self.ratio == sr / hop, "Sample rate should be divisible by hop length"
        self.use_vad = use_vad
        self.sr = sr
        self.mono = mono
        self.top_db = top_db
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.window_size = window_size
        self.stride = stride
        self.strategy = strategy
        self.slice_length = slice * sr // hop
        self.weights = self.compute_weights()
        # Load audio in seconds
        self.audio = self.load_audio(filename)
        self.__spectrogram_full = self.log_mel_spectrogram(audio=self.audio)
        # Get intervals in spectrogram steps
        if use_vad is True:
            self.vad = load_silero_vad()
            intervals = self.detect_voice(self.audio)
            nonsilent = self.get_nonsilent_intervals(audio=self.audio)
            self.__intervals = self.intersect_intervals(intervals, nonsilent)
        else:
            self.__intervals = self.get_nonsilent_intervals(audio=self.audio)
        # Generate mask of non-silent/silent parts of spectrogram[!] in spectrogram steps
        self.mask = self.compute_mask(self.__intervals)
        self.slice_spectrograms()

    def compute_weights(self):
        half = self.window_size // 2
        output = (np.arange(self.window_size) - half) / half
        return np.exp(-(output**2))

    def time_to_step(self, t: float):
        return int(t * self.ratio)

    def step_to_time(self, step: int, mode: str = "floor"):
        if mode == "floor":
            return step / self.ratio
        elif mode == "ceil":
            return (step + 1) / self.ratio
        else:
            raise Exception("Unknown mode")

    def detect_voice(self, audio: np.ndarray):
        raw_intervals = get_speech_timestamps(audio.numpy(), self.vad)
        intervals = np.array(
            [
                [
                    np.floor(el["start"] / self.hop),
                    np.minimum(
                        np.ceil(el["end"] / self.hop),
                        self.__spectrogram_full.size(1) - 1,
                    ),
                ]
                for el in raw_intervals
            ]
        ).astype(int)
        return intervals

    def slice_spectrograms(self):
        # Split into smaller spectrograms (silence removed), prefer to divide using silence
        lengths = [el[1] - el[0] + 1 for el in self.__intervals]
        self.__sequences = []
        summae, seq = 0, []
        for i in range(len(lengths)):
            summae += lengths[i]
            seq += [i]
            if summae >= self.slice_length:
                self.__sequences.append(seq)
                summae, seq = 0, []
        else:
            if summae > 0:
                self.__sequences.append(seq)
        assert np.sum(np.arange(len(lengths))) == sum(
            [sum(el) for el in self.__sequences]
        ), "Missing elements in sequences"
        # Compute sums
        grouped_sums = [
            np.cumsum([el[1] - el[0] + 1 for el in self.__intervals[seq]])
            for seq in self.__sequences
        ]
        self.__sums = np.array([el for seq in grouped_sums for el in seq])

        self.__spectrograms = [
            self.__spectrogram_full[:, self.compute_mask(self.__intervals[seq])]
            for seq in self.__sequences
        ]
        for i, spg in enumerate(self.__spectrograms):
            if spg.shape[1] < self.window_size:
                if self.strategy == "pad":
                    self.__spectrograms[i] = self.pad_spectrogram(spg)
                elif self.strategy == "tile":
                    self.__spectrograms[i] = self.tile_spectrogram(spg)
                else:
                    raise Exception(f"Unknown spectrogram strategy: {self.strategy}")

    def get_spg_pos(self, pos: int):
        """
        returns index of spectrogram and position in spectrogram steps by position in audio
        returns None if position is located in silent part
        """
        condition = (self.__intervals[:, 0] <= pos) & (self.__intervals[:, 1] >= pos)
        index = np.where(condition)[0]
        if len(index) == 0:
            return None
        i = index[0]
        spgpos = self.__sums[i] - (self.__intervals[i, 1] - pos) - 1
        return [i in el for el in self.__sequences].index(True), spgpos

    def get_pos_by_spg_index(self, index: int):
        return (
            self.__intervals[self.__sequences[index][0]][0],
            self.__intervals[self.__sequences[index][-1]][1],
        )

    def get_spectrogram(self, index: int):
        """
        returns spectrogram, silence removed
        """
        return self.__spectrograms[index]

    def get_sliding_windows(self, index: int):
        return (
            self.get_spectrogram(index)
            .unfold(1, self.window_size, self.stride)
            .transpose(1, 0)
        )

    def get_windows_by_spgpos(self, index: int, pos: int):
        """
        returns windows by position in spectrogram steps
        if they were generated altogether per whole spectrogram
        """

        starts = np.arange(
            0, self.__spectrograms[index].size(1) - self.window_size + 1, self.stride
        )
        indices = np.where((starts <= pos) & (pos < (starts + self.window_size)))[0]
        weights = self.weights[pos - starts[indices]]
        return indices, weights

    def sliding_window_indices(self, index: int):
        """
        returns windows by interval in spectrogram steps
        if they were generated altogether per whole spectrogram
        """
        arr_len = self.get_spectrogram(index).size(1)
        # Calculate the total number of sliding windows
        num_windows = (arr_len - self.window_size) // self.stride + 1

        # Create an array of start indices for each sliding window
        start_indices = np.arange(num_windows) * self.stride

        # Create a matrix where each row is a range of indices for the sliding window
        windows = start_indices[:, None] + np.arange(self.window_size)

        # Create an array to store the indices for each element in the original array
        element_to_windows = np.full((arr_len, num_windows), False)

        # For each sliding window, mark the corresponding elements in the array
        for i in range(self.window_size):
            element_to_windows[windows[:, i], np.arange(num_windows)] = True

        # Convert the boolean matrix to a list of indices
        indices = [np.where(element_to_windows[i])[0] for i in range(arr_len)]
        # positions = [np.full_like(ix, i) - ix for i, ix in enumerate(indices)]
        weights = [
            self.weights[np.full_like(ix, i) - ix * self.stride]
            for i, ix in enumerate(indices)
        ]
        return indices, weights

    def load_audio(self, filename: str) -> torch.Tensor:
        """
        Soundfile/Librosa-based audio loader
        """
        try:
            data, sample_rate = read_audio(filename)
        except:
            # fallback to whisper's ffmpeg loader
            data = self.load_audio_whisper(filename, self.sr)
            sample_rate = self.sr
        if self.mono is True and len(data.shape) > 1:
            data = data.mean(axis=1)
        if self.sr != sample_rate:
            data = np.apply_along_axis(
                soxr.resample,
                axis=-1,
                arr=data,
                in_rate=sample_rate,
                out_rate=self.sr,
                quality="soxr_hq",
            )
        assert (
            len(data) > self.window_size * self.hop
        ), "File is too short, expected at least %.2fs" % (
            self.window_size * self.hop / self.sr
        )

        return torch.from_numpy(data)

    @staticmethod
    def load_audio_whisper(file: str, sr: int = 16000):
        """
        Open an audio file and read as mono waveform, resampling as necessary, borrowed from whisper

        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        # This launches a subprocess to decode audio while down-mixing
        # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
        # fmt: off
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError("Failed to load audio:" + str(e.stderr.decode())) from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 is supported right now

        Returns
        -------
        torch.Tensor, shape = (n_mels, n_frames)
            A Tensor that contains the Mel spectrogram
        """

        def mel_filters(device, n_mels: int) -> torch.Tensor:
            """
            load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
            Allows decoupling librosa dependency; saved using:

                np.savez_compressed(
                    "mel_filters.npz",
                    mel_64=librosa.filters.mel(sr=16000, n_fft=400, n_mels=64),
                    mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                    mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
                )
            """
            assert n_mels in {64, 80, 128}, "Unsupported n_mels:" + str(n_mels)

            filters_path = os.path.join(
                os.path.dirname(__file__), "assets", "mel_filters.npz"
            )
            with np.load(filters_path, allow_pickle=False) as f:
                return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

        window = torch.hann_window(self.n_fft).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters(audio.device, self.n_mels).to(audio.dtype)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def pad_spectrogram(self, array: torch.Tensor) -> torch.Tensor:
        assert array.shape[0] == self.n_mels, "Incorrect number of mels"
        if array.shape[1] < self.window_size:
            diff = self.window_size - array.shape[1]
            pw = diff // 2
            array = torch.nn.functional.pad(array, (pw, diff - pw), "constant", 0)
        return array

    def tile_spectrogram(self, array: torch.Tensor) -> torch.Tensor:
        assert array.shape[0] == self.n_mels, "Incorrect number of mels"
        if array.shape[1] < self.window_size:
            output = array.repeat((1, ceil(self.window_size / array.shape[1])))[
                :, : self.window_size
            ]
            return output
        return array

    def compute_mask(
        self,
        intervals: np.ndarray,
    ):
        """
        Computes boolean mask of silence measured in spectrogram steps,
        True for non-silent, False for silent parts
        """
        mask = np.zeros(self.__spectrogram_full.size(1), dtype=bool)
        mask[np.concatenate([np.arange(el[0], el[1] + 1) for el in intervals])] = 1
        return mask

    def get_nonsilent_intervals(
        self, audio: torch.Tensor, tolerance: int = 10
    ) -> np.ndarray:
        """
        Returns non-silent intervals in spectrogram steps based on original audio
        """
        intervals = librosa.effects.split(y=audio.numpy(), top_db=self.top_db)
        spgint = np.zeros_like(intervals)
        spgint[:, 0] = np.floor(intervals[:, 0] / self.hop)
        spgint[:, 1] = np.minimum(
            np.ceil(intervals[:, 1] / self.hop), self.__spectrogram_full.size(1) - 1
        )
        # remove too short silent pauses
        removables = []
        for i in reversed(range(1, len(spgint))):
            if (spgint[i, 0] - spgint[i - 1, 1]) < tolerance:
                spgint[i - 1, 1] = spgint[i, 1]
                removables.append(i)
        return np.delete(spgint, removables, 0)

    def intersect_intervals(self, voice: np.ndarray, nonsilent: np.ndarray):
        """
        In addition to VAD also removes parts considered 'silent' (30db difference from peak)
        """
        mask = self.compute_mask(voice)
        silence_mask = ~self.compute_mask(nonsilent)
        mask[silence_mask] = False
        return self.mask_to_intervals(mask)

    def mask_to_intervals(self, mask: np.ndarray) -> np.ndarray:
        change_points = np.diff(mask.astype(int), prepend=0, append=0)
        start_points = np.where(change_points == 1)[0]
        end_points = np.where(change_points == -1)[0] - 1
        intervals = list(zip(start_points, end_points))
        return np.array(intervals)

    def get_fullspectrogram_size(self):
        return self.__spectrogram_full.size(1)

    def get_spectrogram(self, index: int):
        return self.__spectrograms[index]

    def count_spectrograms(self):
        return len(self.__spectrograms)

    def get_intervals(self):
        return self.__intervals

    def iter_intervals(self, seq_index):
        """
        Generates iterator over spectrogram[seq_index].
        Returns intervals of full spectrogram, related to slice spectrogram[index] plus
        locations of intervals within spectrogram slice
        """
        indices = self.__sequences[seq_index]

        def ci(intervals: np.ndarray, indices: list, sums: list):
            s = np.concatenate(([0], sums[indices]))
            for num, i in enumerate(indices):
                yield intervals[i], np.array([s[num], s[num + 1] - 1])

        return ci(intervals=self.__intervals, indices=indices, sums=self.__sums)
