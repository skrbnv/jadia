import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .coskmeans import CosKMeans
from . import models as _m
from .audio import SampleProcessor
from .segment import Segment, Segments
from .gaussian import gaussian_filter1d
import warnings
import os

# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


modelinfo = {
    "lite": {
        "name": "model_lite",
        "checkpoint": "assets/lite.pt",
    }
}


class Jadia:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model: str = "lite",
        clustering_method: str = "kmeans",
        batch_size: int = 16,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model = Model(device=device, model=model)
        self.clustering_method = clustering_method

    def setup(
        self,
        num_voices: int,
        use_vad: bool = True,
        collapse: bool = True,
        collapse_tolerance: float = 0.5,
        adjust_tolerance: float = 0.5,
        min_interval: float = 0.1,
        spectrogram_stride: int = 10,
        spectrogram_length: float = 60,
        spectrogram_strategy: str = "pad",
        update_stride: int = 1,
        clustering_method: str = "coskmeans",
        with_confidence: bool = False,
        **kwargs,
    ):
        """
        Sets up class instance for the next audio to process

        Arguments:
            - num_voices: number of speakers
            - use_vad: use voice activity detection
            - collapse: ignore non-spoken intervals in between when merging same-speaker segments
            - collapse_tolerance: max size of non-spoken intervals to collapse, in seconds
            - adjust_tolerance: edge-wise adjust predictions, in seconds
            - min_interval: min spoken interval, in seconds
            - spectrogram_stride: sliding window stride over spectrogram
            - spectrogram_strategy: pad or tile, how to reconstruct sliding window if spectrogram is too small
            - update_stride: populate predictions by average within stride interval
            - clustering_method: currently only cosine KMeans (coskmeans)
            - kwargs: audio args, see SampleProcessor class init params
        Returns:
            None
        """
        self.num_voices = num_voices
        self.collapse = collapse
        self.collapse_tolerance = collapse_tolerance
        self.tolerance = adjust_tolerance
        self.use_vad = use_vad
        self.min_interval = min_interval
        self.audio_args = kwargs
        self.audio_args["stride"] = spectrogram_stride
        self.audio_args["slice"] = spectrogram_length
        self.audio_args["strategy"] = spectrogram_strategy
        self.update_stride = update_stride
        self.update_fn = (
            self.update_predictions_with_confidence
            if with_confidence is True
            else self.update_predictions
        )
        self.spectrogram_strategy = spectrogram_strategy
        self.clusterer = Clusterer(num_voices=num_voices, method=clustering_method)

    def load_audio(self, filename: str) -> SampleProcessor:
        """
        Loads audio, detects silent|non-spoken parts and creates spectrogram
        Arguments:
            - filename: path to audio file
        Returns:
            - SampleProcessor instance
        """
        self.filename = filename
        self.sample_processor = SampleProcessor(
            filename, use_vad=self.use_vad, **self.audio_args
        )
        return self.sample_processor

    def predict(
        self,
        num_voices: int | None = None,
        smooth: bool = True,
        fast_fit: bool = False,
    ) -> np.ndarray:
        """
        Converts loaded audio to dict of predictions,
        where prediction keys are spectrogram-wise positions
        Arguments:
            - num_voices: number of speakers
            - smooth: use gaussian filter on predictions, may increase accuracy of speaker change around silences
            - fast_fit: compute clusters based on first part of audio, don't collect all embeddings. Good for long dialogues. You may need to increase length of slice (60s by default).
        Returns:
            predictions, numpy array (spectrogram length x num_voices)
        """
        if num_voices is None:
            num_voices = self.num_voices

        # create predictions array with shape (combined_spectrogram_len x num_voices)
        self.predictions = np.zeros(
            (self.sample_processor.get_fullspectrogram_size(), num_voices)
        )

        embeddings = {}
        # for index in count spectrograms
        for index in range(self.sample_processor.count_spectrograms()):
            # create dataset made of sliding windows across that spectrogram
            dataset = DTST(
                data=self.sample_processor.get_sliding_windows(index),
                device=self.device,
            )
            loader = DataLoader(dataset, batch_size=self.batch_size)
            # compute and populate embeddings per sliding window
            embs = []
            with torch.no_grad():
                for inputs in loader:
                    be = self.model(inputs)
                    embs.append(be.cpu().numpy())
            embs = np.vstack(embs)
            # normalize embeddings (project onto unit hypersphere)
            embeddings[index] = embs / np.expand_dims(np.linalg.norm(embs, axis=1), 1)
            # if we compute centroids using first spectrogram
            if fast_fit is True:
                if index == 0:
                    assert (
                        len(embeddings[index]) >= num_voices
                    ), "Unable to process audio, number of embeddings generated less than number of voices [too little embeddings to compute clusters]. You may try setting fast_fit to False, or use vad=False for setup() or process()"
                self.update_fn(
                    indices=[index],
                    embeddings=[embeddings[index]],
                    fit=True if index == 0 else False,
                    stride=self.update_stride,
                )
                embeddings = {}
        # if we compute centroids using combined spectrogram
        if fast_fit is False:
            assert (
                sum([len(embs) for embs in embeddings.values()]) >= num_voices
            ), "Unable to process audio, number of embeddings generated less than number of voices [too little embeddings to compute clusters]. You may try using vad=False for setup() or process()"
            self.update_fn(
                indices=list(embeddings.keys()),
                embeddings=list(embeddings.values()),
                fit=True,
                stride=self.update_stride,
            )
        if smooth is True:
            self.smooth_predictions()
        return self.predictions

    def update_predictions(
        self,
        indices: list[int],
        embeddings: list[np.ndarray],
        fit: bool = False,
        stride: int = 1,
    ):
        """
        Updates predictions based on embeddings
        Arguments:
            - indices: list of spectrograms to update predictions against
            - embeddings: according embeddings
            - fit: fit+predict or just predict without fitting
            - stride: average using stride for predictions
        Returns:
            None
        """
        if fit is True:
            labels = self.clusterer.fit_predict(np.vstack(embeddings))
            if self.num_voices > len(np.unique(labels)):
                warnings.warn(
                    "Number of clusters is less than number of speakers. Try setting fast_fit to false if you're using it."
                )
            lengths = [emb.shape[0] for emb in embeddings]
            starts = np.cumsum([0] + lengths[:-1])
            predictions = np.split(
                labels, [start + length for start, length in zip(starts, lengths)]
            )
        else:
            predictions = [self.clusterer.predict(np.vstack(embeddings))]

        for seq_index, preds_per_seq_index in zip(indices, predictions):
            windows_nums, matching_weights = (
                self.sample_processor.sliding_window_indices(seq_index)
            )
            for global_interval, local_interval in self.sample_processor.iter_intervals(
                seq_index=seq_index
            ):
                embedding_indices = windows_nums[
                    local_interval[0] : local_interval[1] + 1
                ]
                weights = matching_weights[local_interval[0] : local_interval[1] + 1]
                for i in range(0, local_interval[1] - local_interval[0] + 1, stride):
                    t = i + global_interval[0]
                    if len(embedding_indices[i]) == 0:
                        # we can have no windows for some positions due to large stride
                        continue
                    preds = preds_per_seq_index[embedding_indices[i]]
                    sw = sum(weights[i])
                    for s in np.unique(preds):
                        self.predictions[t : t + stride, s] = (
                            np.sum(weights[i][preds == s]) / sw
                        )

    def update_predictions_with_confidence(
        self,
        indices: list[int],
        embeddings: list[np.ndarray],
        fit: bool = False,
        stride: int = 1,
    ):
        """
        Updates predictions based on embeddings
        Arguments:
            - indices: list of spectrograms to update predictions against
            - embeddings: according embeddings
            - fit: fit+predict or just predict without fitting
            - stride: average using stride for predictions
        Returns:
            None
        """
        if fit is True:
            # compute centroids and return predictions with confidence,
            # - combined across all spectrograms if multiple
            clusterer_estimates = self.clusterer.fit_predict_with_confidence(
                np.vstack(embeddings)
            )
            if self.num_voices > clusterer_estimates.shape[1]:
                warnings.warn(
                    "Number of clusters is less than number of speakers. Try setting fast_fit to false if you're using it."
                )
            # get number of embeddings per spectrogram
            lengths = [emb.shape[0] for emb in embeddings]
            starts = np.cumsum([0] + lengths[:-1])
            # split estimates with confidence back into groups per spectrogram
            estimates = np.split(
                clusterer_estimates,
                [start + length for start, length in zip(starts, lengths)],
            )
        else:
            estimates = [self.clusterer.predict_with_confidence(np.vstack(embeddings))]

        # for each spectgrogram get all according estimates
        for spg_index, estimates_per_spg_index in zip(indices, estimates):
            # get ids of sliding windows and matching weights
            # per each position within spectrogram
            windows_nums, matching_weights = (
                self.sample_processor.sliding_window_indices(spg_index)
            )
            # for each interval withing spectrogram we get position of such interval
            # but on a global audio sample (in spectrogram steps, not in pure audio samples)
            for global_interval, local_interval in self.sample_processor.iter_intervals(
                seq_index=spg_index
            ):
                # get indices of embeddings/estimates per position + according weights
                embedding_indices = windows_nums[
                    local_interval[0] : local_interval[1] + 1
                ]
                weights = matching_weights[local_interval[0] : local_interval[1] + 1]
                # for each strided position within local interval
                for i in range(0, local_interval[1] - local_interval[0] + 1, stride):
                    # compute according position within global interval
                    t = i + global_interval[0]
                    # we can have no sliding windows for some positions due to large stride
                    if len(embedding_indices[i]) == 0:
                        continue
                    # get model estimates for each sliding window per position
                    ests = estimates_per_spg_index[embedding_indices[i]]
                    # apply weights assigned to each sliding widow to estimates
                    mm = np.expand_dims(weights[i], axis=1) * ests
                    self.predictions[t : t + stride, :] = np.expand_dims(
                        np.sum(mm, axis=0) / np.sum(mm), axis=0
                    )

    def process(
        self,
        filename: str,
        num_voices: int,
        use_vad: bool = True,
        fast_fit: bool = False,
        **kwargs,
    ) -> Segments:
        """
        Simplified combined function to process audio from loading to predictions
        Arguments (see Sample class constructor for default values):
            - filename:
            - num_voices: number of speakers
            - use_vad: use voice activity detection
            - fast_fit: fit using part of record (60 seconds by default)
            + kwargs for SampleProcessor
        Returns:
            instance of Segments
        """
        self.setup(num_voices=num_voices, use_vad=use_vad, **kwargs)
        self.sample_processor = self.load_audio(filename=filename)
        predictions = self.predict(num_voices=num_voices, fast_fit=fast_fit)
        segments = self.predictions_to_segments(predictions)
        return segments

    def follow(self, filename: str) -> list:
        """
        Function similar to process, but it uses previous clusterer and audio settings.
        Arguments:
            - filename: str
        Returns:
            instance of Segments
        """
        self.sample_processor = self.load_audio(filename=filename)
        predictions = self.predict(num_voices=self.num_voices, smooth=True)
        segments = self.predictions_to_segments(predictions, tolerance=self.tolerance)
        return segments

    def smooth_predictions(self):
        """
        Smoothes predictions, helps with incorrect speaker change predictions near silent parts
        Arguments:
            none
        Returns:
            none
        """
        # Parameters for the Gaussian kernel
        for s in range(len(self.predictions[0])):
            arr = gaussian_filter1d(
                self.predictions[:, s], sigma=15
            )  # apply_convolution([el[s] for el in predictions.values()], kernel)
            arr[~self.sample_processor.mask] = 0
            self.predictions[:, s] = arr

    def predictions_to_segments(
        self,
        predictions: np.ndarray,
    ):
        """
        Converts dictionary of predictions to Segments instance
        Arguments:
            - predictions: dict of predictions from previous computations
        Returns:
            instance of Segments
        """
        assert hasattr(
            self, "collapse"
        ), "Need to run setup() before calling this method"
        best = np.argmax(predictions, axis=1)
        best[np.max(predictions, axis=1) == 0] = -1
        segments = Segments()
        change_indices = np.where(np.diff(best) != 0)[0] + 1
        change_indices = np.concatenate(([0], change_indices, [len(best)]))
        for start, end in zip(change_indices[:-1], change_indices[1:]):
            segments.append(
                Segment(
                    start=self.sample_processor.step_to_time(start),
                    end=self.sample_processor.step_to_time(end - 1),
                    speaker=(best[start] if best[start] > -1 else None),
                )
            )

        if self.tolerance > 0:
            # adjust segments:
            # if interval close to silence (but not cigar), followed by different speaker,
            # shift edge towards silence
            edges = np.array(
                [
                    self.sample_processor.step_to_time(el)
                    for iv in self.sample_processor.get_intervals()
                    for el in iv
                ]
            )
            segments.adjust(edges=edges, tolerance=self.tolerance)
        # drop too short segments
        segments.drop_short(self.min_interval)
        # merge segments
        segments.merge(collapse=self.collapse, tolerance=self.collapse_tolerance)
        # remove silent segments
        segments.drop_silent()

        return segments

    def order_speakers(self):
        """
        Modifies speakers' order, makes them appear in same order as regular enumeration
        """
        mv = np.argmax(self.predictions, axis=1)[
            np.argwhere(np.max(self.predictions, axis=1) > 0)
        ]
        _, ixs = np.unique(mv, return_index=True)
        self.predictions = self.predictions[:, np.argsort(ixs)]


class Model:
    def __init__(
        self, device: torch.device = torch.device("cpu"), model: str = "lite"
    ) -> None:
        self.device = device
        if model not in modelinfo.keys():
            raise Exception("Unkown model name")
        model_call = getattr(_m, modelinfo[model]["name"])
        self.model = model_call().to(device)
        self.model.eval()
        cp = os.path.dirname(__file__)
        with open(os.path.join(cp, modelinfo[model]["checkpoint"]), "rb") as f:
            checkpoint = torch.load(f, weights_only=True)
        self.model.load_state_dict(checkpoint)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class DTST(Dataset):
    def __init__(
        self,
        data: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).to(dtype=dtype)
        else:
            self.data = data.to(dtype=dtype)
        self.dtype = dtype
        self.device = device

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index].to(device=self.device)


class Clusterer:
    def __init__(
        self, num_voices: int | None = None, method: str = "coskmeans"
    ) -> None:
        self.method = method.lower()
        """
        if self.method == "kmeans":
            assert (
                num_voices is not None
            ), "With K-means you need to specify number of speakers"
            self.engine = KMeans(n_clusters=num_voices)
        """
        if self.method == "coskmeans":
            assert (
                num_voices is not None
            ), "With CosKmeans you need to specify number of speakers"
            self.engine = CosKMeans(n_clusters=num_voices)
            """
            elif self.method == "agg":
                assert (
                    num_voices is not None
                ), "With Agglomerative clustering you need to specify number of speakers"
                self.engine = AgglomerativeClustering(
                    n_clusters=num_voices, metric="cosine", linkage="average"
                )
            elif self.method == "dbscan":
                assert (
                    num_voices is not None
                ), "With Agglomerative clustering you need to specify number of speakers"
                self.engine = DBSCAN(eps=1e-2, metric="cosine")
            """
        else:
            raise Exception(
                "Unknown clustering method, only kmeans, agg or dbscan are supported"
            )

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self.engine.fit_predict(embeddings)
        # if self.method == "kmeans" or self.method == "coskmeans":
        self.centroids = self.engine.cluster_centers_
        """
        elif self.method == "agg" or self.method == "dbscan":
            self.__refdata = embeddings
            self.__reflabels = labels
        """
        return labels

    def fit_predict_with_confidence(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.method == "coskmeans", "Only CosKMeans supported with this method"
        return self.engine.fit_predict_with_confidence(embeddings)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.engine.predict(embeddings)

    def predict_with_confidence(self, embeddings: np.ndarray) -> np.ndarray:
        return self.engine.predict_with_confidence(embeddings)
