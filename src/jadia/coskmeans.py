import numpy as np


class CosKMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 300,
        init_count: int = 50,
        tol: float = 1e-4,
    ):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iters = max_iters  # Maximum number of iterations
        self.tol = tol  # Tolerance to declare convergence
        self.init_count = init_count

    def initialize_centers(self, X: np.ndarray) -> None:
        """
        Guessing cluster centers
        Arguments:
            - X: embeddings (num, embeddings_dim)
        Returns:
            None
        """

        def evaluate_centers(
            centers: np.ndarray, X: np.ndarray, radius: float | None = None
        ) -> float:
            """
            Evaluates cluster centers vs inputs in two steps:
            1. For each centroid computes min distance to other cluster centers
            2. Computes number of elements in half that distance
            3. Sums all elements inside these raduises and evaluates vs total number of distances
            As a result, chooses centroids with best density nearby cluster centers
            If radius is not None, uses radius instead of 1.
            Arguments:
                - centers: centroids
                - X: inputs (embeddings)
                - radius: radius around each centroid to count elements
            Returns:
                count of elements in proximity to centroids vs total number of distances
            """
            if radius is None:
                radii = self.get_radii()
            else:
                radii = [radius for _ in centers]
            distances = 1 - X @ centers.T
            elements = [
                np.sum(distances[:, i] < radii[i] / 2) for i in range(len(radii))
            ]
            return sum(elements) / np.size(distances)

        #
        options = [
            self.initialize_centers_single_pass(X) for _ in range(self.init_count)
        ]
        scores = [evaluate_centers(option, X) for option in options]
        self.cluster_centers_ = options[scores.index(max(scores))]
        self.cluster_centers_ = self.norm(self.cluster_centers_)

    def initialize_centers_single_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Single K-means cluster centers run
        Arguments:
            - inputs (embeddings)
        Returns:
            centers of clusters
        """
        # Step 1: Initialize the first center randomly
        centers = np.expand_dims(X[np.random.choice(X.shape[0])], 0)

        # Step 2: Initialize the remaining centers using k-means++ method
        for _ in range(1, self.n_clusters):
            # Compute the distance of each point to the nearest center
            # distances = np.array(
            #     [min(np.linalg.norm(x - c) for c in centers) for x in X]
            # )
            min_dist = np.min(1 - centers @ X.T, axis=0)
            # Probability distribution for selecting the next center
            probs = min_dist**2 / np.sum(min_dist**2)

            # Choose the next center based on the probability distribution
            next_center = X[np.random.choice(X.shape[0], p=probs)]
            # centers.append(next_center)
            centers = np.vstack((centers, next_center))

        self.cluster_centers_ = centers
        return centers

    def norm(self, X: np.ndarray, eps=1e-8) -> np.ndarray:
        """
        Returns normalized vectors
        Arguments:
            - X: inputs
        Returns:
            numpy array of normalized inputs
        """
        norm = np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
        norm[norm == 0] = (np.ones_like(norm) * eps)[norm == 0]
        return X / norm

    def fit(self, X: np.ndarray) -> None:
        """
        Fit pass without prediction
        Arguments:
            - X: inputs (embeddings)
        Returns:
            None
        """
        self.initialize_centers(X)

        for _ in range(self.max_iters):
            distances = 1 - X @ self.cluster_centers_.T
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array(
                [X[self.labels_ == j].mean(axis=0) for j in range(self.n_clusters)]
            )
            new_centers = self.norm(new_centers)
            metric = np.mean(1 - np.max(new_centers @ self.cluster_centers_.T, axis=1))
            if metric < self.tol:
                break
            self.cluster_centers_ = new_centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction pass without fitting
        Arguments:
            - X: inputs (embeddings)
        Returns:
            numpy array of labels
        """
        distances = 1 - X @ self.cluster_centers_.T
        return np.argmin(distances, axis=1)

    def predict_with_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction pass without fitting, returns numpy array of weighted
        predictions per each sample per each cluster
        Arguments:
            - X: inputs (embeddings)
        Returns:
            numpy array of weighted predictions (num_samples x num_clusters)
        """
        distances = 1 - X @ self.cluster_centers_.T
        # preds = 1 / (distances / np.min(distances, axis=1, keepdims=1))
        # weights = 1 - (distances / np.sum(distances, axis=1, keepdims=1))
        # return preds * weights
        return (1 / distances) / np.sum(1 / distances, axis=1, keepdims=True)

    def fit_predict(self, X):
        """
        Fit + predict
        """
        self.fit(X)
        return self.predict(X)

    def fit_predict_with_confidence(self, X):
        """
        Fit + predict
        """
        self.fit(X)
        return self.predict_with_confidence(X)

    def get_radii(self):
        """
        Computes cluster radii
        Arguments:
            None
        Returns:
            numpy array of radii around each centroid
        """
        assert len(self.cluster_centers_) > 0, "No speakers to compute radii"
        if len(self.cluster_centers_) == 1:
            return [np.array(2.0)]
        ccdists = 1 - self.cluster_centers_ @ self.cluster_centers_.T
        ccc = [np.concatenate((row[:i], row[i + 1 :])) for i, row in enumerate(ccdists)]
        radii = [np.min(el) for el in ccc]
        return radii
