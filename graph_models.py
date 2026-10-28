from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from torch import Tensor, nn
from torch.nn import functional as F


class _AttentionFusionGCN(nn.Module):
    def __init__(
        self,
        text_dimension: int,
        metadata_dimension: int,
        hidden_dimension: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dimension, hidden_dimension)
        self.metadata_projection = nn.Linear(metadata_dimension, hidden_dimension)
        self.view_attention = nn.Linear(hidden_dimension, 1, bias=False)
        self.classifier = nn.Linear(hidden_dimension, class_count)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )

    def fuse(self, text_hidden: Tensor, metadata_hidden: Tensor) -> Tensor:
        views = torch.stack((text_hidden, metadata_hidden), dim=1)
        attention = torch.softmax(self.view_attention(torch.tanh(views)), dim=1)
        return torch.sum(attention * views, dim=1)

    def forward(
        self,
        text_features: Tensor,
        metadata_features: Tensor,
        text_adjacency: Tensor,
        metadata_adjacency: Tensor,
    ) -> tuple[Tensor, Tensor]:
        text_hidden = F.relu(
            torch.sparse.mm(
                text_adjacency,
                self.text_projection(text_features),
            )
        )
        metadata_hidden = F.relu(
            torch.sparse.mm(
                metadata_adjacency,
                self.metadata_projection(metadata_features),
            )
        )
        fused = self.fuse(text_hidden, metadata_hidden)
        return self.classifier(fused), fused


class DualViewAttentionGCN(ClassifierMixin, BaseEstimator):
    """Two-view feature-graph GCN with attention fusion and supervised contrastive loss."""

    def __init__(
        self,
        n_components: int = 64,
        n_neighbors: int = 5,
        hidden_dimension: int = 64,
        epochs: int = 30,
        learning_rate: float = 0.01,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
        contrastive_batch_size: int = 512,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.hidden_dimension = hidden_dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.contrastive_batch_size = contrastive_batch_size
        self.random_state = random_state

    @staticmethod
    def _input_views(X: object) -> tuple[list[str], list[str]]:
        if not isinstance(X, pd.DataFrame) or "statement" not in X:
            raise ValueError(
                "The dual-view GCN requires a DataFrame with statement and metadata columns."
            )
        statements = X["statement"].fillna("").astype(str).tolist()
        raw_metadata = (
            X["metadata"].fillna("").astype(str).tolist()
            if "metadata" in X
            else [""] * len(statements)
        )
        metadata = [
            value if value.strip() else "no metadata"
            for value in raw_metadata
        ]
        if not statements:
            raise ValueError("The dual-view GCN requires at least one statement.")
        return statements, metadata

    def _embed_view(
        self, features: csr_matrix, fitted_reducer: TruncatedSVD | None = None
    ) -> tuple[np.ndarray, TruncatedSVD | None]:
        mean = np.asarray(features.mean(axis=0)).ravel()
        second_moment = np.asarray(features.multiply(features).mean(axis=0)).ravel()
        total_variance = np.maximum(second_moment - mean**2, 0.0).sum()
        if total_variance <= 1e-12:
            return np.ones((features.shape[0], 1), dtype=np.float32), None
        if features.shape[1] <= 1:
            embeddings = features.toarray().astype(np.float32, copy=False)
            return normalize(embeddings).astype(np.float32, copy=False), None
        if fitted_reducer is None:
            component_count = min(
                self.n_components,
                features.shape[1] - 1,
                features.shape[0] - 1,
            )
            fitted_reducer = TruncatedSVD(
                n_components=component_count,
                n_iter=5,
                random_state=self.random_state,
            )
            embeddings = fitted_reducer.fit_transform(features)
        else:
            embeddings = fitted_reducer.transform(features)
        return normalize(embeddings).astype(np.float32, copy=False), fitted_reducer

    def _build_training_graph(self, embeddings: np.ndarray) -> Tensor:
        neighbor_count = min(self.n_neighbors, len(embeddings) - 1)
        if neighbor_count < 1:
            raise ValueError("Graph training requires at least two examples.")
        neighbors = NearestNeighbors(
            n_neighbors=neighbor_count + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(embeddings)
        distances, indices = neighbors.kneighbors(embeddings)
        rows = np.repeat(np.arange(len(embeddings)), neighbor_count)
        selected_indices = np.empty((len(embeddings), neighbor_count), dtype=np.int64)
        selected_weights = np.empty((len(embeddings), neighbor_count), dtype=np.float32)
        for row in range(len(embeddings)):
            mask = indices[row] != row
            row_indices = indices[row][mask][:neighbor_count]
            row_distances = distances[row][mask][:neighbor_count]
            if len(row_indices) < neighbor_count:
                missing = neighbor_count - len(row_indices)
                row_indices = np.concatenate((row_indices, indices[row][mask][:missing]))
                row_distances = np.concatenate((row_distances, distances[row][mask][:missing]))
            selected_indices[row] = row_indices
            selected_weights[row] = np.maximum(1.0 - row_distances, 1e-6)

        columns = selected_indices.reshape(-1)
        values = selected_weights.reshape(-1)
        adjacency = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(embeddings), len(embeddings)),
            dtype=np.float32,
        )
        adjacency = adjacency.maximum(adjacency.T)
        adjacency.setdiag(1.0)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
        normalized_adjacency = (
            sparse.diags(inverse_sqrt_degree)
            @ adjacency
            @ sparse.diags(inverse_sqrt_degree)
        ).tocoo()
        indices_tensor = torch.from_numpy(
            np.vstack((normalized_adjacency.row, normalized_adjacency.col)).astype(
                np.int64
            )
        )
        values_tensor = torch.from_numpy(normalized_adjacency.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            normalized_adjacency.shape,
            check_invariants=True,
        ).coalesce()

    @staticmethod
    def _supervised_contrastive_loss(
        embeddings: Tensor, labels: Tensor, temperature: float
    ) -> Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        logits = normalized_embeddings @ normalized_embeddings.T / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        identity = torch.eye(
            len(labels),
            device=labels.device,
            dtype=torch.bool,
        )
        positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
        valid_anchors = positive_mask.any(dim=1)
        if not valid_anchors.any():
            return embeddings.sum() * 0.0
        logits = logits.masked_fill(identity, torch.finfo(logits.dtype).min)
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_counts = positive_mask.sum(dim=1).clamp_min(1)
        per_anchor_loss = -(
            log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_counts
        )
        return per_anchor_loss[valid_anchors].mean()

    def _contrastive_indices(
        self, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        class_values = np.unique(labels)
        per_class_limit = max(
            2,
            self.contrastive_batch_size // max(1, len(class_values)),
        )
        sampled = []
        for class_value in class_values:
            indices = np.flatnonzero(labels == class_value)
            if len(indices) > per_class_limit:
                indices = rng.choice(indices, per_class_limit, replace=False)
            sampled.extend(indices.tolist())
        if len(sampled) > self.contrastive_batch_size:
            sampled = rng.choice(
                np.asarray(sampled),
                self.contrastive_batch_size,
                replace=False,
            ).tolist()
        return np.asarray(sampled, dtype=np.int64)

    def fit(self, X: object, y: object) -> DualViewAttentionGCN:
        statements, metadata = self._input_views(X)
        labels = np.asarray(y)
        if len(statements) != len(labels):
            raise ValueError("Input statements and labels must have equal lengths.")
        if len(np.unique(labels)) < 2:
            raise ValueError("The dual-view GCN requires at least two classes.")
        if self.epochs < 1 or self.temperature <= 0 or self.n_neighbors < 1:
            raise ValueError("Epochs, temperature, and neighbor count must be positive.")
        torch.manual_seed(self.random_state)
        self.classes_, encoded_labels = np.unique(labels, return_inverse=True)
        self.text_vectorizer_ = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=80_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=80_000,
                        sublinear_tf=True,
                    ),
                ),
            ],
            transformer_weights={"word": 1.0, "character": 0.6},
        )
        self.metadata_vectorizer_ = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=30_000,
            sublinear_tf=True,
        )
        text_features = csr_matrix(self.text_vectorizer_.fit_transform(statements))
        metadata_features = csr_matrix(
            self.metadata_vectorizer_.fit_transform(metadata)
        )
        self.train_text_embeddings_, self.text_reducer_ = self._embed_view(
            text_features
        )
        self.train_metadata_embeddings_, self.metadata_reducer_ = self._embed_view(
            metadata_features
        )
        text_adjacency = self._build_training_graph(self.train_text_embeddings_)
        metadata_adjacency = self._build_training_graph(
            self.train_metadata_embeddings_
        )

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network_ = _AttentionFusionGCN(
            self.train_text_embeddings_.shape[1],
            self.train_metadata_embeddings_.shape[1],
            self.hidden_dimension,
            len(self.classes_),
        ).to(self.device_)
        text_tensor = torch.from_numpy(self.train_text_embeddings_).to(self.device_)
        metadata_tensor = torch.from_numpy(
            self.train_metadata_embeddings_
        ).to(self.device_)
        text_adjacency = text_adjacency.to(self.device_)
        metadata_adjacency = metadata_adjacency.to(self.device_)
        labels_tensor = torch.from_numpy(encoded_labels.astype(np.int64)).to(
            self.device_
        )
        class_counts = np.bincount(encoded_labels, minlength=len(self.classes_))
        class_weights = len(labels) / (
            len(self.classes_) * np.maximum(class_counts, 1)
        )
        weights_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(
            self.device_
        )
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4,
        )
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            self.network_.train()
            optimizer.zero_grad()
            logits, fused = self.network_(
                text_tensor,
                metadata_tensor,
                text_adjacency,
                metadata_adjacency,
            )
            contrastive_indices = self._contrastive_indices(encoded_labels, rng)
            contrastive_projection = self.network_.contrastive_projection(
                fused[contrastive_indices]
            )
            contrastive_loss = self._supervised_contrastive_loss(
                contrastive_projection,
                labels_tensor[contrastive_indices],
                self.temperature,
            )
            classification_loss = F.cross_entropy(
                logits,
                labels_tensor,
                weight=weights_tensor,
            )
            loss = (
                classification_loss
                + self.contrastive_weight * contrastive_loss
            )
            loss.backward()
            optimizer.step()

        self.network_.eval()
        self.text_neighbor_index_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(labels)),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(self.train_text_embeddings_)
        self.metadata_neighbor_index_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(labels)),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(self.train_metadata_embeddings_)
        self._text_features_count_ = text_features.shape[1]
        self._metadata_features_count_ = metadata_features.shape[1]
        self._feature_names_ = np.concatenate(
            (
                np.asarray(
                    [f"text::{name}" for name in self.text_vectorizer_.get_feature_names_out()],
                    dtype=object,
                ),
                np.asarray(
                    [
                        f"metadata::{name}"
                        for name in self.metadata_vectorizer_.get_feature_names_out()
                    ],
                    dtype=object,
                ),
            )
        )
        return self

    def _transform_views(
        self, X: object
    ) -> tuple[np.ndarray, np.ndarray]:
        statements, metadata = self._input_views(X)
        text_features = csr_matrix(self.text_vectorizer_.transform(statements))
        metadata_features = csr_matrix(
            self.metadata_vectorizer_.transform(metadata)
        )
        text_embeddings, _ = self._embed_view(
            text_features,
            self.text_reducer_,
        )
        metadata_embeddings, _ = self._embed_view(
            metadata_features,
            self.metadata_reducer_,
        )
        return text_embeddings, metadata_embeddings

    @staticmethod
    def _weighted_neighbor_mean(
        embeddings: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        similarities = np.maximum(1.0 - distances, 1e-6)
        weights = similarities / similarities.sum(axis=1, keepdims=True)
        return np.einsum("ij,ijk->ik", weights, embeddings[indices])

    def predict_proba(self, X: object) -> np.ndarray:
        text_embeddings, metadata_embeddings = self._transform_views(X)
        text_distances, text_indices = self.text_neighbor_index_.kneighbors(
            text_embeddings
        )
        metadata_distances, metadata_indices = (
            self.metadata_neighbor_index_.kneighbors(metadata_embeddings)
        )
        text_neighbors = self._weighted_neighbor_mean(
            self.train_text_embeddings_,
            text_distances,
            text_indices,
        )
        metadata_neighbors = self._weighted_neighbor_mean(
            self.train_metadata_embeddings_,
            metadata_distances,
            metadata_indices,
        )
        with torch.no_grad():
            text_tensor = torch.from_numpy(text_embeddings).to(self.device_)
            metadata_tensor = torch.from_numpy(metadata_embeddings).to(self.device_)
            text_neighbors_tensor = torch.from_numpy(text_neighbors).to(self.device_)
            metadata_neighbors_tensor = torch.from_numpy(
                metadata_neighbors
            ).to(self.device_)
            text_hidden = F.relu(
                0.5 * self.network_.text_projection(text_tensor)
                + 0.5 * self.network_.text_projection(text_neighbors_tensor)
            )
            metadata_hidden = F.relu(
                0.5 * self.network_.metadata_projection(metadata_tensor)
                + 0.5
                * self.network_.metadata_projection(metadata_neighbors_tensor)
            )
            fused = self.network_.fuse(text_hidden, metadata_hidden)
            probabilities = torch.softmax(self.network_.classifier(fused), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X: object) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_feature_names_out(self) -> np.ndarray:
        return self._feature_names_

    def inspect_features(
        self, statement: str, metadata: str | None = None
    ) -> tuple[list[str], pd.DataFrame, int]:
        statement_features = csr_matrix(
            self.text_vectorizer_.transform([statement])
        )
        metadata_features = csr_matrix(
            self.metadata_vectorizer_.transform(
                [metadata if metadata else "no metadata"]
            )
        )
        feature_row = sparse.hstack(
            (statement_features, metadata_features),
            format="csr",
        )
        top_indices = np.argsort(feature_row.data)[::-1][:12]
        indices = feature_row.indices[top_indices]
        values = feature_row.data[top_indices]
        top_features = pd.DataFrame(
            {
                "Feature": self._feature_names_[indices],
                "TF-IDF weight": values,
            }
        )
        word_vectorizer = self.text_vectorizer_.transformer_list[0][1]
        tokens = word_vectorizer.build_analyzer()(statement)
        return tokens, top_features, feature_row.nnz


__all__ = ["DualViewAttentionGCN"]
