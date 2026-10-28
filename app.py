from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
from html import escape
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import pickle
from pathlib import Path
import re
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import pymysql
import streamlit as st
from scipy.special import softmax
from scipy.sparse import csr_matrix, issparse
from streamlit.errors import StreamlitSecretNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from graph_models import DualViewAttentionGCN


AUTH_SECRETS_PATH = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
APP_LOG_PATH = Path(__file__).resolve().parent / "logs" / "app.log"


def configure_app_logger() -> logging.Logger:
    logger = logging.getLogger("fake_news_text_analyzer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == APP_LOG_PATH.resolve()
        for handler in logger.handlers
    ):
        APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            APP_LOG_PATH,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


APP_LOGGER = configure_app_logger()


def log_failure(operation: str, error: Exception) -> None:
    APP_LOGGER.exception(
        "Operation failed: %s (%s): %s",
        operation,
        type(error).__name__,
        error,
    )


def read_recent_app_logs(max_lines: int = 80) -> str:
    if not APP_LOG_PATH.is_file():
        return "No application errors have been logged."
    lines = APP_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def load_dashboard_auth() -> dict[str, str]:
    if AUTH_SECRETS_PATH.is_file():
        secrets_data = tomllib.loads(AUTH_SECRETS_PATH.read_text(encoding="utf-8"))
        auth_config = secrets_data.get("auth", {})
    else:
        try:
            auth_config = st.secrets.get("auth", {})
        except StreamlitSecretNotFoundError:
            auth_config = {}
    return {
        "username": str(auth_config.get("username", "")),
        "password": str(auth_config.get("password", "")),
    }


def update_dashboard_password(new_password: str) -> None:
    if not AUTH_SECRETS_PATH.is_file():
        raise ValueError(
            "Password changes require the local private "
            ".streamlit/secrets.toml file."
        )
    contents = AUTH_SECRETS_PATH.read_text(encoding="utf-8")
    lines = contents.splitlines(keepends=True)
    auth_start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"\s*\[auth\]\s*(?:#.*)?(?:\r?\n)?", line)
        ),
        None,
    )
    if auth_start is None:
        raise ValueError("The private secrets file has no [auth] section.")
    auth_end = next(
        (
            index
            for index in range(auth_start + 1, len(lines))
            if re.match(r"\s*\[.*\]\s*(?:#.*)?(?:\r?\n)?$", lines[index])
        ),
        len(lines),
    )
    password_index = next(
        (
            index
            for index in range(auth_start + 1, auth_end)
            if re.match(r"\s*password\s*=", lines[index])
        ),
        None,
    )
    if password_index is None:
        raise ValueError("The [auth] section has no password setting.")
    newline = "\r\n" if "\r\n" in contents else "\n"
    lines[password_index] = f"password = {json.dumps(new_password)}{newline}"
    temporary_path = AUTH_SECRETS_PATH.with_name("secrets.toml.tmp")
    temporary_path.write_text("".join(lines), encoding="utf-8", newline="")
    temporary_path.replace(AUTH_SECRETS_PATH)


def require_dashboard_login() -> None:
    auth_config = load_dashboard_auth()
    configured_username = auth_config["username"]
    configured_password = auth_config["password"]
    if not configured_username or not configured_password:
        st.error(
            "Dashboard login is not configured. Add `[auth]`, `username`, "
            "and `password` to the private `.streamlit/secrets.toml` file "
            "using the example configuration."
        )
        st.stop()

    if st.session_state.get("dashboard_authenticated"):
        with st.sidebar.expander("Change dashboard password"):
            with st.form("change_dashboard_password_form"):
                current_password = st.text_input(
                    "Current password", type="password"
                )
                new_password = st.text_input(
                    "New password", type="password"
                )
                confirm_password = st.text_input(
                    "Confirm new password", type="password"
                )
                change_submitted = st.form_submit_button("Update password")
            if change_submitted:
                current_matches = hmac.compare_digest(
                    current_password.encode("utf-8"),
                    configured_password.encode("utf-8"),
                )
                if not current_matches:
                    st.sidebar.error("Current password is incorrect.")
                elif len(new_password) < 14:
                    st.sidebar.error(
                        "Choose a password with at least 14 characters."
                    )
                elif not hmac.compare_digest(
                    new_password.encode("utf-8"),
                    confirm_password.encode("utf-8"),
                ):
                    st.sidebar.error("The new passwords do not match.")
                else:
                    try:
                        update_dashboard_password(new_password)
                    except (OSError, ValueError) as error:
                        log_failure("change dashboard password", error)
                        st.sidebar.error(
                            f"Could not update password: {error}"
                        )
                    else:
                        st.sidebar.success(
                            "Password updated. Use the new password next time you sign in."
                        )
        with st.sidebar.expander("Application logs"):
            st.code(read_recent_app_logs(), language="log")
        if st.sidebar.button("Log out", key="dashboard_logout"):
            st.session_state.pop("dashboard_authenticated", None)
            st.rerun()
        return

    st.title("Dashboard sign in")
    st.caption("Local access is restricted to the configured dashboard user.")
    with st.form("dashboard_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        username_matches = hmac.compare_digest(
            username.encode("utf-8"), configured_username.encode("utf-8")
        )
        password_matches = hmac.compare_digest(
            password.encode("utf-8"), configured_password.encode("utf-8")
        )
        if username_matches and password_matches:
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        st.error("Incorrect username or password.")
    st.stop()


def build_analysis_export(
    dataset: pd.DataFrame,
    dataset_name: str,
    model_metrics: dict[str, dict[str, object]],
    trained_models: dict[str, object],
    online_articles: pd.DataFrame,
) -> bytes:
    """Bundle the current dataset, measured results, graphs, and models."""
    bundle = BytesIO()
    with ZipFile(bundle, mode="w") as archive:
        archive.writestr(
            "dataset/labeled_dataset.csv",
            dataset.to_csv(index=False),
        )
        metric_rows = [
            {
                "model": name,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "report"
                },
            }
            for name, metrics in model_metrics.items()
        ]
        metrics_frame = pd.DataFrame(metric_rows)
        archive.writestr(
            "analysis/model_comparison.csv",
            metrics_frame.to_csv(index=False),
        )
        report_payload = {
            "dataset_name": dataset_name,
            "dataset_rows": len(dataset),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_metrics": model_metrics,
            "note": (
                "Metrics are local measurements for this app's dataset split. "
                "Predictions are model estimates, not independent fact checks."
            ),
        }
        archive.writestr(
            "analysis/report.json",
            json.dumps(
                report_payload,
                indent=2,
                allow_nan=False,
                default=lambda value: (
                    value.item() if isinstance(value, np.generic) else str(value)
                ),
            ),
        )

        model_figure = go.Figure()
        for metric, label in (
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro-F1"),
            ("micro_f1", "Micro-F1"),
        ):
            model_figure.add_trace(
                go.Bar(
                    name=label,
                    x=list(model_metrics),
                    y=[float(model_metrics[name][metric]) for name in model_metrics],
                )
            )
        model_figure.update_layout(
            title="Held-out model comparison",
            barmode="group",
            yaxis={"title": "Score", "range": [0, 1]},
            xaxis_title="Model",
        )
        labels = (
            dataset["label"].fillna("Unknown").value_counts().rename_axis("Label")
        )
        label_figure = go.Figure(
            go.Bar(
                x=labels.index.tolist(),
                y=labels.values.tolist(),
                marker_color=[
                    GRAPH_COLORS.get(label, "#8b92a0")
                    for label in labels.index.tolist()
                ],
            )
        )
        label_figure.update_layout(
            title="Labeled dataset distribution",
            xaxis_title="Dataset label",
            yaxis_title="Records",
        )
        archive.writestr(
            "graphs/model_comparison.html",
            pio.to_html(model_figure, full_html=True, include_plotlyjs="cdn"),
        )
        archive.writestr(
            "graphs/dataset_labels.html",
            pio.to_html(label_figure, full_html=True, include_plotlyjs="cdn"),
        )
        if "publisher" in dataset:
            sources = (
                dataset["publisher"]
                .fillna("Unknown source")
                .replace("", "Unknown source")
                .value_counts()
                .head(20)
            )
            source_figure = go.Figure(
                go.Bar(x=sources.index.tolist(), y=sources.values.tolist())
            )
            source_figure.update_layout(
                title="Top dataset publishers/speakers",
                xaxis_title="Publisher / speaker",
                yaxis_title="Records",
            )
            archive.writestr(
                "graphs/dataset_sources.html",
                pio.to_html(source_figure, full_html=True, include_plotlyjs="cdn"),
            )
        if not online_articles.empty:
            archive.writestr(
                "dataset/scraped_headlines.csv",
                online_articles.to_csv(index=False),
            )
            if {"publisher", "predicted_category"}.issubset(online_articles.columns):
                counts = online_articles.groupby(
                    ["publisher", "predicted_category"], observed=True
                ).size()
                headline_figure = go.Figure()
                for category in CLASS_ORDER:
                    headline_figure.add_trace(
                        go.Bar(
                            name=category,
                            x=sorted(online_articles["publisher"].unique()),
                            y=[
                                int(counts.get((publisher, category), 0))
                                for publisher in sorted(
                                    online_articles["publisher"].unique()
                                )
                            ],
                            marker_color=GRAPH_COLORS[category],
                        )
                    )
                headline_figure.update_layout(
                    title="Scraped headline model estimates by source",
                    barmode="stack",
                    xaxis_title="Source",
                    yaxis_title="Headlines",
                )
                archive.writestr(
                    "graphs/scraped_headline_estimates.html",
                    pio.to_html(headline_figure, full_html=True, include_plotlyjs="cdn"),
                )

        manifest: dict[str, object] = {
            "dataset_name": dataset_name,
            "model_files": [],
            "warning": (
                "Only load model artifacts from sources you trust. Serialized "
                "joblib files can execute code when loaded."
            ),
        }
        for model_name, model in trained_models.items():
            filename = re.sub(r"[^a-z0-9]+", "_", model_name.casefold()).strip("_")
            model_buffer = BytesIO()
            joblib.dump(model, model_buffer, compress=3)
            model_path = f"models/{filename}.joblib"
            archive.writestr(model_path, model_buffer.getvalue())
            manifest["model_files"].append(model_path)
        archive.writestr(
            "models/manifest.json",
            json.dumps(manifest, indent=2),
        )
        archive.writestr(
            "README.txt",
            "Fake News Text Analyzer export\n\n"
            "Contents: labeled dataset, measured model metrics, interactive "
            "HTML graphs, and serialized fitted model artifacts. The graph "
            "HTML files load Plotly.js from a CDN. Dataset labels represent "
            "the supplied dataset; scraped headline categories, when "
            "included, are model estimates and not verified labels.\n\n"
            "Security: joblib model files are trusted-code artifacts. Do not "
            "load them unless you trust the person who produced this archive.\n",
        )
    return bundle.getvalue()


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

LABEL_MAP = {
    "pants-fire": "False",
    "false": "False",
    "fake": "False",
    "fabricated": "False",
    "misinformation": "False",
    "barely-true": "Suspicious",
    "half-true": "Suspicious",
    "suspicious": "Suspicious",
    "mixed": "Suspicious",
    "partly-false": "Suspicious",
    "mostly-true": "True",
    "true": "True",
    "real": "True",
    "factual": "True",
}
CLASS_ORDER = ["False", "Suspicious", "True"]
MODEL_NAMES = [
    "Logistic Regression",
    "Multinomial Naive Bayes",
    "Complement Naive Bayes",
    "SGD Classifier",
    "Linear SVM (word + character)",
    "Dual-view TF-IDF + metadata",
    "Dual-view GCN + attention + SupCon",
]
MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Existing balanced linear classifier baseline",
    "Multinomial Naive Bayes": "New: probabilistic model for sparse word features",
    "Complement Naive Bayes": "New: Naive Bayes variant suited to imbalanced text",
    "SGD Classifier": "New: log-loss linear model trained with stochastic gradient descent",
    "Linear SVM (word + character)": (
        "New: margin-based classifier using word and character TF-IDF"
    ),
    "Dual-view TF-IDF + metadata": (
        "Word + character text features fused with non-identifying context metadata"
    ),
    "Dual-view GCN + attention + SupCon": (
        "Two-view graph convolution with learned attention fusion and "
        "supervised contrastive regularization"
    ),
}
REPORTED_INTERNAL_ABLATIONS = [
    {
        "Model variant": "Text GCN View Only",
        "Accuracy (%)": "Not reported",
        "Macro-F1 (%)": "Not reported",
        "Precision (%)": "Not reported",
        "Recall (%)": "Not reported",
    },
    {
        "Model variant": "Metadata GCN View Only",
        "Accuracy (%)": "Not reported",
        "Macro-F1 (%)": "Not reported",
        "Precision (%)": "Not reported",
        "Recall (%)": "Not reported",
    },
    {
        "Model variant": "Dual-View GCN (Concatenation)",
        "Accuracy (%)": "Not reported",
        "Macro-F1 (%)": "Not reported",
        "Precision (%)": "Not reported",
        "Recall (%)": "Not reported",
    },
    {
        "Model variant": "Attention Fusion + SupCon (proposed)",
        "Accuracy (%)": "Not reported",
        "Macro-F1 (%)": "Not reported",
        "Precision (%)": "Not reported",
        "Recall (%)": "Not reported",
    },
]
REFERENCE_DATASET_SPECS = [
    {
        "Dataset name": "3Sources",
        "#Samples": 169,
        "#Views": 3,
        "Feature dimensions": "3,560 / 3,631 / 3,068",
        "#Classes": 6,
        "Data type": "Textual dataset",
    },
    {
        "Dataset name": "Citeseer",
        "#Samples": 3312,
        "#Views": 2,
        "Feature dimensions": "3,703 / 3,312",
        "#Classes": 6,
        "Data type": "Textual dataset",
    },
    {
        "Dataset name": "GRAZ02",
        "#Samples": 1476,
        "#Views": 6,
        "Feature dimensions": "512 / 32 / 256 / 500 / 500 / 680",
        "#Classes": 4,
        "Data type": "Object images",
    },
    {
        "Dataset name": "MNIST",
        "#Samples": 2000,
        "#Views": 3,
        "Feature dimensions": "30 / 9 / 9",
        "#Classes": 10,
        "Data type": "Digit dataset",
    },
    {
        "Dataset name": "NGS",
        "#Samples": 500,
        "#Views": 3,
        "Feature dimensions": "2,000 / 2,000 / 2,000",
        "#Classes": 5,
        "Data type": "Textual dataset",
    },
    {
        "Dataset name": "Out-Scene",
        "#Samples": 2688,
        "#Views": 4,
        "Feature dimensions": "512 / 432 / 256 / 48",
        "#Classes": 8,
        "Data type": "Scene image",
    },
    {
        "Dataset name": "NoisyMNIST",
        "#Samples": 10000,
        "#Views": 3,
        "Feature dimensions": "784 / 784 / 784",
        "#Classes": 10,
        "Data type": "Digit dataset",
    },
    {
        "Dataset name": "LIAR (Ours)",
        "#Samples": 10240,
        "#Views": 2,
        "Feature dimensions": "384 (SBERT) / 384 (Meta)",
        "#Classes": 3,
        "Data type": "Text/Metadata",
    },
]
GRAPH_COLORS = {
    "False": "#FF4D8D",
    "Suspicious": "#FFB454",
    "True": "#34D399",
    "Online article": "#00D4FF",
}
LABEL_COLUMN_NAMES = {
    "label",
    "labels",
    "label_name",
    "class",
    "class_label",
    "target",
    "veracity",
    "rating",
}
TEXT_COLUMN_NAMES = {
    "statement",
    "statement_text",
    "claim",
    "claim_text",
    "text",
    "news_text",
    "article_text",
    "content",
    "article",
    "body",
    "news",
    "headline",
    "title",
}
METADATA_COLUMN_NAMES = (
    "subject",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
)
KNOWN_LABELS = set(LABEL_MAP)
PUBLISHER_COLUMN_NAMES = {
    "source",
    "site",
    "site_url",
    "publisher",
    "domain",
    "website",
    "outlet",
    "speaker",
}
AUTHOR_COLUMN_NAMES = {
    "author",
    "authors",
    "speaker",
    "reporter",
    "journalist",
    "writer",
    "byline",
    "creator",
}
MAX_NEWS_PAGE_BYTES = 5 * 1024 * 1024
MAX_HEADLINES_PER_SOURCE = 20
MAX_GRAPH_DATASET_NODES = 180
MAX_DATASET_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_DATASET_MEMBER_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class NewsSource:
    url: str
    allowed_hosts: frozenset[str]
    article_path: re.Pattern[str]


NEWS_SOURCES = {
    "Africa Intelligence": NewsSource(
        url="https://www.africaintelligence.com/",
        allowed_hosts=frozenset({"africaintelligence.com", "www.africaintelligence.com"}),
        article_path=re.compile(r"^/(?:[a-z-]+/){1,2}\d{4}/\d{2}/\d{2}/"),
    ),
    "Al Jazeera": NewsSource(
        url="https://www.aljazeera.com/",
        allowed_hosts=frozenset({"aljazeera.com", "www.aljazeera.com"}),
        article_path=re.compile(
            r"^/(?:news|features|opinions|economy|sports|culture|video|program|podcasts)/"
        ),
    ),
    "Voice of America": NewsSource(
        url="https://www.voanews.com/",
        allowed_hosts=frozenset({"voanews.com", "www.voanews.com"}),
        article_path=re.compile(r"^/a/"),
    ),
    "IGIHE": NewsSource(
        url="https://igihe.com/",
        allowed_hosts=frozenset({"igihe.com", "www.igihe.com"}),
        article_path=re.compile(r"^/[a-z-]+/article/"),
    ),
    "New Vision": NewsSource(
        url="https://www.newvision.co.ug/",
        allowed_hosts=frozenset({"newvision.co.ug", "www.newvision.co.ug"}),
        article_path=re.compile(r"^/category/[a-z-]+/[^/]+-NV_\d+_\d+$"),
    ),
    "BBC News": NewsSource(
        url="https://www.bbc.com/news",
        allowed_hosts=frozenset({"www.bbc.com"}),
        article_path=re.compile(r"^/news/articles/"),
    ),
}


class ArticleLinkParser(HTMLParser):
    """Extract bounded article headlines from an approved news landing page."""

    def __init__(self, source: NewsSource) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.articles: dict[str, str] = {}
        self._active_url: str | None = None
        self._active_text: list[str] = []
        self._active_title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        anchor_attributes = dict(attrs)
        href = anchor_attributes.get("href")
        if not href:
            return
        article_url = urljoin(self.source.url, href)
        parsed_url = urlsplit(article_url)
        if (
            parsed_url.scheme == "https"
            and parsed_url.hostname
            and parsed_url.hostname.lower() in self.source.allowed_hosts
            and self.source.article_path.match(parsed_url.path)
        ):
            self._active_url = article_url
            self._active_text = []
            self._active_title = (
                anchor_attributes.get("title")
                or re.sub(
                    r"^Read full article:\s*",
                    "",
                    anchor_attributes.get("aria-label", ""),
                    flags=re.IGNORECASE,
                )
            )

    def handle_data(self, data: str) -> None:
        if self._active_url:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_url:
            title = " ".join(" ".join(self._active_text).split()) or self._active_title
            if len(title) > 15:
                self.articles.setdefault(self._active_url, title)
            self._active_url = None
            self._active_text = []
            self._active_title = ""


class RejectRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_source_articles(source_name: str) -> pd.DataFrame:
    """Fetch a bounded list of headlines from one configured news outlet."""
    source = NEWS_SOURCES.get(source_name)
    if source is None:
        raise ValueError(f"Unknown news source: {source_name}")

    request = Request(
        source.url,
        headers={"User-Agent": "FakeNewsTextAnalyzer/1.0 (public headline screening)"},
    )
    opener = build_opener(RejectRedirectHandler())
    try:
        with opener.open(request, timeout=15) as response:
            final_url = urlsplit(response.geturl())
            if (
                final_url.scheme != "https"
                or not final_url.hostname
                or final_url.hostname.lower() not in source.allowed_hosts
            ):
                raise ValueError(
                    f"{source_name} redirected outside its approved domain."
                )
            page = response.read(MAX_NEWS_PAGE_BYTES + 1)
    except HTTPError as error:
        raise ValueError(
            f"{source_name} returned HTTP status {error.code}; try again later."
        ) from error
    except URLError as error:
        raise ValueError(f"Could not reach {source_name}: {error.reason}") from error

    if len(page) > MAX_NEWS_PAGE_BYTES:
        raise ValueError(f"{source_name} page exceeded the 5 MB download limit.")

    parser = ArticleLinkParser(source)
    parser.feed(page.decode("utf-8", errors="replace"))
    articles = list(parser.articles.items())[:MAX_HEADLINES_PER_SOURCE]
    if not articles:
        raise ValueError(
            f"No article links were found for {source_name}. "
            "The site may have changed its page layout or restricted access."
        )

    return pd.DataFrame(
        [
            {
                "statement": title,
                "label": "Online article",
                "publisher": source_name,
                "url": url,
                "node_type": "Online article",
            }
            for url, title in articles
        ]
    )


def normalize_publisher(value: str) -> str:
    publisher = value.strip()
    if not publisher:
        return "Unknown source"
    parsed = urlsplit(publisher if "://" in publisher else f"//{publisher}")
    host = parsed.hostname
    if host:
        return host.lower().removeprefix("www.")
    return publisher[:80]


def render_article_graph(
    dataset: pd.DataFrame, online_articles: pd.DataFrame
) -> None:
    """Render a bounded, interactive graph connected by shared publisher."""
    existing = dataset.copy()
    existing["publisher"] = existing["publisher"].map(normalize_publisher)
    if len(existing) > MAX_GRAPH_DATASET_NODES:
        existing = existing.sample(
            n=MAX_GRAPH_DATASET_NODES, random_state=42
        )
    existing = existing.reset_index(drop=True)
    existing["url"] = ""
    existing["node_type"] = "Dataset article"
    existing["label"] = existing["label"].fillna("Unknown")
    unknown_source_mask = (
        existing["publisher"].astype(str).str.strip().str.lower().isin(
            {"", "unknown", "unknown source", "nan"}
        )
    )
    existing.loc[unknown_source_mask, "publisher"] = [
        f"unknown-source-{index}"
        for index in existing.index[unknown_source_mask]
    ]
    graph_nodes = pd.concat(
        [existing, online_articles],
        ignore_index=True,
    )
    graph_nodes = graph_nodes[
        graph_nodes["statement"].fillna("").str.strip().ne("")
    ].reset_index(drop=True)
    graph_nodes["publisher"] = graph_nodes["publisher"].map(normalize_publisher)
    graph_nodes["node_id"] = [
        f"node-{index + 1:04d}" for index in range(len(graph_nodes))
    ]

    publisher_groups = {
        publisher: group.index.tolist()
        for publisher, group in graph_nodes.groupby("publisher", sort=True)
    }
    edges: list[tuple[int, int]] = []
    for node_indices in publisher_groups.values():
        if len(node_indices) > 1:
            anchor = node_indices[0]
            edges.extend((anchor, node_index) for node_index in node_indices[1:])

    publisher_names = list(publisher_groups)
    node_positions: dict[int, tuple[float, float]] = {}
    publisher_count = max(1, len(publisher_names))
    for publisher_index, publisher in enumerate(publisher_names):
        node_indices = publisher_groups[publisher]
        center_angle = 2 * 3.141592653589793 * publisher_index / publisher_count
        center_x = 5 * math.cos(center_angle)
        center_y = 5 * math.sin(center_angle)
        local_radius = min(1.4, 0.18 * len(node_indices))
        for position, node_index in enumerate(node_indices):
            angle = 2 * 3.141592653589793 * position / max(1, len(node_indices))
            node_positions[node_index] = (
                center_x + local_radius * math.cos(angle),
                center_y + local_radius * math.sin(angle),
            )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in edges:
        edge_x.extend([node_positions[source][0], node_positions[target][0], None])
        edge_y.extend([node_positions[source][1], node_positions[target][1], None])

    node_colors = [
        GRAPH_COLORS.get(
            row["label"] if row["node_type"] != "Online article" else "Online article",
            "#8b92a0",
        )
        for _, row in graph_nodes.iterrows()
    ]
    node_symbols = [
        "diamond" if node_type == "Online article" else "circle"
        for node_type in graph_nodes["node_type"]
    ]
    hover_text = [
        (
            f"<b>{escape(str(row['node_id']))}: "
            f"{escape(str(row['statement'])[:220])}</b><br>"
            f"Publisher: {escape(str(row['publisher']))}<br>"
            f"Category: {escape(str(row['label']))}<br>"
            f"Type: {escape(str(row['node_type']))}"
            + (
                f"<br><a href='{escape(str(row['url']), quote=True)}'>Open article</a>"
                if row["url"]
                else ""
            )
        )
        for _, row in graph_nodes.iterrows()
    ]
    positions = [node_positions[index] for index in graph_nodes.index]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1, "color": "rgba(120, 130, 150, 0.35)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            mode="markers",
            marker={
                "size": 12,
                "color": node_colors,
                "symbol": node_symbols,
                "line": {"width": 1, "color": "#ffffff"},
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=650,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{len(graph_nodes)} nodes and {len(edges)} same-publisher links. "
        "Dataset nodes are sampled to at most 180 for responsiveness. "
        "Each link joins records from the same publisher; it does not mean they "
        "discuss the same claim."
    )
    node_table = graph_nodes[
        ["node_id", "publisher", "label", "node_type", "statement", "url"]
    ].rename(
        columns={
            "node_id": "Node ID",
            "publisher": "Publisher",
            "label": "Category",
            "node_type": "Node type",
            "statement": "Statement / headline",
            "url": "Article URL",
        }
    )
    edge_table = pd.DataFrame(
        [
            {
                "From node": graph_nodes.loc[source, "node_id"],
                "To node": graph_nodes.loc[target, "node_id"],
                "Publisher": graph_nodes.loc[source, "publisher"],
                "Relationship": "Same publisher",
            }
            for source, target in edges
        ],
        columns=["From node", "To node", "Publisher", "Relationship"],
    )
    nodes_tab, edges_tab = st.tabs(
        [f"Nodes ({len(node_table)})", f"Edges ({len(edge_table)})"]
    )
    with nodes_tab:
        st.dataframe(
            node_table,
            width="stretch",
            hide_index=True,
            alt="Graph nodes with stable IDs, publisher, category, record type, text, and URL",
        )
    with edges_tab:
        if edge_table.empty:
            st.info(
                "No edges for the current selection. Choose multiple records "
                "from the same publisher to create shared-publisher links."
            )
        else:
            st.dataframe(
                edge_table,
                width="stretch",
                hide_index=True,
                alt="Graph edges showing node IDs connected by shared publisher",
            )


def load_liar_tsv(file_bytes: bytes) -> pd.DataFrame:
    """Load LIAR data or a labeled text dataset with a recognizable header."""
    try:
        content = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            "The dataset must be a UTF-8 encoded LIAR TSV or CSV file."
        ) from error

    sample = "\n".join(content.splitlines()[:20])
    if not sample.strip():
        raise ValueError("The uploaded dataset is empty.")

    detected_delimiter = next(
        (candidate for candidate in ("\t", ",", ";", "|") if candidate in sample),
        None,
    )
    delimiter = detected_delimiter or "\t"
    csv.field_size_limit(max(csv.field_size_limit(), len(content)))
    rows = []
    reader_options = {"quoting": csv.QUOTE_NONE} if delimiter == "\t" else {}
    for row in csv.reader(StringIO(content), delimiter=delimiter, **reader_options):
        if delimiter != "\t":
            while row and not row[-1].strip():
                row.pop()
        if row and any(value.strip() for value in row):
            rows.append(row)
    if not rows:
        raise ValueError("The uploaded dataset is empty.")

    if all(
        re.fullmatch(r"column\d+", value.strip(), flags=re.IGNORECASE)
        for value in rows[0]
    ):
        rows.pop(0)
    if not rows:
        raise ValueError("The uploaded dataset contains no article records.")

    def normalize_header(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    first_row = [normalize_header(value) for value in rows[0]]
    label_index = next(
        (index for index, name in enumerate(first_row) if name in LABEL_COLUMN_NAMES),
        None,
    )
    text_indices = [index for index, name in enumerate(first_row) if name in TEXT_COLUMN_NAMES]
    has_header = label_index is not None and bool(text_indices)

    if has_header:
        if any(len(row) > len(first_row) for row in rows[1:]):
            raise ValueError(
                "The dataset contains a data row with more columns than its header."
            )
        for row in rows[1:]:
            row.extend([""] * (len(first_row) - len(row)))
        data = pd.DataFrame(rows[1:], columns=first_row, dtype=str).fillna("")
        text_columns = [
            name
            for name in first_row
            if name in TEXT_COLUMN_NAMES and name in data.columns
        ]
        preferred_text = next(
            (
                name
                for name in (
                    "statement",
                    "claim",
                    "text",
                    "content",
                    "article",
                    "body",
                    "news",
                    "headline",
                    "title",
                )
                if name in text_columns
            ),
            None,
        )
        if preferred_text is None:
            raise ValueError(
                "Could not identify the statement text column. Name it statement, "
                "claim, text, content, article, body, news, headline, or title."
            )
        if "title" in data.columns and preferred_text != "title":
            data["statement"] = (
                data["title"].str.strip() + " " + data[preferred_text].str.strip()
            ).str.strip()
        else:
            data["statement"] = data[preferred_text].str.strip()
        data["label"] = data[first_row[label_index]].str.strip().str.lower()
        publisher_column = next(
            (name for name in first_row if name in PUBLISHER_COLUMN_NAMES),
            None,
        )
        if publisher_column:
            data["publisher"] = data[publisher_column].str.strip()
        else:
            data["publisher"] = "Unknown source"
        author_column = next(
            (name for name in first_row if name in AUTHOR_COLUMN_NAMES),
            None,
        )
        data["author"] = (
            data[author_column].str.strip() if author_column else ""
        )
        metadata_columns = [
            name for name in METADATA_COLUMN_NAMES if name in data.columns
        ]
        data["metadata"] = data.apply(
            lambda row: " ".join(
                f"{column.replace('_', ' ')} {row[column].strip()}"
                for column in metadata_columns
                if row[column].strip()
            ),
            axis=1,
        )
    elif all(len(row) == len(LIAR_COLUMNS) for row in rows):
        data = pd.DataFrame(rows, columns=LIAR_COLUMNS, dtype=str).fillna("")
        data["statement"] = data["statement"].str.strip()
        data["label"] = data["label"].str.strip().str.lower()
        data["publisher"] = data["speaker"].str.strip()
        data["author"] = data["speaker"].str.strip()
        data["metadata"] = data.apply(
            lambda row: " ".join(
                f"{column.replace('_', ' ')} {row[column].strip()}"
                for column in METADATA_COLUMN_NAMES
                if row[column].strip()
            ),
            axis=1,
        )
    else:
        column_count = len(rows[0])
        if any(len(row) != column_count for row in rows):
            raise ValueError(
                "The dataset has inconsistent row lengths. Ensure each row has "
                "the same number of delimited columns."
            )

        raw_data = pd.DataFrame(rows, dtype=str).fillna("")
        label_scores = {
            column: raw_data[column]
            .str.strip()
            .str.lower()
            .isin(KNOWN_LABELS)
            .mean()
            for column in raw_data.columns
        }
        label_candidates = [
            column for column, score in label_scores.items() if score >= 0.8
        ]
        if len(label_candidates) != 1:
            raise ValueError(
                f"Could not identify the label column in this {column_count}-column "
                "file. Add a header naming the label column (label/class/target) "
                "and text column (statement/text/article), or use standard LIAR "
                "format."
            )

        label_column = label_candidates[0]
        text_candidates = [
            column for column in raw_data.columns if column != label_column
        ]
        if not text_candidates:
            raise ValueError("The dataset must contain a text column as well as labels.")
        text_column = max(
            text_candidates,
            key=lambda column: raw_data[column].str.len().mean(),
        )
        data = pd.DataFrame(
            {
                "label": raw_data[label_column].str.strip().str.lower(),
                "statement": raw_data[text_column].str.strip(),
                "publisher": "Unknown source",
                "author": "",
                "metadata": "no metadata",
            }
        )

    data["label"] = data["label"].map(LABEL_MAP)
    data["statement"] = data["statement"].str.strip()
    if "metadata" not in data:
        data["metadata"] = "no metadata"
    data["metadata"] = data["metadata"].fillna("").str.strip()
    data.loc[data["metadata"].eq(""), "metadata"] = "no metadata"
    if "publisher" not in data:
        data["publisher"] = "Unknown source"
    data["publisher"] = data["publisher"].fillna("").replace("", "Unknown source")
    if "author" not in data:
        data["author"] = ""
    data["author"] = data["author"].fillna("").str.strip()
    data = data.dropna(subset=["label"])
    data = data[data["statement"].ne("")]
    return data.reset_index(drop=True)


def load_liar_archive(file_bytes: bytes) -> pd.DataFrame:
    """Load the training split from a LIAR dataset ZIP archive."""
    if len(file_bytes) > MAX_DATASET_ARCHIVE_BYTES:
        raise ValueError("The dataset ZIP exceeds the 100 MB upload limit.")
    try:
        with ZipFile(BytesIO(file_bytes)) as archive:
            train_files = [
                member
                for member in archive.infolist()
                if Path(member.filename).name.lower() == "train.tsv"
                and not member.is_dir()
            ]
            if not train_files:
                raise ValueError("The ZIP archive does not contain a train.tsv file.")
            if len(train_files) > 1:
                preferred = [
                    member
                    for member in train_files
                    if member.filename.lower().endswith("liar_dataset-master/train.tsv")
                ]
                if len(preferred) != 1:
                    raise ValueError(
                        "The ZIP archive contains multiple train.tsv files; "
                        "upload an archive with one training split."
                    )
                train_file = preferred[0]
            else:
                train_file = train_files[0]
            if train_file.file_size > MAX_DATASET_MEMBER_BYTES:
                raise ValueError("The train.tsv file in the ZIP exceeds 100 MB.")
            return load_liar_tsv(archive.read(train_file))
    except BadZipFile as error:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from error
    except RuntimeError as error:
        raise ValueError(
            "The ZIP archive could not be read. Check whether it is encrypted "
            "or damaged."
        ) from error


def load_dataset_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if Path(filename).suffix.lower() == ".zip":
        return load_liar_archive(file_bytes)
    return load_liar_tsv(file_bytes)


def create_word_character_features() -> FeatureUnion:
    return FeatureUnion(
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


def create_dual_view_features() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", create_word_character_features(), "statement"),
            (
                "metadata",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
                "metadata",
            ),
        ],
        transformer_weights={"text": 1.0, "metadata": 0.35},
    )


def create_model(
    model_name: str = "Logistic Regression",
) -> Pipeline | DualViewAttentionGCN:
    feature_transformer: TfidfVectorizer | FeatureUnion | ColumnTransformer = (
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )
    )
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "Multinomial Naive Bayes":
        classifier = MultinomialNB(alpha=0.5)
    elif model_name == "Complement Naive Bayes":
        classifier = ComplementNB(alpha=0.5)
    elif model_name == "SGD Classifier":
        classifier = SGDClassifier(
            loss="log_loss",
            max_iter=1_000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "Linear SVM (word + character)":
        feature_transformer = create_word_character_features()
        classifier = LinearSVC(
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "Dual-view TF-IDF + metadata":
        feature_transformer = create_dual_view_features()
        classifier = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "Dual-view GCN + attention + SupCon":
        return DualViewAttentionGCN(
            n_components=64,
            n_neighbors=5,
            hidden_dimension=64,
            epochs=30,
            contrastive_weight=0.1,
            temperature=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("tfidf", feature_transformer),
            ("classifier", classifier),
        ]
    )


def uses_metadata_view(model: Pipeline | DualViewAttentionGCN) -> bool:
    if isinstance(model, DualViewAttentionGCN):
        return True
    return isinstance(model.named_steps["tfidf"], ColumnTransformer)


def model_input(
    model: Pipeline | DualViewAttentionGCN, dataset: pd.DataFrame
) -> pd.DataFrame | pd.Series:
    if uses_metadata_view(model):
        metadata = (
            dataset["metadata"]
            if "metadata" in dataset
            else pd.Series("no metadata", index=dataset.index)
        )
        return pd.DataFrame(
            {"statement": dataset["statement"], "metadata": metadata}
        )
    return dataset["statement"]


def split_dataset(
    dataset: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    groups = dataset["publisher"].fillna("Unknown source").astype(str)
    if groups.nunique() >= 8:
        split_count = min(groups.nunique(), max(2, round(1 / test_size)))
        splitter = StratifiedGroupKFold(
            n_splits=split_count,
            shuffle=True,
            random_state=random_state,
        )
        class_distribution = dataset["label"].value_counts(normalize=True)
        best_split: tuple[float, list[int], list[int]] | None = None
        for train_indices, holdout_indices in splitter.split(
            dataset, dataset["label"], groups
        ):
            holdout = dataset.iloc[holdout_indices]
            holdout_counts = holdout["label"].value_counts().reindex(
                class_distribution.index, fill_value=0
            )
            if holdout_counts.min() < 2:
                continue
            observed_distribution = holdout_counts / len(holdout)
            score = abs(len(holdout) / len(dataset) - test_size) + float(
                (observed_distribution - class_distribution).abs().sum()
            )
            if best_split is None or score < best_split[0]:
                best_split = (
                    score,
                    train_indices.tolist(),
                    holdout_indices.tolist(),
                )
        if best_split is not None:
            return (
                dataset.iloc[best_split[1]].copy(),
                dataset.iloc[best_split[2]].copy(),
                True,
            )

    train_data, holdout_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["label"],
    )
    return train_data.copy(), holdout_data.copy(), False


def predict_texts(
    model: Pipeline | DualViewAttentionGCN,
    statements: list[str],
    metadata: list[str] | None = None,
) -> pd.DataFrame:
    """Return model estimates and per-class scores for non-empty statements."""
    if uses_metadata_view(model):
        metadata_values = metadata or ["no metadata"] * len(statements)
        if len(metadata_values) != len(statements):
            raise ValueError("Provide one metadata value per statement.")
        prediction_input = pd.DataFrame(
            {
                "statement": statements,
                "metadata": [
                    value if value.strip() else "no metadata"
                    for value in metadata_values
                ],
            }
        )
    else:
        prediction_input = statements
    predictions = model.predict(prediction_input)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prediction_input)
    else:
        decision_scores = model.decision_function(prediction_input)
        if decision_scores.ndim == 1:
            decision_scores = np.column_stack(
                (-decision_scores, decision_scores)
            )
        probabilities = softmax(decision_scores, axis=1)
    results = pd.DataFrame(
        {
            "statement": statements,
            "predicted_category": predictions,
            "model_score": probabilities.max(axis=1),
        }
    )
    for index, label in enumerate(model.classes_):
        results[f"score_{label}"] = probabilities[:, index]
    return results


def test_mysql_connection(
    host: str,
    port: int,
    username: str,
    password: str,
    database_name: str,
) -> None:
    with pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database_name,
        connect_timeout=5,
        charset="utf8mb4",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM model_predictions LIMIT 0")
            cursor.execute("SELECT 1 FROM dataset_source_authors LIMIT 0")


@st.cache_data(ttl=30, max_entries=8, show_spinner=False)
def get_mysql_storage_status(
    host: str,
    port: int,
    username: str,
    password: str,
    database_name: str,
) -> dict[str, int | str]:
    """Return live connection identity and saved record counts."""
    with pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database_name,
        connect_timeout=5,
        charset="utf8mb4",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE(), CURRENT_USER(), VERSION()")
            database, account, version = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM model_predictions")
            prediction_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM dataset_source_authors")
            source_count = int(cursor.fetchone()[0])
    return {
        "database": str(database),
        "account": str(account),
        "version": str(version),
        "prediction_count": prediction_count,
        "source_count": source_count,
    }


def save_predictions_to_mysql(
    host: str,
    port: int,
    username: str,
    password: str,
    database_name: str,
    predictions: pd.DataFrame,
    model_name: str,
) -> int:
    """Persist explicitly submitted manual predictions to MySQL."""
    test_mysql_connection(host, port, username, password, database_name)
    score_columns = [
        column for column in predictions.columns if column.startswith("score_")
    ]
    records = [
        (
            str(row["statement"]),
            str(row["predicted_category"]),
            float(row["model_score"]),
            json.dumps(
                {
                    column.removeprefix("score_"): float(row[column])
                    for column in score_columns
                }
            ),
            model_name,
            datetime.now(timezone.utc).replace(tzinfo=None),
        )
        for row in predictions.to_dict(orient="records")
    ]
    with pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database_name,
        connect_timeout=5,
        charset="utf8mb4",
        autocommit=False,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO model_predictions (
                    statement,
                    predicted_category,
                    model_score,
                    class_scores,
                    model_name,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                records,
            )
        connection.commit()
    get_mysql_storage_status.clear()
    return len(records)


def save_dataset_source_authors_to_mysql(
    host: str,
    port: int,
    username: str,
    password: str,
    database_name: str,
    dataset: pd.DataFrame,
    dataset_name: str,
) -> int:
    """Persist labeled dataset rows with their source and author metadata."""
    records = []
    for row_index, row in enumerate(dataset.to_dict(orient="records")):
        record_values = {
            "dataset_name": dataset_name,
            "source_record_id": str(row.get("id", "") or ""),
            "author": str(row.get("author", "") or ""),
            "publisher": str(row.get("publisher", "") or "Unknown source"),
            "label": str(row.get("label", "") or ""),
            "statement": str(row.get("statement", "") or ""),
            "metadata": str(row.get("metadata", "") or ""),
            "row_index": row_index,
        }
        record_hash = hashlib.sha256(
            json.dumps(
                record_values,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            (
                record_hash,
                record_values["dataset_name"],
                record_values["source_record_id"],
                record_values["author"],
                record_values["publisher"],
                record_values["label"],
                record_values["statement"],
                record_values["metadata"],
                datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )

    with pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database_name,
        connect_timeout=5,
        charset="utf8mb4",
        autocommit=False,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM dataset_source_authors LIMIT 0")
            cursor.executemany(
                """
                INSERT IGNORE INTO dataset_source_authors (
                    record_hash,
                    dataset_name,
                    source_record_id,
                    author,
                    publisher,
                    label,
                    statement,
                    metadata,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )
            inserted_count = cursor.rowcount
        connection.commit()
    get_mysql_storage_status.clear()
    return inserted_count


def dataset_details(
    dataset: pd.DataFrame,
    model: Pipeline | DualViewAttentionGCN,
    dataset_name: str,
) -> pd.DataFrame:
    """Summarize the loaded dataset and the features actually used by the model."""
    tfidf = (
        model
        if isinstance(model, DualViewAttentionGCN)
        else model.named_steps["tfidf"]
    )
    feature_count = len(tfidf.get_feature_names_out())
    dual_view = uses_metadata_view(model)
    metadata_available = (
        "metadata" in dataset and dataset["metadata"].ne("no metadata").any()
    )
    return pd.DataFrame(
        [
            {
                "Dataset name": dataset_name,
                "#Samples": len(dataset),
                "#Views": (
                    "Text + metadata"
                    if dual_view and metadata_available
                    else "1 text view"
                ),
                "Feature dimensions": (
                    f"Combined TF-IDF features: {feature_count:,}"
                    if dual_view
                    else f"TF-IDF text: {feature_count:,}"
                ),
                "#Classes": dataset["label"].nunique(),
                "Data type": "Text + optional context metadata",
            }
        ]
    )


def inspect_text_features(
    model: Pipeline | DualViewAttentionGCN,
    statement: str,
    metadata: str | None = None,
) -> tuple[list[str], pd.DataFrame, int]:
    if isinstance(model, DualViewAttentionGCN):
        return model.inspect_features(statement, metadata)

    vectorizer = model.named_steps["tfidf"]
    if uses_metadata_view(model):
        text_union = vectorizer.transformers_[0][1]
        word_vectorizer = text_union.transformer_list[0][1]
        feature_input = pd.DataFrame(
            {
                "statement": [statement],
                "metadata": [metadata or "no metadata"],
            }
        )
    elif isinstance(vectorizer, FeatureUnion):
        word_vectorizer = vectorizer.transformer_list[0][1]
        feature_input = [statement]
    else:
        word_vectorizer = vectorizer
        feature_input = [statement]
    tokens = word_vectorizer.build_analyzer()(statement)
    transformed_features = vectorizer.transform(feature_input)
    feature_row = (
        transformed_features.tocsr()
        if issparse(transformed_features)
        else csr_matrix(transformed_features)
    )
    feature_names = vectorizer.get_feature_names_out()
    top_indices = sorted(
        zip(feature_row.indices, feature_row.data),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    top_features = pd.DataFrame(
        [
            {"Feature": feature_names[index], "TF-IDF weight": weight}
            for index, weight in top_indices
        ]
    )
    return tokens, top_features, feature_row.nnz


def render_prediction_color_key() -> None:
    categories = [*CLASS_ORDER, "Online article"]
    figure = go.Figure()
    for index, category in enumerate(categories):
        figure.add_trace(
            go.Scatter(
                x=[index],
                y=[0],
                mode="markers+text",
                marker={
                    "color": GRAPH_COLORS[category],
                    "size": 18,
                    "symbol": "square",
                },
                text=[category],
                textposition="top center",
                textfont={"size": 13},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        height=85,
        margin={"l": 8, "r": 8, "t": 26, "b": 0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [-0.5, len(categories) - 0.5]},
        yaxis={"visible": False, "range": [-0.6, 0.6]},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        alt="Color key: False is pink, Suspicious is amber, True is green, and Online article is cyan",
    )


def render_label_counts_chart(counts: pd.DataFrame, alt: str) -> None:
    figure = go.Figure(
        go.Bar(
            x=counts["Category"],
            y=counts["Records"],
            marker_color=[
                GRAPH_COLORS.get(category, "#8b92a0")
                for category in counts["Category"]
            ],
            hovertemplate="Category: %{x}<br>Records: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Category",
        yaxis_title="Records",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(figure, width="stretch", alt=alt)


def main() -> None:
    st.set_page_config(page_title="Fake News Text Analyzer", page_icon="📰", layout="wide")
    logo_path = Path(__file__).resolve().parent / "assets" / "fake_news_logo.svg"
    st.sidebar.image(
        str(logo_path),
        width=220,
        alt="Fake News Text Analyzer research dashboard logo",
    )
    st.sidebar.caption("© 2026 Mupenzi Damian · PhD student")
    require_dashboard_login()
    st.image(
        str(logo_path),
        width=300,
        alt="Fake News Text Analyzer research dashboard logo",
    )
    st.title("Fake News Text Analyzer")
    st.write(
        "Compare text-only baselines with a dual-view text-and-metadata model, "
        "review held-out performance, and estimate statement labels."
    )
    st.info(
        "**Prediction context:** The model classifies statements using patterns "
        "learned from labeled training examples. Its outputs are model estimates, "
        "not independent fact checks."
    )
    st.caption("Prediction and graph color key")
    render_prediction_color_key()

    with st.sidebar:
        st.header("XAMPP MySQL")
        try:
            mysql_config = st.secrets.get("connections", {}).get("mysql", {})
        except StreamlitSecretNotFoundError:
            mysql_config = {}
        mysql_host = str(mysql_config.get("host", "127.0.0.1"))
        mysql_port = int(mysql_config.get("port", 3306))
        mysql_username = str(mysql_config.get("username", "fake_news_app"))
        mysql_password = str(mysql_config.get("password", ""))
        mysql_database = str(
            mysql_config.get("database", "fake_news_modeling")
        )
        if st.button("Refresh XAMPP MySQL status"):
            get_mysql_storage_status.clear()
        try:
            mysql_status = get_mysql_storage_status(
                mysql_host,
                mysql_port,
                mysql_username,
                mysql_password,
                mysql_database,
            )
        except pymysql.MySQLError as error:
            log_failure("refresh XAMPP MySQL status", error)
            st.error(
                "XAMPP MySQL is not connected "
                f"({type(error).__name__}): {error}. Start MySQL in XAMPP "
                "and check the configured account and table permissions."
            )
        else:
            st.success(
                f"Connected to XAMPP MariaDB `{mysql_status['database']}` "
                f"as `{mysql_status['account']}`."
            )
            st.caption(
                f"Stored in XAMPP: "
                f"{mysql_status['source_count']:,} dataset/source records; "
                f"{mysql_status['prediction_count']:,} manual prediction records."
            )
        if st.button("Test XAMPP MySQL connection"):
            try:
                test_mysql_connection(
                    mysql_host,
                    mysql_port,
                    mysql_username,
                    mysql_password,
                    mysql_database,
                )
            except (pymysql.MySQLError, ValueError) as error:
                log_failure("test XAMPP MySQL connection", error)
                st.error(
                    "Could not connect to XAMPP MySQL "
                    f"({type(error).__name__}): {error}. Start MySQL in the XAMPP "
                    "Control Panel and check the host, port, username, "
                    "password, and database permissions."
                )
            else:
                st.success(
                    f"Connected to MySQL database `{mysql_database}`; "
                    "prediction and dataset-source tables are available."
                )
        st.caption(
            f"Target: `{mysql_username}@{mysql_host}:{mysql_port}` / "
            f"`{mysql_database}`. The account only needs access to the "
            "`model_predictions` and `dataset_source_authors` tables."
        )

        st.header("Training data")
        uploaded_file = st.file_uploader(
            "Upload a LIAR ZIP, TSV/CSV, or labeled text dataset",
            type=["zip", "tsv", "csv", "txt"],
            help=(
                "For a LIAR ZIP, the app uses its train.tsv file. Also supports "
                "standard 14-column LIAR files or delimited files with a header "
                "identifying label/class and statement/text columns. "
                "Maximum file size: 500 MB."
            ),
        )
        default_path = Path("data") / "train.tsv"
        desktop_liar_archive = (
            Path.home()
            / "Desktop"
            / "PHD new version"
            / "PHD new project"
            / "liar_dataset-master.zip"
        )
        desktop_dataset = Path.home() / "Desktop" / "news_articles.csv"
        if not default_path.is_file():
            if desktop_liar_archive.is_file():
                default_path = desktop_liar_archive
            elif desktop_dataset.is_file():
                default_path = desktop_dataset
        use_default = not uploaded_file and default_path.is_file()
        if use_default:
            data_source_label = (
                "Using local LIAR dataset archive (train.tsv split)"
                if default_path.suffix.lower() == ".zip"
                else "Using local dataset"
            )
            st.caption(f"{data_source_label}: {default_path}")
        elif not uploaded_file:
            st.caption("Upload a dataset to enable training.")

        train_clicked = st.button(
            "Train and evaluate",
            type="primary",
            disabled=not uploaded_file and not use_default,
        )

    if train_clicked:
        try:
            raw_bytes = (
                uploaded_file.getvalue()
                if uploaded_file
                else default_path.read_bytes()
            )
            dataset_filename = uploaded_file.name if uploaded_file else default_path.name
            dataset = load_dataset_file(raw_bytes, dataset_filename)
            counts = dataset["label"].value_counts()
            if len(dataset) < 10 or len(counts) < 2 or counts.min() < 2:
                raise ValueError(
                    "The dataset must contain at least 10 valid statements and at "
                    "least 2 examples in each of at least two recognized classes."
                )

            development_dataset, test_dataset, group_test_split = split_dataset(
                dataset, test_size=0.2, random_state=42
            )
            has_validation_split = (
                len(dataset) >= 30 and counts.min() >= 5
            )
            group_validation_split = False
            if has_validation_split:
                train_dataset, validation_dataset, group_validation_split = (
                    split_dataset(
                        development_dataset,
                        test_size=0.125,
                        random_state=43,
                    )
                )
            else:
                train_dataset = development_dataset
                validation_dataset = None
            trained_models: dict[str, Pipeline | DualViewAttentionGCN] = {}
            model_metrics: dict[str, dict[str, object]] = {}
            for model_name in MODEL_NAMES:
                candidate = create_model(model_name)
                candidate.fit(
                    model_input(candidate, train_dataset),
                    train_dataset["label"],
                )
                if validation_dataset is not None:
                    validation_predictions = candidate.predict(
                        model_input(candidate, validation_dataset)
                    )
                    validation_macro_f1 = f1_score(
                        validation_dataset["label"],
                        validation_predictions,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                    candidate.fit(
                        model_input(candidate, development_dataset),
                        development_dataset["label"],
                    )
                else:
                    validation_macro_f1 = None
                predictions = candidate.predict(
                    model_input(candidate, test_dataset)
                )
                report = classification_report(
                    test_dataset["label"],
                    predictions,
                    labels=CLASS_ORDER,
                    output_dict=True,
                    zero_division=0,
                )
                model_metrics[model_name] = {
                    "validation_macro_f1": validation_macro_f1,
                    "accuracy": accuracy_score(test_dataset["label"], predictions),
                    "macro_precision": precision_score(
                        test_dataset["label"],
                        predictions,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    ),
                    "macro_recall": recall_score(
                        test_dataset["label"],
                        predictions,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    ),
                    "micro_f1": f1_score(
                        test_dataset["label"],
                        predictions,
                        labels=CLASS_ORDER,
                        average="micro",
                        zero_division=0,
                    ),
                    "macro_f1": f1_score(
                        test_dataset["label"],
                        predictions,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    ),
                    "report": report,
                    "dataset_size": len(dataset),
                }
                trained_models[model_name] = candidate

            selected_model_name = max(
                MODEL_NAMES,
                key=lambda name: (
                    model_metrics[name]["validation_macro_f1"]
                    if validation_dataset is not None
                    else model_metrics[name]["macro_f1"]
                ),
            )
            st.session_state["trained_models"] = trained_models
            st.session_state["model_metrics"] = model_metrics
            st.session_state["selected_model_name"] = selected_model_name
            st.session_state.pop("active_prediction_model", None)
            st.session_state["model"] = trained_models[selected_model_name]
            st.session_state["training_dataset"] = dataset
            st.session_state.pop("analysis_export", None)
            st.session_state["training_dataset_name"] = (
                "LIAR (train split)"
                if "liar" in dataset_filename.casefold()
                else Path(dataset_filename).stem
            )
            st.session_state.pop("online_articles", None)
            st.session_state.pop("online_articles_model_name", None)
            st.session_state.pop("manual_predictions", None)
            st.session_state.pop("manual_prediction_metadata", None)
            st.session_state["metrics"] = model_metrics[selected_model_name]
            st.session_state["used_validation_split"] = (
                validation_dataset is not None
            )
            st.session_state["used_group_test_split"] = group_test_split
            st.session_state["used_group_validation_split"] = (
                group_validation_split
            )
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            log_failure("load dataset and train models", error)
            st.error(
                f"Could not load the dataset and train the models "
                f"({type(error).__name__}): {error}"
            )
            return

    trained_models = st.session_state.get("trained_models")
    if trained_models and any(name not in trained_models for name in MODEL_NAMES):
        st.session_state.pop("trained_models", None)
        st.session_state.pop("model_metrics", None)
        st.session_state.pop("selected_model_name", None)
        st.session_state.pop("model", None)
        st.session_state.pop("metrics", None)
        st.info(
            "The available model set has changed. Select **Train and evaluate** "
            "to compare the updated models."
        )
        return
    if trained_models:
        default_model_name = st.session_state.get(
            "selected_model_name", MODEL_NAMES[0]
        )
        if default_model_name not in MODEL_NAMES:
            default_model_name = MODEL_NAMES[0]
        selected_model_name = st.selectbox(
            "Prediction model",
            options=MODEL_NAMES,
            index=MODEL_NAMES.index(default_model_name),
            key="active_prediction_model",
            help=(
                "Choose which trained model powers headline and manual "
                "predictions. The default is selected by validation macro-F1."
            ),
        )
        model = trained_models[selected_model_name]
        st.session_state["selected_model_name"] = selected_model_name
        st.session_state["model"] = model
        st.session_state["metrics"] = st.session_state["model_metrics"][
            selected_model_name
        ]
    else:
        model = st.session_state.get("model")
    if model is None:
        if use_default:
            st.info(
                f"Ready to train on **{default_path.name}**. Select "
                "**Train and evaluate** in the sidebar."
            )
        else:
            st.info(
                "Upload a labeled dataset and select **Train and evaluate** "
                "to get started."
            )
        return

    prediction_model_name = st.session_state.get(
        "selected_model_name", "Legacy model"
    )
    st.subheader("News analysis dashboard")
    st.write(
        "Explore your labeled dataset and scraped headlines as separate "
        "contributions. Scraped headline categories are model estimates, not "
        "fact-checks."
    )
    selected_sources = st.pills(
        "Headline sources",
        options=list(NEWS_SOURCES),
        selection_mode="multi",
        default=[
            "Africa Intelligence",
            "Al Jazeera",
            "Voice of America",
            "IGIHE",
            "New Vision",
            "BBC News",
        ],
    )
    if st.button(
        "Fetch latest headlines",
        key="fetch_news_headlines",
        disabled=not selected_sources,
    ):
        fetched_articles = []
        source_errors = []
        for source_name in selected_sources:
            try:
                fetched_articles.append(fetch_source_articles(source_name))
            except ValueError as error:
                log_failure(f"fetch headlines from {source_name}", error)
                source_errors.append(f"{source_name}: {error}")

        st.session_state["news_source_errors"] = source_errors
        if fetched_articles:
            online_articles = pd.concat(fetched_articles, ignore_index=True)
            predictions = predict_texts(
                model,
                online_articles["statement"].tolist(),
                metadata=["no metadata"] * len(online_articles),
            )
            online_articles = pd.concat(
                [
                    online_articles.reset_index(drop=True),
                    predictions.drop(columns="statement"),
                ],
                axis=1,
            )
            st.session_state["online_articles"] = online_articles
            st.session_state["online_articles_model_name"] = prediction_model_name
        else:
            st.session_state["online_articles"] = pd.DataFrame(
                columns=["statement", "label", "publisher", "url", "node_type"]
            )
            st.session_state["online_articles_model_name"] = prediction_model_name

    online_articles = st.session_state.get("online_articles")
    if (
        online_articles is not None
        and not online_articles.empty
        and st.session_state.get("online_articles_model_name")
        != prediction_model_name
    ):
        model_predictions = predict_texts(
            model,
            online_articles["statement"].astype(str).tolist(),
            metadata=["no metadata"] * len(online_articles),
        ).drop(columns="statement")
        stale_prediction_columns = [
            column
            for column in online_articles.columns
            if column in {"predicted_category", "model_score"}
            or column.startswith("score_")
        ]
        online_articles = pd.concat(
            [
                online_articles.drop(columns=stale_prediction_columns).reset_index(
                    drop=True
                ),
                model_predictions.reset_index(drop=True),
            ],
            axis=1,
        )
        st.session_state["online_articles"] = online_articles
        st.session_state["online_articles_model_name"] = prediction_model_name
    for source_error in st.session_state.get("news_source_errors", []):
        st.warning(source_error)
    if online_articles is None:
        online_articles = pd.DataFrame(
            columns=[
                "statement",
                "label",
                "publisher",
                "url",
                "node_type",
                "predicted_category",
                "model_score",
            ]
        )
        st.info("Select one or more sources and fetch headlines to add their contribution.")
    elif online_articles.empty:
        st.info("No headlines were fetched. Check the source messages above and try again.")

    training_dataset = st.session_state.get("training_dataset")
    if training_dataset is None:
        st.error("Training dataset is missing. Train the model again to rebuild it.")
        return
    metadata_coverage = (
        training_dataset["metadata"].ne("no metadata").mean()
        if "metadata" in training_dataset
        else 0.0
    )

    (
        dashboard_tab,
        benchmark_tab,
        processing_tab,
        dataset_tab,
        scraped_tab,
        predictions_tab,
        relations_tab,
    ) = st.tabs(
        [
            "Overview",
            "Research benchmarks",
            "Text processing",
            "Dataset analysis",
            "Scraped contribution",
            "Predictions",
            "Relationships",
        ]
    )
    with dashboard_tab:
        with st.container(horizontal=True):
            st.metric("Dataset records", f"{len(training_dataset):,}", border=True)
            st.metric("Scraped headlines", f"{len(online_articles):,}", border=True)
            st.metric(
                "Scraped sources",
                f"{online_articles['publisher'].nunique():,}",
                border=True,
            )
            metrics = st.session_state.get("metrics")
            if metrics:
                st.metric("Model accuracy", f"{metrics['accuracy']:.1%}", border=True)
            else:
                st.metric("Model accuracy", "Not available", border=True)

        dataset_counts = (
            training_dataset["label"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Records")
        )
        chart_col, evaluation_col = st.columns(2)
        with chart_col:
            with st.container(border=True):
                st.subheader("Existing dataset labels")
                render_label_counts_chart(
                    dataset_counts,
                    "Number of labeled records in the existing dataset, "
                    "colored by category",
                )
        with evaluation_col:
            with st.container(border=True):
                st.subheader("Held-out model evaluation")
                if metrics:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                    st.metric("Micro-F1", f"{metrics['micro_f1']:.1%}")
                    report = metrics["report"]
                    report_table = pd.DataFrame(
                        {
                            label: {
                                "Precision": report[label]["precision"],
                                "Recall": report[label]["recall"],
                                "F1-score": report[label]["f1-score"],
                                "Support": int(report[label]["support"]),
                            }
                            for label in CLASS_ORDER
                        }
                    ).T
                    st.dataframe(
                        report_table.style.format(
                            {
                                "Precision": "{:.3f}",
                                "Recall": "{:.3f}",
                                "F1-score": "{:.3f}",
                            }
                        ),
                        alt="Held-out precision, recall, F1-score, and support by category",
                    )
                else:
                    st.info("Train the model to view held-out evaluation metrics.")

        model_metrics = st.session_state.get("model_metrics")
        if model_metrics:
            st.subheader("Text, dual-view, and dual-view GCN comparison")
            st.caption(
                "Models include Logistic Regression, Multinomial and "
                "Complement Naive Bayes, SGD, a word/character Linear SVM, "
                "the dual-view word/character + context model, and a proposed "
                "two-view GCN with attention fusion and supervised contrastive "
                "training. Separate text and metadata feature graphs use five "
                "nearest training examples; held-out examples connect only to "
                "training nodes. Graphs are built from features, never labels. "
                "All models use the same source-aware split where available. Validation "
                "macro-F1 selects the default; independent holdout metrics "
                "are displayed separately."
            )
            comparison = pd.DataFrame(
                [
                    {
                        "Model": model_name,
                        "Contribution": (
                            MODEL_DESCRIPTIONS[model_name]
                            + (
                                " (no usable context metadata; text-only)"
                                if model_name == "Dual-view TF-IDF + metadata"
                                and metadata_coverage == 0
                                else ""
                            )
                        ),
                        "Validation Macro-F1": model_metrics[model_name][
                            "validation_macro_f1"
                        ],
                        "Accuracy": model_metrics[model_name]["accuracy"],
                        "Macro precision": model_metrics[model_name]["macro_precision"],
                        "Macro recall": model_metrics[model_name]["macro_recall"],
                        "Macro-F1": model_metrics[model_name]["macro_f1"],
                        "Micro-F1": model_metrics[model_name]["micro_f1"],
                    }
                    for model_name in MODEL_NAMES
                ]
            )
            st.dataframe(
                comparison.style.format(
                    {
                        "Validation Macro-F1": (
                            lambda score: f"{score:.1%}"
                            if pd.notna(score)
                            else "Not available"
                        ),
                        "Accuracy": "{:.1%}",
                        "Macro precision": "{:.1%}",
                        "Macro recall": "{:.1%}",
                        "Macro-F1": "{:.1%}",
                        "Micro-F1": "{:.1%}",
                    }
                ),
                hide_index=True,
                alt="Model selection and independent holdout comparison of accuracy, macro precision, macro recall, macro-F1, and micro-F1 across seven models",
            )
            if st.session_state.get("used_validation_split"):
                st.caption(
                    "Selection uses a validation slice from the development set; "
                    "holdout metrics come from a separate 20% test split."
                )
                if st.session_state.get("used_group_test_split"):
                    st.caption(
                        "Publisher/speaker groups are kept separate between "
                        "development and test, providing a stronger check on "
                        "generalization to unseen sources or speakers."
                    )
                else:
                    st.caption(
                        "There are not enough distinct publishers/speakers for "
                        "a group-held-out test; unseen-source performance is "
                        "not measured."
                    )
            else:
                st.warning(
                    "This dataset is too small for a reliable separate validation "
                    "split. The displayed holdout macro-F1 is used for model "
                    "selection; performance estimates may be unstable."
                )
            st.subheader("Download analysis package")
            st.caption(
                "The ZIP includes the labeled dataset, measured model metrics, "
                "interactive HTML graphs, and trained model files. If headlines "
                "were fetched, their records and model estimates are included "
                "separately. Graphs load Plotly.js from a CDN."
            )
            if st.button(
                "Prepare analysis, models, graphs, and dataset ZIP",
                key="prepare_analysis_export",
            ):
                try:
                    st.session_state["analysis_export"] = build_analysis_export(
                        training_dataset,
                        st.session_state.get(
                            "training_dataset_name", "Loaded labeled dataset"
                        ),
                        st.session_state.get("model_metrics", {}),
                        st.session_state.get("trained_models", {}),
                        online_articles,
                    )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    pickle.PicklingError,
                ) as error:
                    log_failure("prepare analysis export", error)
                    st.session_state.pop("analysis_export", None)
                    st.error(
                        "Could not prepare the analysis download "
                        f"({type(error).__name__}): {error}"
                    )
            if "analysis_export" in st.session_state:
                st.download_button(
                    "Download full analysis ZIP",
                    data=st.session_state["analysis_export"],
                    file_name="fake_news_analysis_export.zip",
                    mime="application/zip",
                    key="download_analysis_export",
                )

    with benchmark_tab:
        st.subheader("Reported internal architectural ablation")
        st.caption(
            "The supplied results contain no values for these four ablation "
            "rows, so they are intentionally shown as not reported."
        )
        st.dataframe(
            pd.DataFrame(REPORTED_INTERNAL_ABLATIONS),
            hide_index=True,
            alt="Internal model ablation variants; missing performance values are marked not reported",
        )
        st.info(
            "The values above are architectural ablation results supplied for "
            "this project; missing measurements remain marked as not reported. "
            "Local model measurements are available in Overview. Scraped "
            "headlines are unlabeled and are used for prediction/source analysis "
            "only; model-generated labels are not treated as ground truth."
        )

    with processing_tab:
        st.subheader("How text becomes a model prediction")
        is_attention_gcn = isinstance(model, DualViewAttentionGCN)
        if is_attention_gcn:
            st.caption(
                "The dual-view GCN encodes text and non-identifying context "
                "metadata separately, builds a feature-only graph for each "
                "view, and learns attention weights to fuse the graph "
                "representations. Supervised contrastive loss is used only "
                "during training on labeled training examples. No pretrained "
                "SBERT embeddings are used."
            )
        else:
            st.caption(
                "Text is tokenized into sparse TF-IDF features. The metadata "
                "baseline uses available non-identifying LIAR context; "
                "speaker and publisher identity are excluded. This selected "
                "model does not use graph convolution or supervised "
                "contrastive training."
            )
        if uses_metadata_view(model) and metadata_coverage == 0:
            st.info(
                "This dataset has no eligible context metadata. The metadata "
                "view contains only a placeholder and cannot contribute "
                "meaningful contextual information."
            )
        processing_steps = [
            ("1. Text", "Original statement or headline"),
            ("2. Tokens", "Word unigrams/bigrams and character n-grams"),
            ("3. Features", "Word and character TF-IDF"),
            (
                "4. Metadata view",
                "Non-identifying context TF-IDF"
                if uses_metadata_view(model)
                else "Not used by this model",
            ),
            ("5. Graph + fusion", "Per-view GCN + attention + SupCon")
            if is_attention_gcn
            else ("5. Classifier", st.session_state.get("selected_model_name", "Selected model")),
            ("6. Estimate", st.session_state.get("selected_model_name", "Selected model")),
        ]
        step_columns = st.columns(len(processing_steps))
        for column, (title, description) in zip(step_columns, processing_steps):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(description)
        vectorizer = (
            model
            if isinstance(model, DualViewAttentionGCN)
            else model.named_steps["tfidf"]
        )
        st.write(
            f"The fitted model uses **{len(vectorizer.get_feature_names_out()):,}** "
            "TF-IDF vocabulary features across its available views."
        )
        preview_index = st.slider(
            "Choose a loaded dataset row to inspect",
            min_value=1,
            max_value=len(training_dataset),
            value=1,
            key="text_processing_row",
        )
        sample = training_dataset.iloc[preview_index - 1]
        sample_text = str(sample["statement"])
        sample_metadata = str(sample.get("metadata", "no metadata"))
        tokens, top_features, nonzero_features = inspect_text_features(
            model, sample_text, metadata=sample_metadata
        )
        st.text_area(
            "Original statement",
            value=sample_text,
            height=110,
            disabled=True,
            key="text_processing_statement",
        )
        token_col, feature_col = st.columns(2)
        with token_col:
            st.metric("Analyzer tokens / n-grams", f"{len(tokens):,}")
            st.caption(
                "Preview of the exact analyzer output (unigrams and adjacent "
                "bigrams), capped at 80 items."
            )
            st.code("  ".join(tokens[:80]) or "(no tokens recognized)")
        with feature_col:
            st.metric("Non-zero TF-IDF features", f"{nonzero_features:,}")
            if top_features.empty:
                st.info(
                    "No vocabulary features matched this statement. The model "
                    "will predict using its learned class priors."
                )
            else:
                st.dataframe(
                    top_features.style.format({"TF-IDF weight": "{:.4f}"}),
                    hide_index=True,
                    alt="Highest-weight TF-IDF features for the selected dataset statement",
                )
                feature_figure = go.Figure(
                    go.Bar(
                        x=top_features["TF-IDF weight"],
                        y=top_features["Feature"],
                        orientation="h",
                        marker_color="#00D4FF",
                        hovertemplate="%{y}<br>TF-IDF: %{x:.4f}<extra></extra>",
                    )
                )
                feature_figure.update_layout(
                    xaxis_title="TF-IDF weight",
                    yaxis={"autorange": "reversed"},
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    height=320,
                )
                st.plotly_chart(
                    feature_figure,
                    width="stretch",
                    alt="Top TF-IDF features for the selected statement, ranked by weight",
                )
        example_prediction = predict_texts(
            model, [sample_text], metadata=[sample_metadata]
        ).iloc[0]
        prediction_summary = pd.DataFrame(
            [
                {
                    "Dataset label": sample["label"],
                    "Model estimate": example_prediction["predicted_category"],
                    "Top model score": f"{example_prediction['model_score']:.1%}",
                }
            ]
        )
        st.dataframe(
            prediction_summary,
            hide_index=True,
            alt="Selected statement's dataset label, predicted category, and top model score",
        )
        st.caption(
            "The top model score is not a calibrated probability. This row is "
            "from the training dataset, so its prediction is an inspection "
            "example—not an independent evaluation."
        )

    with dataset_tab:
        st.subheader("Dataset details")
        dataset_name = st.session_state.get(
            "training_dataset_name", "Loaded labeled dataset"
        )
        st.subheader("Store dataset source and author records")
        st.caption(
            "This stores loaded labeled dataset rows, including statement text, "
            "label, author/speaker, publisher/source, and available context. "
            "It does not save scraped headlines or model-generated labels."
        )
        dataset_save_consent = st.checkbox(
            "I agree to store this dataset's statement text and source/author "
            "metadata in XAMPP MySQL.",
            key="dataset_source_save_consent",
        )
        if st.button(
            "Save dataset source and author records to XAMPP MySQL",
            disabled=not dataset_save_consent,
            key="save_dataset_source_authors",
        ):
            try:
                saved_dataset_rows = save_dataset_source_authors_to_mysql(
                    mysql_host,
                    mysql_port,
                    mysql_username,
                    mysql_password,
                    mysql_database,
                    training_dataset,
                    dataset_name,
                )
            except (pymysql.MySQLError, ValueError) as error:
                log_failure("save dataset source and author records", error)
                st.error(
                    "Could not save dataset source and author records to "
                    f"XAMPP MySQL ({type(error).__name__}): {error}. Check the "
                    "connection and dataset_source_authors table permissions."
                )
            else:
                st.success(
                    f"Saved {saved_dataset_rows:,} new row(s) to "
                    f"`{mysql_database}.dataset_source_authors`. Identical "
                    "rows already stored were skipped."
                )
        details = dataset_details(training_dataset, model, dataset_name)
        st.dataframe(
            details,
            hide_index=True,
            alt="Dataset name, sample count, model views, feature dimensions, classes, and data type",
        )
        st.caption(
            "Feature dimensions report the trained TF-IDF vocabulary. Publisher "
            "metadata is a model input only for the dual-view models. The "
            "proposed graph model learns a compact TF-IDF projection at train "
            "time; no pretrained SBERT embeddings are loaded."
        )
        st.metric(
            "Rows with model metadata",
            f"{metadata_coverage:.1%}",
            help="A row counts as available when the dataset provides context fields such as subject, party, job, or historical credibility counts. Speaker and publisher identity are excluded to avoid source-label leakage.",
        )
        st.subheader("Reference dataset specifications")
        st.dataframe(
            pd.DataFrame(REFERENCE_DATASET_SPECS),
            hide_index=True,
            alt="User-provided sample, view, feature, class, and data-type reference specifications",
        )
        st.caption(
            "These comparison figures were supplied as a reference and are not "
            "verified against files loaded by this app. The loaded LIAR train "
            "split is reported separately above."
        )

        st.subheader("Existing dataset analysis")
        dataset_labels = sorted(training_dataset["label"].dropna().unique().tolist())
        selected_labels = st.multiselect(
            "Filter dataset labels",
            options=dataset_labels,
            default=dataset_labels,
            key="dashboard_dataset_labels",
        )
        filtered_dataset = training_dataset[
            training_dataset["label"].isin(selected_labels)
        ]
        st.metric("Records in selected labels", f"{len(filtered_dataset):,}")
        source_counts = (
            filtered_dataset["publisher"]
            .fillna("Unknown source")
            .replace("", "Unknown source")
            .value_counts()
            .head(15)
            .rename_axis("Source")
            .reset_index(name="Records")
        )
        label_counts = (
            filtered_dataset["label"]
            .value_counts()
            .rename_axis("Label")
            .reset_index(name="Records")
        )
        source_chart, label_chart = st.columns(2)
        with source_chart:
            st.bar_chart(
                source_counts,
                x="Source",
                y="Records",
                alt="Top publishers represented in the existing labeled dataset",
            )
        with label_chart:
            render_label_counts_chart(
                label_counts.rename(columns={"Label": "Category"}),
                "Existing dataset records by label, colored by category",
            )
        st.dataframe(
            filtered_dataset[["publisher", "label", "statement"]].head(100),
            hide_index=True,
            alt="First 100 filtered records from the existing labeled dataset",
        )

    with scraped_tab:
        st.subheader("Scraped headline contribution")
        st.caption(
            "Each headline is kept separate from the labeled training dataset. "
            "The dual-view model uses headline text and only metadata supplied "
            "for that prediction; scraped headlines do not receive publisher "
            "identity as a model feature."
        )
        if online_articles.empty:
            st.info(
                "Choose the headline sources above, then select **Fetch latest "
                "headlines** to see source contributions and model estimates."
            )
        else:
            scraped_sources = sorted(online_articles["publisher"].unique().tolist())
            filtered_sources = st.multiselect(
                "Filter scraped sources",
                options=scraped_sources,
                default=scraped_sources,
                key="dashboard_scraped_sources",
            )
            filtered_scraped = online_articles[
                online_articles["publisher"].isin(filtered_sources)
            ]
            if "predicted_category" in filtered_scraped.columns:
                scraped_labels = sorted(
                    filtered_scraped["predicted_category"].dropna().unique().tolist()
                )
                selected_predictions = st.multiselect(
                    "Filter model estimates",
                    options=scraped_labels,
                    default=scraped_labels,
                    key="dashboard_scraped_labels",
                )
                filtered_scraped = filtered_scraped[
                    filtered_scraped["predicted_category"].isin(selected_predictions)
                ]

            source_counts = (
                filtered_scraped["publisher"]
                .value_counts()
                .rename_axis("Source")
                .reset_index(name="Headlines")
            )
            st.bar_chart(
                source_counts,
                x="Source",
                y="Headlines",
                alt="Scraped headline contribution by news source",
            )
            if (
                not filtered_scraped.empty
                and "predicted_category" in filtered_scraped.columns
            ):
                prediction_counts = filtered_scraped.groupby(
                    ["publisher", "predicted_category"], observed=True
                ).size()
                source_names = sorted(filtered_scraped["publisher"].unique())
                category_figure = go.Figure()
                for category in CLASS_ORDER:
                    category_figure.add_trace(
                        go.Bar(
                            name=category,
                            x=source_names,
                            y=[
                                int(prediction_counts.get((source, category), 0))
                                for source in source_names
                            ],
                            marker_color=GRAPH_COLORS[category],
                            hovertemplate=(
                                "Source: %{x}<br>"
                                f"Estimate: {category}<br>"
                                "Headlines: %{y}<extra></extra>"
                            ),
                        )
                    )
                category_figure.update_layout(
                    barmode="stack",
                    xaxis_title="News source",
                    yaxis_title="Number of headlines",
                    legend_title="Model estimate",
                    margin={"l": 10, "r": 10, "t": 20, "b": 10},
                )
                st.plotly_chart(
                    category_figure,
                    width="stretch",
                    alt=(
                        "Scraped headline counts per source, colored by model "
                        "estimate: False, Suspicious, or True"
                    ),
                )
                headline_results = filtered_scraped[
                    ["publisher", "statement", "predicted_category", "model_score", "url"]
                ].rename(
                    columns={
                        "publisher": "Source",
                        "statement": "Headline",
                        "predicted_category": "Model estimate",
                        "model_score": "Model score",
                        "url": "Article URL",
                    }
                )
                headline_results["Model score"] = headline_results[
                    "Model score"
                ].map(lambda score: f"{score:.1%}")
                st.dataframe(
                    headline_results,
                    hide_index=True,
                    alt="Scraped headlines with model estimates, scores, and source links",
                )

    with relations_tab:
        st.subheader("Publisher relationship graph")
        st.write(
            "Nodes represent records from the labeled dataset and scraped "
            "headlines. Edges connect records from the same publisher only; "
            "they do not imply shared claims or verified relationships. "
            "The graph uses publisher metadata, not token or TF-IDF similarity."
        )
        include_dataset = st.toggle(
            "Include existing dataset nodes",
            value=True,
            key="relationship_include_dataset",
        )
        relationship_sources = st.multiselect(
            "Include scraped sources",
            options=sorted(online_articles["publisher"].unique().tolist()),
            default=sorted(online_articles["publisher"].unique().tolist()),
            key="relationship_sources",
        )
        graph_dataset = training_dataset if include_dataset else training_dataset.iloc[0:0]
        graph_articles = online_articles[
            online_articles["publisher"].isin(relationship_sources)
        ]
        if graph_dataset.empty and graph_articles.empty:
            st.info("Enable dataset nodes or select a scraped source to display the graph.")
        else:
            render_article_graph(graph_dataset, graph_articles)

    with predictions_tab:
        st.subheader("Predict statement categories")
        st.write(
            "Enter one statement per line to get model estimates. Predictions "
            "reflect patterns learned from the labeled training data. The "
            "dual-view model can use one line of non-identifying context metadata "
            "per statement. Scores are model outputs, not fact checks."
        )
        with st.form("statement_prediction_form"):
            statements_text = st.text_area(
                "Statements",
                placeholder="Paste one statement or headline per line...",
                height=180,
                key="prediction_statements",
            )
            metadata_text = ""
            if uses_metadata_view(model) and metadata_coverage > 0:
                metadata_text = st.text_area(
                    "Optional context metadata (one line per statement)",
                    placeholder=(
                        "Party, subject, job, or context; leave blank "
                        "if unavailable."
                    ),
                    height=100,
                    help=(
                        "Keep metadata lines in the same order as statements. "
                        "Speaker and publisher identity are excluded to reduce "
                        "source-label leakage."
                    ),
                    key="prediction_metadata",
                )
            predict_clicked = st.form_submit_button(
                "Generate predictions",
                type="primary",
            )

        if predict_clicked:
            statements = [
                line.strip()
                for line in statements_text.splitlines()
                if line.strip()
            ]
            metadata_values = [
                line.strip() for line in metadata_text.splitlines()
            ]
            if not statements:
                st.warning("Enter at least one non-empty statement.")
            elif len(statements) > 100:
                st.error("Enter no more than 100 statements at a time.")
            elif metadata_text.strip() and len(metadata_values) != len(statements):
                st.error(
                    "When providing metadata, enter exactly one metadata line "
                    "for each non-empty statement."
                )
            else:
                st.session_state["manual_predictions"] = predict_texts(
                    model,
                    statements,
                    metadata=metadata_values if metadata_text.strip() else None,
                )
                st.session_state["manual_prediction_metadata"] = (
                    metadata_values if metadata_text.strip() else None
                )
                st.session_state["manual_predictions_model_name"] = (
                    prediction_model_name
                )

        prediction_results = st.session_state.get("manual_predictions")
        if prediction_results is not None:
            if (
                st.session_state.get("manual_predictions_model_name")
                != prediction_model_name
            ):
                prediction_results = predict_texts(
                    model,
                    prediction_results["statement"].astype(str).tolist(),
                    metadata=st.session_state.get("manual_prediction_metadata"),
                )
                st.session_state["manual_predictions"] = prediction_results
                st.session_state["manual_predictions_model_name"] = (
                    prediction_model_name
                )
            st.caption(
                "Model scores are classifier outputs and are not calibrated "
                "probabilities that a statement is true."
            )
            prediction_counts = (
                prediction_results["predicted_category"]
                .value_counts()
                .rename_axis("Model estimate")
                .reset_index(name="Statements")
            )
            with st.container(horizontal=True):
                st.metric("Statements predicted", f"{len(prediction_results):,}", border=True)
                st.metric(
                    "Categories predicted",
                    f"{prediction_results['predicted_category'].nunique():,}",
                    border=True,
                )
            st.bar_chart(
                prediction_counts,
                x="Model estimate",
                y="Statements",
                alt="Number of submitted statements assigned to each model category",
            )

            display_results = prediction_results.rename(
                columns={
                    "statement": "Statement",
                    "predicted_category": "Model estimate",
                    "model_score": "Top model score",
                    **{
                        f"score_{label}": f"Score: {label}"
                        for label in model.classes_
                    },
                }
            ).copy()
            score_columns = [
                column
                for column in display_results.columns
                if column == "Top model score" or column.startswith("Score: ")
            ]
            display_results[score_columns] = display_results[score_columns].map(
                lambda score: f"{score:.1%}"
            )
            st.dataframe(
                display_results,
                hide_index=True,
                alt="Model category estimates and per-category scores for submitted statements",
            )
            st.subheader("Save prediction records")
            save_consent = st.checkbox(
                "I understand the submitted statement text and model results "
                "will be stored in XAMPP MySQL.",
                key="prediction_save_consent",
            )
            if st.button(
                "Save current predictions to XAMPP MySQL",
                disabled=not save_consent,
                key="save_prediction_results",
            ):
                try:
                    saved_count = save_predictions_to_mysql(
                        mysql_host,
                        mysql_port,
                        mysql_username,
                        mysql_password,
                        mysql_database,
                        prediction_results,
                        st.session_state.get(
                            "manual_predictions_model_name",
                            prediction_model_name,
                        ),
                    )
                except (pymysql.MySQLError, ValueError) as error:
                    log_failure("save manual predictions", error)
                    st.error(
                        "Could not save prediction records to XAMPP MySQL "
                        f"({type(error).__name__}): {error}. Check the connection "
                        "settings, database permissions, and server availability."
                    )
                else:
                    st.success(
                        f"Saved {saved_count} prediction record(s) to "
                        f"`{mysql_database}.model_predictions`."
                    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log_failure("unexpected dashboard failure", error)
        st.error(
            "The dashboard stopped because of an unexpected error "
            f"({type(error).__name__}): {error}. The full traceback is "
            f"recorded in `{APP_LOG_PATH}`."
        )
