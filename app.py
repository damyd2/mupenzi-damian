from __future__ import annotations

import csv
from html import escape
from html.parser import HTMLParser
from io import StringIO
import math
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError
from sqlalchemy.exc import SQLAlchemyError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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
BBC_NEWS_URL = "https://www.bbc.com/news"
MAX_BBC_PAGE_BYTES = 5 * 1024 * 1024
MAX_BBC_ARTICLES = 20
MAX_GRAPH_DATASET_NODES = 180


class BBCArticleLinkParser(HTMLParser):
    """Extract article headline links from the BBC News landing page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.articles: dict[str, str] = {}
        self._active_url: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        article_url = urljoin(BBC_NEWS_URL, href)
        parsed_url = urlsplit(article_url)
        if (
            parsed_url.scheme == "https"
            and parsed_url.netloc.lower() == "www.bbc.com"
            and re.match(r"^/news/articles/", parsed_url.path)
        ):
            self._active_url = article_url
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_url:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_url:
            title = " ".join(" ".join(self._active_text).split())
            if len(title) > 15:
                self.articles.setdefault(self._active_url, title)
            self._active_url = None
            self._active_text = []


class RejectRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_bbc_articles() -> pd.DataFrame:
    """Fetch a bounded list of headlines from the fixed BBC News landing page."""
    request = Request(
        BBC_NEWS_URL,
        headers={"User-Agent": "FakeNewsTextAnalyzer/1.0 (public headline graph)"},
    )
    opener = build_opener(RejectRedirectHandler())
    try:
        with opener.open(request, timeout=15) as response:
            final_url = urlsplit(response.geturl())
            if final_url.scheme != "https" or final_url.netloc.lower() != "www.bbc.com":
                raise ValueError("BBC News redirected outside the approved domain.")
            page = response.read(MAX_BBC_PAGE_BYTES + 1)
    except HTTPError as error:
        raise ValueError(
            f"BBC News returned HTTP status {error.code}; try again later."
        ) from error
    except URLError as error:
        raise ValueError(f"Could not reach BBC News: {error.reason}") from error

    if len(page) > MAX_BBC_PAGE_BYTES:
        raise ValueError("The BBC News page exceeded the 5 MB download limit.")

    parser = BBCArticleLinkParser()
    parser.feed(page.decode("utf-8", errors="replace"))
    articles = list(parser.articles.items())[:MAX_BBC_ARTICLES]
    if not articles:
        raise ValueError(
            "No BBC article links were found. BBC may have changed its page layout."
        )

    return pd.DataFrame(
        [
            {
                "statement": title,
                "label": "Online article",
                "publisher": "BBC News",
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

    publisher_groups = {
        publisher: group.index.tolist()
        for publisher, group in graph_nodes.groupby("publisher", sort=True)
    }
    edges: list[tuple[int, int]] = []
    for node_indices in publisher_groups.values():
        if len(node_indices) > 1:
            edges.extend(
                (node_indices[index], node_indices[(index + 1) % len(node_indices)])
                for index in range(len(node_indices))
            )

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

    color_by_label = {
        "False": "#d64f4f",
        "Suspicious": "#f2b84b",
        "True": "#35a853",
        "Online article": "#4285f4",
    }
    node_colors = [
        color_by_label.get(
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
            f"<b>{escape(str(row['statement'])[:220])}</b><br>"
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
        "Links indicate a shared publisher, not that articles discuss the same claim."
    )
    node_table = graph_nodes[["publisher", "label", "node_type", "statement", "url"]]
    st.dataframe(node_table, width="stretch", hide_index=True)


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
    for row in csv.reader(StringIO(content), delimiter=delimiter):
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
    elif all(len(row) == len(LIAR_COLUMNS) for row in rows):
        data = pd.DataFrame(rows, columns=LIAR_COLUMNS, dtype=str).fillna("")
        data["statement"] = data["statement"].str.strip()
        data["label"] = data["label"].str.strip().str.lower()
        data["publisher"] = data["speaker"].str.strip()
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
            }
        )

    data["label"] = data["label"].map(LABEL_MAP)
    data["statement"] = data["statement"].str.strip()
    if "publisher" not in data:
        data["publisher"] = "Unknown source"
    data["publisher"] = data["publisher"].fillna("").replace("", "Unknown source")
    data = data.dropna(subset=["label"])
    data = data[data["statement"].ne("")]
    return data.reset_index(drop=True)


def create_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Fake News Text Analyzer", page_icon="📰", layout="wide")
    st.title("Fake News Text Analyzer")
    st.write(
        "Train a text-only baseline on LIAR-format data, review its held-out "
        "performance, and estimate the label of a statement."
    )
    st.warning(
        "This is a research/demo classifier, not a fact-checker. Its output is a "
        "model prediction, not verification of a claim."
    )

    with st.sidebar:
        st.header("Database")
        try:
            mysql_config = st.secrets.get("connections", {}).get("mysql", {})
        except StreamlitSecretNotFoundError:
            mysql_config = {}
        mysql_configured = bool(mysql_config.get("url"))
        if mysql_configured:
            if st.button("Test MySQL connection"):
                try:
                    connection = st.connection("mysql", type="sql")
                    connection.query("SELECT 1 AS connected", ttl=0)
                except (SQLAlchemyError, StreamlitAPIException, ValueError) as error:
                    st.error(
                        "Could not connect to MySQL "
                        f"({type(error).__name__}). Check the connection URL in "
                        ".streamlit/secrets.toml and confirm the database is reachable."
                    )
                else:
                    st.success("Connected to MySQL.")
        else:
            st.info(
                "Add a MySQL connection URL in .streamlit/secrets.toml. "
                "See .streamlit/secrets.toml.example."
            )

        st.header("Training data")
        uploaded_file = st.file_uploader(
            "Upload a LIAR TSV/CSV or labeled text dataset",
            type=["tsv", "csv", "txt"],
            help=(
                "Supports standard 14-column LIAR files or other delimited files "
                "with a header identifying label/class and statement/text columns. "
                "Maximum file size: 500 MB."
            ),
        )
        default_path = Path("data") / "train.tsv"
        desktop_dataset = Path.home() / "Desktop" / "news_articles.csv"
        if not default_path.is_file() and desktop_dataset.is_file():
            default_path = desktop_dataset
        use_default = not uploaded_file and default_path.is_file()
        if use_default:
            st.caption(f"Using local dataset: {default_path}")
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
            dataset = load_liar_tsv(raw_bytes)
            counts = dataset["label"].value_counts()
            if len(dataset) < 10 or len(counts) < 2 or counts.min() < 2:
                raise ValueError(
                    "The dataset must contain at least 10 valid statements and at "
                    "least 2 examples in each of at least two recognized classes."
                )

            train_text, test_text, train_labels, test_labels = train_test_split(
                dataset["statement"],
                dataset["label"],
                test_size=0.2,
                random_state=42,
                stratify=dataset["label"],
            )
            model = create_model()
            model.fit(train_text, train_labels)
            predictions = model.predict(test_text)
            report = classification_report(
                test_labels,
                predictions,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )
            st.session_state["model"] = model
            st.session_state["training_dataset"] = dataset
            st.session_state["metrics"] = {
                "accuracy": accuracy_score(test_labels, predictions),
                "micro_f1": f1_score(
                    test_labels,
                    predictions,
                    labels=CLASS_ORDER,
                    average="micro",
                    zero_division=0,
                ),
                "report": report,
                "dataset_size": len(dataset),
            }
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"Could not train the model: {error}")
            return

    metrics = st.session_state.get("metrics")
    if metrics:
        st.subheader("Held-out evaluation")
        first, second, third, fourth = st.columns(4)
        first.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        second.metric("Micro-F1", f"{metrics['micro_f1']:.1%}")
        third.metric("Valid statements", f"{metrics['dataset_size']:,}")
        fourth.metric("Evaluation split", "20% held out")

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
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
            ),
            width="stretch",
        )

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

    st.subheader("Article relationship graph")
    st.write(
        "Dataset articles and headlines scraped from the BBC News page are "
        "shown as nodes. Edges connect articles from the same publisher."
    )
    if st.button("Fetch latest BBC News articles", key="fetch_bbc_articles"):
        try:
            st.session_state["online_articles"] = fetch_bbc_articles()
        except ValueError as error:
            st.error(f"Could not add BBC articles: {error}")
            return

    online_articles = st.session_state.get("online_articles")
    if online_articles is None:
        st.info("Fetch BBC articles to add them to the graph.")
    else:
        training_dataset = st.session_state.get("training_dataset")
        if training_dataset is None:
            st.error("Training dataset is missing. Train the model again to rebuild it.")
        else:
            render_article_graph(training_dataset, online_articles)

    st.subheader("Analyze a statement")
    statement = st.text_area(
        "Statement text",
        placeholder="Paste a news statement or claim here...",
        height=140,
    )
    if st.button("Analyze statement", disabled=not statement.strip()):
        prediction = model.predict([statement.strip()])[0]
        probabilities = model.predict_proba([statement.strip()])[0]
        probability_by_label = dict(zip(model.classes_, probabilities))

        st.markdown(f"### Predicted category: **{prediction}**")
        st.caption(
            "The percentages below are model scores and should not be interpreted "
            "as the probability that a claim is factually true."
        )
        st.bar_chart(
            pd.Series(
                {
                    label: probability_by_label.get(label, 0.0)
                    for label in CLASS_ORDER
                },
                name="Model score",
            )
        )


if __name__ == "__main__":
    main()
