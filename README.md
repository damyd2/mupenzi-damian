# Fake News Text Analyzer

Project branding: **PhD student Mupenzi Damian**. The app displays an
original project logo and a 2026 copyright/author credit in the sidebar.

A Streamlit research demo that trains and compares seven classifier
configurations on a labeled LIAR-format or compatible dataset: Logistic Regression, Multinomial
Naive Bayes, Complement Naive Bayes, a log-loss SGD Classifier, a
word/character Linear SVM, and a dual-view model combining word and character
TF-IDF text features with available non-identifying LIAR context metadata
(subject, party, context, job, and credibility counts). Speaker/publisher
identity is excluded from model features to reduce source-label leakage. A
seventh model implements a dual-view GCN: word/character text and context
metadata are encoded separately, projected to at most 64 dimensions, and each
view gets its own five-neighbor cosine feature graph. Graph convolution is
followed by learned attention fusion; supervised contrastive loss regularizes
the fused training representations alongside class-weighted cross-entropy.
Validation/test examples connect only to training nodes. Their labels are
never used to construct either graph. This is a local implementation of that
architecture, not a reproduction of the listed external SOTA methods or an
independently validated publication result. A
stratified validation split selects the default model; where enough distinct
publishers or speakers exist, those groups are kept separate between
validation/test and training. A separate holdout split reports accuracy, macro-precision,
macro-recall, macro-F1, and micro-F1. Small datasets that cannot support the
splits are explicitly flagged because their selection estimate is less
reliable.
In the current 10,269-row LIAR run, the dual-view GCN reached 0.421 validation
Macro-F1, 44.8% holdout accuracy, and 44.5% holdout Macro-F1. This is one
fixed grouped split, not a repeated-run estimate or publication-ready evidence.
The **Research benchmarks** view keeps unreported architectural ablations
marked as not reported and separates them from the locally measured results in
Overview.

The trained model can screen headlines fetched from Africa Intelligence,
Al Jazeera, Voice of America, IGIHE, New Vision, and BBC News. Headline estimates
are based on text alone and are not fact-checks or evidence that a source is
unreliable. Only bounded headline lists are fetched; article bodies are not
downloaded.

Newly scraped headlines have no verified training labels. Their model estimates
are used only for source-contribution analysis and predictions, never as
ground-truth labels for fitting or evaluating models.
The metadata-aware alternative is a lightweight TF-IDF baseline inspired by
the two-view approach in the supplied research text. The dual-view GCN is an
experimental feature-graph model using learned attention and supervised
contrastive training. It is not a dense SBERT implementation or a faithful
reproduction of external SOTA methods. Scraped headlines have no verified
context metadata, so predictions use text only; publisher identity is not fed
to the classifier. For manual predictions, optional context metadata can be
entered one line per statement.

The analysis dashboard keeps the labeled training dataset and scraped headline
contributions separate. It includes model comparison, overview/evaluation
metrics, source and label breakdowns, interactive filters, and a publisher
relationship graph. The
**Text processing** view previews the exact analyzer tokens, sparse TF-IDF
features, and an example model estimate for a selected dataset row. It
documents TF-IDF rather than suggesting the app creates dense embeddings.
**Dataset analysis** reports the
loaded dataset's actual sample and class counts plus the trained TF-IDF
vocabulary size. Publisher metadata is used for graph relationships, not as a
model input; the app does not load the 384-dimensional SBERT/Meta embeddings
listed in some proposed dataset specifications. A separate reference table
shows the comparison figures supplied for 3Sources, Citeseer, GRAZ02, MNIST,
NGS, Out-Scene, NoisyMNIST, and LIAR (Ours); those figures are explicitly
marked as unverified reference values.
The **Predictions** view accepts up to 100 newline-separated statements and
shows each model estimate with per-category scores. Scores describe the
classifier output, not the factual probability that a claim is true.
Scraped headline estimate charts use distinct colors: pink for False, amber
for Suspicious, and green for True.
An always-visible color key also identifies online articles in cyan; labeled
dataset charts and graph nodes use the same category colors.
The relationship graph exposes a Nodes table with stable node IDs and an Edges
table mapping connected IDs to their shared-publisher relationship. Edges
connect records from the same publisher only; they do not imply shared claims.

## Run locally

Install the dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Training dataset

The app accepts a LIAR ZIP archive and loads its `train.tsv` split, or you can
upload a TSV, CSV, or TXT dataset. Locally, it automatically uses
`Desktop/PHD new version/PHD new project/liar_dataset-master.zip` when present
and no `data/train.tsv` exists. The archive's README identifies this as the
original LIAR dataset and says it is for research use; it does not identify it
as LIAR-2. Keep the dataset archive out of the public app repository.

## XAMPP MySQL connection

Start **MySQL** in the XAMPP Control Panel. The app uses a dedicated MySQL
account configured in `[connections.mysql]` in `.streamlit/secrets.toml`; use
`.streamlit/secrets.toml.example` as a template. Keep the private file out of
source control. That account should have only `SELECT` and `INSERT` access to the
`fake_news_modeling.model_predictions` and
`fake_news_modeling.dataset_source_authors` tables. The sidebar checks the
live connection and shows saved record counts; use **Refresh XAMPP MySQL
status** to update them immediately. The database can be viewed in phpMyAdmin
at `http://localhost/phpmyadmin/`.

Manual prediction results are not saved automatically.
After generating predictions, explicitly consent and select **Save current
predictions to XAMPP MySQL** to store each statement, predicted category,
model score, per-class scores, model name, and UTC timestamp. Scraped headlines
are not saved by this feature. Restrict database access appropriately because
saved statements may contain sensitive information.

The **Dataset analysis** view separately offers an explicit-consent action to
store loaded labeled dataset rows in `dataset_source_authors`. Records include
the dataset name, source record ID when available, author/speaker, publisher,
label, statement text, available context metadata, and import timestamp.
Repeated saves skip identical rows. Scraped headlines and model-generated
labels are not stored by this dataset action.

For a deployed app, configure a MySQL host reachable from the deployment
environment; `127.0.0.1` refers to the app's own machine, not your local XAMPP
installation. Do not expose a database publicly without appropriate network
restrictions and credentials.

## Local dashboard login and exports

The local dashboard requires the `[auth]` username and password from the
private `.streamlit/secrets.toml` file. The configured username is `admin`;
the generated password is stored only in that local private file. Keep it out
of source control and do not share it.
The Streamlit server binds to `127.0.0.1` in `.streamlit/config.toml` and is
intended for local access. Do not expose this simple app-level login to a
public or shared network; configure a dedicated identity provider and
deployment-grade authentication first.
Once signed in, expand **Change dashboard password** in the sidebar. Enter the
current password and a matching new password of at least 14 characters. The
new password is written to the ignored local secrets file and applies to
future sign-ins.

After training, the **Overview** tab can prepare a ZIP export containing the
loaded dataset, local model-comparison metrics, interactive HTML graphs, and
serialized fitted models. If headlines have been fetched, their records and
model estimates are included separately. The graphs load Plotly.js from a
CDN. Joblib model files are trusted-code artifacts; only load them from a
source you trust.

## Application logs

The app writes timestamped errors and tracebacks to `logs/app.log`, rotating
the log at 2 MB and keeping up to three backups. While signed in, expand
**Application logs** in the sidebar to inspect recent entries. User statement
text and passwords are not intentionally written to the log. Error messages
are also shown in the app when an operation fails. The `logs/` directory is
ignored by Git.

## Deploy

This project is published in
[`damyd2/mupenzi-damian`](https://github.com/damyd2/mupenzi-damian). In
Streamlit Community Cloud, choose that repository, the `main` branch, and
`app.py` as the app entry point.

Before deploying, add these values under the app's **Settings > Secrets**.
Choose a strong dashboard password and set the MySQL connection to a hosted
server reachable from Streamlit Community Cloud:

```toml
[auth]
username = "admin"
password = "replace-with-a-strong-random-password"

[connections.mysql]
host = "your-reachable-mysql-host"
port = 3306
username = "your-app-user"
password = "your-app-password"
database = "fake_news_modeling"
```

The local XAMPP server at `127.0.0.1` cannot be reached from Streamlit
Community Cloud. Do not copy local credentials into a public repository.
