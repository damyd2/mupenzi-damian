# Fake News Text Analyzer

A Streamlit demo that trains and evaluates a text classifier using a labeled
LIAR-format or compatible dataset.

## Run locally

Install the dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## MySQL connection

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set the
MySQL connection URL. Keep `secrets.toml` private; it is excluded by
`.gitignore`. Use **Test MySQL connection** in the app sidebar to verify the
connection.

For Streamlit Community Cloud, add the connection settings in the app's
**Settings > Secrets** instead of uploading a local secrets file. A database on
`127.0.0.1` is only accessible from the same machine; a deployed app needs a
database host reachable from Streamlit Community Cloud. Do not expose a
database publicly without appropriate network restrictions and credentials.

## Deploy

Create a GitHub repository containing `app.py` and `requirements.txt`, then
select that repository and `app.py` when creating an app in Streamlit Community
Cloud.
