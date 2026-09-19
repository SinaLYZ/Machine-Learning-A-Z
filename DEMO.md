# Interactive demo

`app.py` is a small [Streamlit](https://streamlit.io) app with two tabs: a salary predictor (simple linear regression) and a startup profit predictor (multiple linear regression). Both train on the same CSVs the notebooks use, so the numbers match.

## Run it locally

```sh
pip install -r requirements.txt streamlit
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Deploy it for free

1. Push this repo to GitHub (with `app.py` and `requirements.txt` at the root).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click "New app."
3. Point it at this repo, branch `main`, main file path `app.py`.
4. Streamlit installs `requirements.txt` (add `streamlit` to it, or list it in a separate `packages` field) and gives you a public URL like `https://<your-app>.streamlit.app`.

Add that URL — with a screenshot — near the top of the root `README.md`. A live, clickable demo is the single highest-leverage thing you can add to a portfolio repo like this one.
