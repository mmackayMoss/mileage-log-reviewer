# Mileage Log Reviewer

Side project for analyzing mileage expense logs.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mileage-log-reviewer-iv3rkhnkjf.streamlit.app)

### How to run it on your own machine

1. Create a virtual environment

   ```
   $ python3 -m venv env
   ```

2. Activate the virtual environment

   ```
   $ source env/bin/activate
   ```

3. Create streamlit secrets

   Create a `.streamlit` directory in the root of the project, then create a `secrets.toml` file in that directory. Add your environment variables to the `secrets.toml` file.

4. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

5. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

   Type `Ctrl+C` to stop the server.

6. When finished, deactivate the virtual environment

   ```
   $ deactivate
   ```

### Possible issues

If you get a nltk certificate error, run:

```
$ bash '/Applications/Python 3.13/Install Certificates.command'
```