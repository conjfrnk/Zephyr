# Zephyr

Zephyr is a Flask application for planning running routes that aim to cover every road within user specified ZIP codes. It leverages OpenStreetMap data through **osmnx** and **networkx** to build graphs and compute looped routes. Runs and user preferences are stored in a SQLite database created automatically in the project directory.

The web interface uses Leaflet maps and provides features to:

- Load road networks for one or more ZIP codes.
- Automatically generate loop routes from a chosen starting location with optional hill avoidance and basic weather checks.
- Mark planned runs as completed and track overall progress with mileage statistics.

## Running locally

Install the required packages and start the Flask server:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser. A local `zephyr.db` file will hold run history and preferences.

## Next steps

- Add Google OAuth2 authentication.
- Deploy to AWS Elastic Beanstalk using PostgreSQL instead of SQLite.
- Explore integration with the Strava API.

Zephyr is distributed under the terms of the GNU GPL v3, see `LICENSE` for details.
