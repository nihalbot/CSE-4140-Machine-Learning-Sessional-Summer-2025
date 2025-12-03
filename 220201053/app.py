from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# CSV load
movies = pd.read_csv("tmdb_5000_movies.csv")

# Keep only available columns
# If your CSV has poster links, you can add 'poster_path' here
movies = movies[['title', 'overview', 'release_date']]

# Convert to list of dicts for easy access in Jinja
movies_list = movies.to_dict(orient='records')

@app.route("/", methods=["GET", "POST"])
def index():
    search_results = []
    if request.method == "POST":
        query = request.form.get("movie_name", "").lower()
        for movie in movies_list:
            if query in movie['title'].lower():
                search_results.append(movie)
    return render_template("index.html", movies=search_results)

if __name__ == "__main__":
    app.run(debug=True)
