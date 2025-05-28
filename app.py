from email import message
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, validators, IntegerField

from recomender_system import search_engine


# index.html
class SongForm(FlaskForm):
    song = StringField("Utwór", [validators.Length(min=4)])
    artist = StringField("Artysta", validators=[validators.Length(min=4)])
    numberOfSongs = IntegerField("Liczba podobnych utworów do wyświetlenia")


app = Flask(__name__)
app.config["SECRET_KEY"] = "dunno"


@app.route("/", methods=["POST", "GET"])
def index():
    form = SongForm()
    feedback_message = None
    average_distance = None

    if form.validate_on_submit():
        song_title = form.song.data
        artist_name = form.artist.data
        number_of_song = form.numberOfSongs.data

        recommended_songs_list, avg_dist = search_engine.find_similar_songs(
            song_title, artist_name, number_of_song
        )

        feedback_message = recommended_songs_list
        average_distance = avg_dist

        return render_template(
            "index.html",
            form=form,
            feedback_message=feedback_message,
            average_distance=average_distance,
        )

    return render_template(
        "index.html",
        form=form,
        feedback_message=feedback_message,
        average_distance=average_distance,
    )


if __name__ == "__main__":
    app.run()
