from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, validators, IntegerField


# index.html
class SongForm(FlaskForm):
    song = StringField("Utwór", [validators.Length(min=4, max=25)])
    artist = StringField("Artysta", validators=[validators.Length(min=4, max=25)])
    numberOfSongs = IntegerField("Artysta")


app = Flask(__name__)
app.config["SECRET_KEY"] = "dunno"


@app.route("/", methods=["POST", "GET"])
def index():
    form = SongForm()

    if form.validate_on_submit():
        song_title = form.song.data
        artist_name = form.artist.data
        number_of_song = form.number.data

        message_to_display = f"Otrzymano utwór: '{song_title}' artysty: '{artist_name}'. Rekomendacje pojawią się tutaj wkrótce!"

        return render_template(
            "index.html", form=form, feedback_message=message_to_display
        )

    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run()
