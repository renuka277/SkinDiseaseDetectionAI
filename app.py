import os

from flask import (
    Flask,
    render_template,
    request
)

from werkzeug.utils import secure_filename

from src.predict import predict


# ======================================================
# Flask App
# ======================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======================================================
# Home Page
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# Prediction
# ======================================================

@app.route("/predict", methods=["POST"])
def prediction():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No file selected."

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    result = predict(filepath)

    filename = os.path.basename(filepath)

    return render_template(
    "result.html",
    image=filename,
    result=result
)


# ======================================================
# Run App
# ======================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )