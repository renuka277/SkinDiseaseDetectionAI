import os
import base64

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

UPLOAD_FOLDER = "/tmp/uploads"

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

    with open(filepath, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = file.mimetype or "image/jpeg"

    image_src = f"data:{mime_type};base64,{image_data}"

    return render_template(
        "result.html",
        image=image_src,
        result=result
    )


# ======================================================
# Run App
# ======================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )