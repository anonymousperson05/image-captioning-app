from flask import Flask, render_template, request, flash
from PIL import Image

from utils import (
    generate_caption,
    generate_caption_blip,
    image_to_base64,
    load_model,
    preprocess_image,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me"

# Load trained model
try:
    model, device, word2idx, idx2word = load_model()
except Exception as exc:
    import logging
    logging.exception("Model failed to load: %s", exc)
    model, device, word2idx, idx2word = None, "cpu", None, None


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_data = None

    if request.method == "POST":
        file = request.files.get("image")
        mode = request.form.get("mode", "custom")  

        if not file or file.filename == "":
            flash("Please choose an image file to upload.")
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
            except Exception:
                flash("Could not read that file as an image.")
            else:
                image_data = image_to_base64(image)

                try:
                    if mode == "blip":
                        #  HIGH ACCURACY MODE
                        caption = generate_caption_blip(image)
                    else:
                        #  YOUR MODEL
                        tensor = preprocess_image(image, device)

                        caption = generate_caption(
                            model,
                            tensor,
                            word2idx,
                            idx2word
                        )

                except Exception as e:
                    flash(f"Error generating caption: {str(e)}")

    return render_template("index.html", caption=caption, image_data=image_data)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)