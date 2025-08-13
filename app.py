from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
import tensorflow as tf
import numpy as np
import cv2
import os, json, base64, uuid
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = 'kelinci-secret'

# ====== LOGIN CONFIG ======
DUMMY_USERNAME = 'user_dummy'
DUMMY_PASSWORD = 'password_dummy'

# ====== LOAD MODEL ======
interpreter = tf.lite.Interpreter(model_path="model_mobilenetv2_flask.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

keras_model = load_model("model_mobilenetv2_flask.h5")  # perbaikan nama file

# ====== KONFIG ======
class_names = ["KELINCI ABU", "KELINCI COKLAT", "KELINCI HITAM", "KELINCI PUTIH"]
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 40.0
STAT_PATH = "stats.json"

if os.path.exists(STAT_PATH):
    with open(STAT_PATH, "r") as f:
        stats = json.load(f)
else:
    stats = {label: 0 for label in class_names}

# ====== PREPROCESS ======
def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File tidak valid.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ====== GRAD-CAM ======
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_superimpose(img_path, heatmap, cam_path="static/heatmap.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

# ====== PDF REPORT ======
def generate_pdf(prediction, confidence, img_path, heatmap_path, output_path="laporan.pdf"):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laporan Prediksi Warna Bulu Kelinci")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Prediksi: {prediction}")
    c.drawString(50, height - 120, f"Confidence: {confidence:.2f}%")
    c.drawString(50, height - 160, "Gambar Asli:")
    c.drawImage(ImageReader(img_path), 50, height - 400, width=200, height=200)
    c.drawString(300, height - 160, "Grad-CAM:")
    c.drawImage(ImageReader(heatmap_path), 300, height - 400, width=200, height=200)
    c.showPage()
    c.save()
    return output_path

# ====== AUTH MIDDLEWARE ======
def login_required(func):
    def wrapper(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# ====== ROUTES ======
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == DUMMY_USERNAME and password == DUMMY_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('Login berhasil!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Username atau password salah.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login'))

@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
@login_required
def home():
    return render_template("index.html", stats=stats)

@app.route("/submit", methods=["POST"])
@login_required
def submit():
    prediction = None
    confidence = None
    warning = None
    image_data = None
    heatmap_file = None
    error = None

    file = request.files.get("file")
    if not file:
        error = "Tidak ada gambar yang diunggah."
    else:
        try:
            unique_id = str(uuid.uuid4())
            uploaded_path = os.path.join("static", f"uploaded_{unique_id}.jpg")
            file.save(uploaded_path)

            file.seek(0)
            img_input = preprocess_image(file)

            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = int(np.argmax(output_data[0]))
            confidence = float(output_data[0][pred_idx]) * 100

            if confidence < CONFIDENCE_THRESHOLD:
                warning = "⚠️ Gambar kurang jelas atau bukan kelinci."
            else:
                prediction = class_names[pred_idx]
                stats[prediction] += 1
                with open(STAT_PATH, "w") as f:
                    json.dump(stats, f)

            heatmap = make_gradcam_heatmap(img_input, keras_model, 'Conv_1', pred_idx)
            heatmap_file = os.path.join("static", f"heatmap_{unique_id}.jpg")
            save_and_superimpose(uploaded_path, heatmap, cam_path=heatmap_file)

            with open(uploaded_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            generate_pdf(prediction or "Tidak ada", confidence or 0, uploaded_path, heatmap_file)

        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=round(confidence, 2) if confidence else None,
                           image_data=image_data,
                           warning=warning,
                           error=error,
                           stats=stats,
                           heatmap_file=os.path.basename(heatmap_file) if heatmap_file else None)

@app.route("/statistik")
@login_required
def statistik():
    return render_template("statistik.html", stats=stats)

@app.route("/download")
@login_required
def download():
    return send_file("laporan.pdf", as_attachment=True)

@app.route("/profil")
@login_required
def profil():
    return render_template("profil.html")

if __name__ == "__main__":
    app.run(debug=True)
