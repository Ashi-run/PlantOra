from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from predict import predict_deficiency

app = Flask(__name__)

# Set upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return "No selected file", 400

        # Save file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Call your predict function
        predicted_class, confidence = predict_deficiency(file_path)

        # Pass results to HTML
        return render_template("index.html", results=[
            {"filename": file.filename, "result": predicted_class, "confidence": confidence}
        ])

    return render_template("index.html")


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('result.html', filename=url_for('uploaded_file', filename=filename))
    
    return render_template('index.html', error="Invalid file type. Please upload a valid image.")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

