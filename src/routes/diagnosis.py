import os
from pathlib import Path
from flask import Blueprint
from src.db import XrayInput, XrayDiagnosis, db
from src.diagnosis_model.predict import load_model, predict
import threading
import requests

diagnosis = Blueprint("diagnosis", __name__, url_prefix="/api/diagnosis")
model = load_model("ckpt/classification/checkpoint")

images = []
is_running = False
update_queue = False


def run():
    global images
    global is_running
    global update_queue
    if is_running:
        return
    is_running = True
    if update_queue:
        get_xray_inputs()
        update_queue = False

    while len(images) > 0:
        current = images.pop(0)
        handle_image(current)
    is_running = False


def handle_image(info):
    # Chuẩn đoán kết quả sau đó gọi request update
    filename = os.path.join(Path.home(), 'uploads', info["image"])
    id = info["id"]
    diagnosisExist: XrayDiagnosis = XrayDiagnosis.query.filter_by(xray_input_id=id).first()
    results = predict(model, filename)
    if results is None:
        return
    negative_pneumonia, typical_appearance, indeterminate_appearance, atypical_appearance, no_finding = results[0]
    if diagnosisExist is None:
        diagnosis_result = XrayDiagnosis(
            xray_input_id=id,
            negative_pneumonia=negative_pneumonia,
            typical_appearance=typical_appearance,
            indeterminate_appearance=indeterminate_appearance,
            atypical_appearance=atypical_appearance,
            status="COMPLETED",
            note='')
        db.session.add(diagnosis_result)
        db.session.commit()
    else:
        db.session.query(XrayDiagnosis).filter_by(xray_input_id=id).update({
            "negative_pneumonia": negative_pneumonia,
            "typical_appearance": typical_appearance,
            "indeterminate_appearance": indeterminate_appearance,
            "atypical_appearance": atypical_appearance,
        })

        db.session.commit()
    requests.put(f"http://localhost:3000/xrayInput/{id}")


def get_xray_inputs():
    global images
    xrays = XrayInput.query.filter_by(status="IN_PROGRESS").order_by(XrayInput.created_at).all()
    images = [{"image": xray.filepath, "id": xray.id} for xray in xrays]


def on_update_queue():
    global update_queue
    update_queue = True
    if not is_running:
        run()


@diagnosis.get('/')
def push_to_queue():
    global is_running
    th = threading.Thread(target=on_update_queue)
    th.start()
    return {"message": "running"}
