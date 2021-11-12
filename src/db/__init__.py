from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class XrayInput(db.Model):
    __tablename__ = 'xrayInput'
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String())
    status = db.Column(db.String())
    patient_id = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, onupdate=datetime.now())


class XrayOutput(db.Model):
    __tablename__ = 'xrayOutput'
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String())
    xray_input_id = db.Column(db.Integer)


class XrayDiagnosis(db.Model):
    __tablename__ = 'xrayDiagnosis'
    id = db.Column(db.Integer, primary_key=True)
    xray_input_id = db.Column(db.Integer)
    negative_pneumonia = db.Column(db.Float)
    typical_appearance = db.Column(db.Float)
    indeterminate_appearance = db.Column(db.Float)
    atypical_appearance = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, onupdate=datetime.now(), default=datetime.now())
    status = db.Column(db.String())
    note = db.Column(db.String())
