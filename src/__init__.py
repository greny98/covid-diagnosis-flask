import os
from flask import Flask
from src.routes.diagnosis import diagnosis
from src.db import db


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        app.config.from_mapping(
            SECRET_KEY=os.environ.get("dev"),
            SQLALCHEMY_DATABASE_URI=os.environ.get("SQLALCHEMY_DATABASE_URI"))
    else:
        app.config.from_mapping(test_config)

    # Setup db
    db.app = app
    db.init_app(app)
    # Register blueprint
    app.register_blueprint(diagnosis)

    return app
