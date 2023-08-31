from flask import Flask


def create_app():
    app = Flask(__name__)

    from .views import newpose_views, files_views
    app.register_blueprint(newpose_views.bp)
    app.register_blueprint(files_views.bp)

    return app
