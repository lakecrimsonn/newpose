from flask import Blueprint, request
from werkzeug.utils import secure_filename

bp = Blueprint('files', __name__, url_prefix='/files')


@bp.route('/')
def route_file():
    return 'files default page'


@bp.route('/upload', methods=['POST'])
def post_file():
    f = request.files['newfile']
    f.save('data/file_upload'+secure_filename(f.filename))
    return 'success'
