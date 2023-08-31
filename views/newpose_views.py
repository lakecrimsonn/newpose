from flask import Blueprint
from ..utils.newpose import video_start


bp = Blueprint('np', __name__, url_prefix='/np')


@bp.route('/')
def np_route():
    return 'np default page'


@bp.route('/start')
def np_start_route():
    video_start()
    return 'video started'
