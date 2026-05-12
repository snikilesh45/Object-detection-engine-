from .mjpeg_server import app as mjpeg_app
from .webrtc_server import app as webrtc_app

__all__ = ["mjpeg_app", "webrtc_app"]
