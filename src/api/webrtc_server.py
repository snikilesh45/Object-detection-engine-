import cv2
import asyncio
import time
import numpy as np
import av
from contextlib import asynccontextmanager          
from ultralytics import YOLO
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

pcs = set()
model = YOLO("yolo11n.engine")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield                                           # server is running
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

app = FastAPI(lifespan=lifespan)                    # pass lifespan here




class VideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open.")
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        results = model(frame, imgsz=640, verbose=False)
        frame = results[0].plot()

        curr_time = time.perf_counter()
        self.fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time

        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


@app.get("/")
def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<body>
    <h2>YOLO WebRTC Stream</h2>
    <video id="video" autoplay playsinline muted
           style="width:640px; border:1px solid #ccc;"></video>
    <br>
    <button onclick="start()">Start Stream</button>
    <p id="status">Idle</p>

    <script>
    let pc;

    async function start() {
        document.getElementById("status").innerText = "Connecting...";

        pc = new RTCPeerConnection({
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
        });

        const video = document.getElementById("video");

        pc.ontrack = (event) => {
            console.log("Track received:", event.track.kind);
            video.srcObject = event.streams[0];
            document.getElementById("status").innerText = "Streaming";
        };

        pc.oniceconnectionstatechange = () => {
            console.log("ICE state:", pc.iceConnectionState);
            document.getElementById("status").innerText = "ICE: " + pc.iceConnectionState;
        };

        pc.addTransceiver("video", { direction: "recvonly" });

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Wait for ICE gathering on browser side too
        await new Promise((resolve) => {
            if (pc.iceGatheringState === "complete") {
                resolve();
            } else {
                pc.addEventListener("icegatheringstatechange", () => {
                    if (pc.iceGatheringState === "complete") resolve();
                });
            }
        });

        const response = await fetch("/offer", {
            method: "POST",
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type
            }),
            headers: { "Content-Type": "application/json" }
        });

        const answer = await response.json();
        await pc.setRemoteDescription(answer);
    }
    </script>
</body>
</html>
    """)


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    sdp_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    video = VideoTrack()
    pc.addTrack(video)

    await pc.setRemoteDescription(sdp_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #  Wait for server-side ICE gathering to complete
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


