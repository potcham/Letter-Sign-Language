import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from utils_app import Interpreter


st.title("Lenguaje de Señas")
st.write("Interpretando")

model = Interpreter()

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = model.process_frame(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", 
                video_frame_callback=callback, 
                media_stream_constraints={
                "video": True,
                "audio": False},
                rtc_configuration={
                    "iceServers":[{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )