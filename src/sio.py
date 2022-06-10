import socketio

from tools.nlp.dataset_evaluation import evaluate_dataset

sio = socketio.Client()
socket_app = socketio.ASGIApp(sio)


@sio.event
def connect():
    print("[socket.io] 🚀 Connected")


@sio.event
def connect_error(data):
    print("[socket.io] ❗ Connection failed")


@sio.event
def disconnect():
    print("[socket.io] 🔅 Disconnected")


@sio.event
def message(payload):
    event = payload["event"]
    namespace, action = event.split(":")

    if namespace == "nlp":
        if action == "evaluate-dataset":
            evaluate_dataset(payload["data"])
