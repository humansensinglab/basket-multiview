import socket
import json

def notify_frame_done(port, object_key, frame_idx, status="success"):
    """
    Sends a UDP packet to the localhost listener indicating a frame is finished.
    """
    if port is None or port < 0:
        return

    message = {
        "object_key": object_key,
        "frame_idx": frame_idx,
        "status": status
    }
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Send data to localhost on the specified port
        data = json.dumps(message).encode('utf-8')
        sock.sendto(data, ("127.0.0.1", port))
    except Exception as e:
        print(f"[IPC] Failed to send notification: {e}")
    finally:
        sock.close()