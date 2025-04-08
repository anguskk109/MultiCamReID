import cv2

def draw_tracks(frame, tracks, label):
    height, width, _ = frame.shape
    for track in tracks:
        if not track.is_confirmed():
            continue

        if track.original_ltwh is not None:
            x, y, w, h = track.original_ltwh
            x1, y1 = int(x), int(y)
            x2, y2 = int(w), int(h)
        else:
            x1, y1, x2, y2 = map(int, track.to_tlbr())

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

