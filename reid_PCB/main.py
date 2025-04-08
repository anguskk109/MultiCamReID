from video_stream import VideoStream
from detector import YOLODetector
from reid import build_pcb_model, extract_pcb_features
from matcher import GlobalReIDMatcher
from tracker import MultiCameraTracker
from utils import draw_tracks
import cv2

video_paths = ["Video/110.mp4", "Video/111.mp4", "Video/112.mp4"]

# Initialize components
video_streams = [VideoStream(src) for src in video_paths]
detector = YOLODetector("yolov8n.pt")

reid_model = build_pcb_model()

tracker = MultiCameraTracker(num_cams=len(video_paths))
global_matcher = GlobalReIDMatcher(similarity_threshold=0.25)
# Initialize local-to-global ID mapping
global_matcher.local_to_global = {i: {} for i in range(len(video_paths))}

skip_frames = 5
frame_counts = [0 for _ in video_streams]

try:
    while True:
        frames = [stream.read()[1] for stream in video_streams]
        if all(f is None for f in frames):
            break

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            # frame_counts[i] += 1
            # if frame_counts[i] % skip_frames != 0:
            #     continue

            detections = detector.detect(frame, visualize=True)
            if detections:
                features = [extract_pcb_features(reid_model, frame[y1:y2, x1:x2])
                            for (x1, y1, x2, y2, _) in detections]
                formatted = [[[x1, y1, x2, y2], conf, "person"] for (x1, y1, x2, y2, conf) in detections]

                # Step 1: Match features globally
                global_ids = global_matcher.match(features)
                global_matcher.update_memory(global_ids, features)

                # Step 2: Update tracker
                tracks = tracker.update(i, formatted, features, frame)

                 # Step 3: Map detection box to GID
                detection_to_gid = {
                    tuple([x1, y1, x2, y2]): gid
                    for ((x1, y1, x2, y2, _), gid) in zip(detections, global_ids)
                }

                # Step 4: Map local track_id to global_id using original detection box
                for track in tracks:
                    if track.original_ltwh is not None:
                        x, y, w, h = track.original_ltwh
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + w), int(y + h)
                        box = (x, y, w, h)

                        gid = detection_to_gid.get(box)
                        if gid is not None:
                            global_matcher.local_to_global[i][track.track_id] = gid


            else:
                tracks = []


            # Step 5: Draw tracks with global ID (fallback-aware)
            for track in tracks:
                if track.is_confirmed() and track.time_since_update == 0:
                    gid = global_matcher.local_to_global[i].get(track.track_id, f"?")
                    draw_tracks(frame, [track], label=f"ID {gid}")


            cv2.imshow(f"Camera {i+1}", cv2.resize(frame, (0, 0), fx=0.3, fy=0.3))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for stream in video_streams:
        stream.release()
    cv2.destroyAllWindows()
