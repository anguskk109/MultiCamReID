from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiCameraTracker:
    def __init__(
        self,
        num_cams,
        max_age=3,
        n_init=1,
        max_iou_distance=0.3,
        max_cosine_distance=0.5,
        nn_budget=100,
        embedder=None,
        embedder_gpu=True
    ):
        """
        Initializes a separate DeepSORT tracker for each camera.
        """
        self.trackers = [
            DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
                embedder=embedder,
                embedder_gpu=embedder_gpu,
            )
            for _ in range(num_cams)
        ]

    def update(self, cam_idx, detections, features, frame):
        """
        Updates tracker for a specific camera index.
        """
        if cam_idx >= len(self.trackers):
            raise IndexError(f"Invalid camera index: {cam_idx}")
        return self.trackers[cam_idx].update_tracks(
            raw_detections=detections,
            embeds=features,
            # frame=frame
        )
