from trackers import ByteTrackTracker
import supervision as sv
import numpy as np

tracker = ByteTrackTracker()
x1 = 100
y1 = 100
x2 = 300
y2 = 400
confidence = 0.95

detections = sv.Detections(
    xyxy=np.array([[x1, y1, x2, y2]]),
    confidence=np.array([confidence])
)
for i in range(10):
    tracked = tracker.update(detections)
    print(tracked.tracker_id)