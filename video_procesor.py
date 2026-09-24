
from detector import detect_face
from recognizer import recognize
from database_manager import load_database
import cv2

from trackers import ByteTrackTracker


database = load_database()


def frame_processor(frame):
    intel = []

    faces, detections, boxes = detect_face(frame)

    for face in faces:
        facial_area = face["facial_area"]

        x1 = facial_area["x"]
        y1 = facial_area["y"]
        x2 = x1 + facial_area["w"]
        y2 = y1 + facial_area["h"]

        unknown_face = frame[y1:y2, x1:x2]

        result = recognize(
            data_base=database,
            unknown_img=unknown_face
        )

        intel.append({
            "name": result[0],
            "face_pos": facial_area
        })

    return intel



def read_video(video_path):
    tracker = ByteTrackTracker()

    video = cv2.VideoCapture(video_path)

    success = True
    frame_count = 0

    # Final result:
    # name -> representative face image
    people = {}

    active_tracks = set()

    while success:
        success, frame = video.read()

        if not success:
            break

        frame_count += 1

        # Process every 15th frame
        if frame_count % 15 == 0:

            faces, detections, boxes = detect_face(frame)

            tracked = tracker.update(detections)

            print("Frame:", frame_count)

            left, came = active_check(
                active_tracks,
                tracked.tracker_id
            )

            active_tracks -= left

            for track_id in came:

                # Find the detection belonging to this tracker ID
                for i, tracked_id in enumerate(tracked.tracker_id):

                    if tracked_id == track_id:

                        fb = tracked.xyxy[i]

                        x1, y1, x2, y2 = map(int, fb)

                        # Make sure coordinates are valid
                        if x2 <= x1 or y2 <= y1:
                            break

                        face = frame[y1:y2, x1:x2]

                        result = recognize(
                            unknown_img=face,
                            data_base=database
                        )

                        name = result[0]

                        # Store only the first face image
                        # we get for each recognized person
                        if name not in people:
                            people[name] = face.copy()

                        break

            active_tracks |= came

            print("People seen so far:", set(people.keys()))

    video.release()

    print("\n==============================")
    print("People seen in this video:")
    print("==============================")

    for person in people:
        print("-", person)

    return people




def active_check(active_tracks, tracker_ids):

    current_tracks = {
        track_id
        for track_id in tracker_ids
        if track_id != -1
    }

    previous_tracks = set(active_tracks)

    came = current_tracks - previous_tracks
    left = previous_tracks - current_tracks

    if came:
        print("The following IDs are new:", came)

    if left:
        print("The following IDs left the screen:", left)

    return left, came
