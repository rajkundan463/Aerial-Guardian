import numpy as np

MAX_MISSED = 5


class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.missed = 0
        self.history = []

    def update(self, bbox, center):
        self.bbox = bbox
        self.missed = 0
        self.history.append(center)


class SimpleTracker:
    def __init__(self, max_distance=50):
        self.tracks = []
        self.next_id = 0
        self.max_distance = max_distance

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def update(self, detections):

        # Increase missed count
        for track in self.tracks:
            track.missed += 1

        updated_tracks = []

        for det in detections:
            matched = False
            det_center = self._get_center(det)

            for track in self.tracks:
                track_center = self._get_center(track.bbox)
                distance = np.linalg.norm(det_center - track_center)

                if distance < self.max_distance:
                    track.update(det, det_center)
                    updated_tracks.append(track)
                    matched = True
                    break

            if not matched:
                new_track = Track(det, self.next_id)
                new_track.history.append(det_center)
                self.next_id += 1
                updated_tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in updated_tracks if t.missed < MAX_MISSED]

        return self.tracks