"""MMA Vision — fight footage analysis with pose estimation.

Pipeline: video -> (YOLO-pose + ByteTrack) -> fighter tracks -> strike events
-> metrics -> annotated video (+ optional AI commentary).
"""

__version__ = "0.1.0"

# COCO-17 keypoint indices (what YOLO-pose and RTMPose emit)
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
KP_NAMES = [k for k, _ in sorted(KP.items(), key=lambda kv: kv[1])]

SKELETON = [
    ("nose", "l_eye"), ("nose", "r_eye"), ("l_eye", "l_ear"), ("r_eye", "r_ear"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]

FIGHTERS = ("red", "blue")
