import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import os

# parsing arguments. basically passing alert time and video path or webcam
def parse_arguments():
    parser = argparse.ArgumentParser(description="Posture Monitor")

    parser.add_argument('--video', type=str, default=None,
                        help="Path to video file. If not provided, webcam is used.")

    parser.add_argument('--alert-time', type=int, default=60,
                        help="Seconds of continuous bad posture before alert.")

    return parser.parse_args()


# initialize mrdiapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# if recorded video as input then save the output, if webcam then dont save it 
args = parse_arguments()

if args.video is not None:
    cap = cv2.VideoCapture(args.video)
    save_output = True
else:
    cap = cv2.VideoCapture(0)
    save_output = False

# verify if video file is opened/input video is read
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()


# if recorded video, then where and how to save it
out = None

if save_output:
    os.makedirs("output", exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output/processed.mp4", fourcc, fps, (width, height))

    print("Saving processed video to output/processed.mp4")


# calculation of angle
def calculate_angle(a, b, c):
    """
    Calculates angle between three points:
    angle ABC (with B as vertex)
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # cos(theta) = (BA · BC) / (|BA| |BC|)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# caliberation for first 30 frames. assumes that user is sitting corret in the first second(basically gets a reference)
# Calibration variables
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []

# orientation detection
current_orientation = None
orientation_stable_start = None
STABILITY_DURATION = 2  # seconds


front_reference = None
side_reference = None

calibrating = False
calibration_start_time = None
calibration_orientation = None
calibration_data = []
CALIBRATION_DURATION = 3

# bad posture timer
bad_posture_start = None
alert_triggered = False

# threshold for front and side angles
FRONT_THRESHOLD = 8
SIDE_HEAD_THRESHOLD = 8
SIDE_HIP_THRESHOLD = 4
SIDE_COMBO_THRESHOLD = 6

# we will not display every deviation in angle. we will be displaying the mean of last 5 angle values bcz
# Angle might jump:
# 78° → 80° → 75° → 82° → 74°
# Even if posture didn’t really change.
# That can trigger false deviation.
SMOOTHING_WINDOW = 5
front_angle_buffer = []

side_head_buffer = []
side_hip_buffer = []
side_combo_buffer = []


while cap.isOpened():

    # read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # get current time
    current_time = time.time()

    # convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process frame with pose model
    results = pose.process(rgb_frame)

    # safe pose check
    if not results.pose_landmarks:
        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # get landmarks from pose model
    landmarks = results.pose_landmarks.landmark

    # -------------------------
    # EXTRACT LANDMARKS
    # -------------------------

    # get left shoulder landmark
    left_shoulder = (
        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])
    )

    # get right shoulder landmark
    right_shoulder = (
        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])
    )

    # get neck center landmark
    neck_center = (
        (left_shoulder[0] + right_shoulder[0]) // 2,
        (left_shoulder[1] + right_shoulder[1]) // 2
    )

    # get nose landmark
    nose = (
        int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1]),
        int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0])
    )

    # -------------------------
    # FRONT ANGLE
    # -------------------------

    # calculate front angle
    front_angle = calculate_angle(left_shoulder, neck_center, nose)

    # add front angle to buffer
    front_angle_buffer.append(front_angle)

    # remove oldest front angle from buffer if buffer is full
    if len(front_angle_buffer) > SMOOTHING_WINDOW:
        front_angle_buffer.pop(0)

    # calculate mean of front angle buffer
    smoothed_front_angle = np.mean(front_angle_buffer)
    



    # ------------------------- 
    # DYNAMIC SIDE SELECTION
    # -------------------------

    left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
    right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility

    if left_visibility > right_visibility:
        side_shoulder = left_shoulder
        side_hip = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0])
        )
        side_ear = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0])
        )
    else:
        side_shoulder = right_shoulder
        side_hip = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0])
        )
        side_ear = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0])
        )

    # -------------------------
    # SIDE ANGLES
    # -------------------------
    # calculate side head shoulder angle
    side_head_shoulder_angle = calculate_angle(side_ear, side_shoulder, side_hip)

    # calculate side hip vertical line
    hip_vertical = (side_hip[0], side_hip[1] - 100)

    # calculate side hip shoulder angle
    side_hip_shoulder_angle = calculate_angle(side_shoulder, side_hip, hip_vertical)

    # calculate side shoulder hip head angle
    side_shoulder_hip_head_angle = calculate_angle(side_shoulder, side_hip, side_ear)

    # smoothing side angles
    side_head_buffer.append(side_head_shoulder_angle)
    side_hip_buffer.append(side_hip_shoulder_angle)
    side_combo_buffer.append(side_shoulder_hip_head_angle)

    # remove oldest side angles from buffer if buffer is full
    if len(side_head_buffer) > SMOOTHING_WINDOW:
        side_head_buffer.pop(0)
        side_hip_buffer.pop(0)
        side_combo_buffer.pop(0)

    # calculate mean of side angles
    smoothed_head = np.mean(side_head_buffer)
    smoothed_hip = np.mean(side_hip_buffer)
    smoothed_combo = np.mean(side_combo_buffer)

    # -------------------------
    # ORIENTATION DETECTION
    # -------------------------

    dx = abs(left_shoulder[0] - right_shoulder[0])
    dy = abs(left_shoulder[1] - right_shoulder[1])

    # calculate horizontal ratio
    horizontal_ratio = dx / frame.shape[1]
    shoulder_line_angle = np.degrees(np.arctan2(dy, dx))

    if horizontal_ratio > 0.2 and shoulder_line_angle < 20:
        detected_orientation = "FRONT VIEW"
    else:
        detected_orientation = "SIDE VIEW"

    # -------------------------
    # ORIENTATION STABILITY
    # -------------------------

    # This part chooses which orientation ("FRONT VIEW" or "SIDE VIEW") should be shown:
    # - If we are calibrating, just use the orientation we are calibrating for, so it doesn't change suddenly.
    # - If not calibrating, keep track of the person's orientation:
    #     * When the orientation changes (or on the very first run), update the "current_orientation" and also save what time it changed.
    #     * Always use "current_orientation" as the orientation to show right now.
    if calibrating:
        orientation = calibration_orientation
    else:
        if current_orientation is None:
            current_orientation = detected_orientation
            orientation_stable_start = current_time

        elif detected_orientation != current_orientation:
            current_orientation = detected_orientation
            orientation_stable_start = current_time

        orientation = current_orientation

    # This part checks if the orientation has been stable for a while:
    # - If the orientation changed recently (orientation_stable_start is not None), check if enough time has passed.
    # - If yes, set stable to True, meaning the orientation has been stable for a while.
    stable = False
    if orientation_stable_start is not None:
        if current_time - orientation_stable_start >= STABILITY_DURATION:
            stable = True

    # display orientation
    cv2.putText(frame, f"Orientation: {orientation}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    # -------------------------
    # CALIBRATION
    # -------------------------

    if not calibrating and stable:
        if orientation == "FRONT VIEW" and front_reference is None:
            calibrating = True
            calibration_start_time = current_time
            calibration_orientation = "FRONT VIEW"
            calibration_data = []

        elif orientation == "SIDE VIEW" and side_reference is None:
            calibrating = True
            calibration_start_time = current_time
            calibration_orientation = "SIDE VIEW"
            calibration_data = []

    if calibrating:
        if calibration_orientation == "FRONT VIEW":
            calibration_data.append(smoothed_front_angle)
        else:
            calibration_data.append((smoothed_head, smoothed_hip, smoothed_combo))

        # display calibration message
        cv2.putText(frame, "Calibrating...",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        if current_time - calibration_start_time >= CALIBRATION_DURATION:

            if calibration_orientation == "FRONT VIEW":
                front_reference = np.mean(calibration_data)
            else:
                head_vals = [x[0] for x in calibration_data]
                hip_vals = [x[1] for x in calibration_data]
                combo_vals = [x[2] for x in calibration_data]

                side_reference = {
                    "head_shoulder": np.mean(head_vals),
                    "hip_shoulder": np.mean(hip_vals),
                    "shoulder_hip_head": np.mean(combo_vals)
                }

            calibrating = False

    
    if orientation == "FRONT VIEW":

        # Draw lines for front angle
        cv2.line(frame, neck_center, left_shoulder, (255, 0, 0), 2)
        cv2.line(frame, neck_center, nose, (0, 255, 0), 2)

        # Draw vertex
        cv2.circle(frame, neck_center, 5, (0, 0, 255), -1)
    else: 
        cv2.line(frame, side_shoulder, side_ear, (255, 0, 0), 2)
        cv2.line(frame, side_shoulder, side_hip, (0, 255, 0), 2)

        cv2.circle(frame, side_shoulder, 5, (0, 0, 255), -1)

        hip_vertical = (side_hip[0], side_hip[1] - 100)

        cv2.line(frame, side_hip, hip_vertical, (255, 255, 0), 2)
        cv2.line(frame, side_hip, side_shoulder, (0, 255, 255), 2)

        cv2.circle(frame, side_hip, 5, (0, 0, 255), -1)

        cv2.line(frame, side_hip, side_shoulder, (200, 0, 200), 2)
        cv2.line(frame, side_hip, side_ear, (0, 200, 200), 2)

    # -------------------------
    # MONITORING
    # -------------------------

    posture_status = "UNKNOWN"
    color = (255, 255, 255)

    if not calibrating:

        if orientation == "FRONT VIEW" and front_reference is not None:
            deviation = abs(smoothed_front_angle - front_reference)
            if deviation > FRONT_THRESHOLD:
                posture_status = "BAD POSTURE"
                color = (0, 0, 255)
            else:
                posture_status = "GOOD POSTURE"
                color = (0, 255, 0)

        elif orientation == "SIDE VIEW" and side_reference is not None:

            dev1 = abs(smoothed_head - side_reference["head_shoulder"])
            dev2 = abs(smoothed_hip - side_reference["hip_shoulder"])
            dev3 = abs(smoothed_combo - side_reference["shoulder_hip_head"])

            score = 0

            if dev1 > SIDE_HEAD_THRESHOLD:
                score += 1

            if dev2 > SIDE_HIP_THRESHOLD:
                score += 1

            if dev3 > SIDE_COMBO_THRESHOLD:
                score += 1

            if score >= 2:
                posture_status = "BAD POSTURE"
                color = (0, 0, 255)
            else:
                posture_status = "GOOD POSTURE"
                color = (0, 255, 0)


    # display front angle values

    if orientation == "FRONT VIEW":

        cv2.putText(frame,
                    f"Front Angle: {smoothed_front_angle:.1f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        # display front reference values
        if front_reference is not None:
            cv2.putText(frame,
                        f"Front Ref: {front_reference:.1f}",
                        (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            deviation = abs(smoothed_front_angle - front_reference)

            cv2.putText(frame,
                        f"Deviation: {deviation:.1f}",
                        (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2)
    # display side angle values
    if orientation == "SIDE VIEW":

        cv2.putText(frame,
                    f"Head Angle: {smoothed_head:.1f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.putText(frame,
                    f"Hip Shoulder Angle: {smoothed_hip:.1f}",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.putText(frame,
                    f"Combo Angle: {smoothed_combo:.1f}",
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
    # display reference values
        if side_reference is not None:

            cv2.putText(frame,
                        f"Head Ref: {side_reference['head_shoulder']:.1f}",
                        (250, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Hip Ref: {side_reference['hip_shoulder']:.1f}",
                        (250, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Combo Ref: {side_reference['shoulder_hip_head']:.1f}",
                        (250, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

    # -------------------------
    # BAD POSTURE TIMER
    # -------------------------

    bad_duration = 0

    if posture_status == "BAD POSTURE":

        if bad_posture_start is None:
            bad_posture_start = current_time

        bad_duration = current_time - bad_posture_start

        if bad_duration >= args.alert_time:
            alert_triggered = True
    else:
        bad_posture_start = None
        alert_triggered = False

    # display bad time
    cv2.putText(frame,
            f"Bad Time: {bad_duration:.1f}s",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if posture_status == "BAD POSTURE" else (200, 200, 200),
            2)

    # display good/bad posture status
    cv2.putText(frame, posture_status,
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    # display alert message if alert is triggered
    if alert_triggered:
        cv2.putText(frame,
                    "ALERT: Bad posture for 60 seconds!",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

    # -------------------------
    # DISPLAY + SAVE
    # -------------------------

    if save_output and out is not None:
        out.write(frame)

    cv2.imshow("Posture Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# cleanup
cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()



