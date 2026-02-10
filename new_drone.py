import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

# SETUP 
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.time()


STATE_GROUNDED = "GROUNDED"
STATE_TAKEOFF  = "TAKEOFF"
STATE_FLYING   = "FLYING"
STATE_HOVER    = "HOVER"
STATE_LANDING  = "LANDING"

drone_state = STATE_GROUNDED

# Drone position / size on screen (now with 3D depth!)
drone_x, drone_y, drone_z, drone_size = 400, 300, 0.5, 40

# Smoothing (where we are right now)
curr_x, curr_y, curr_z = 400, 300, 0.5

GROUND_Y_RATIO   = 0.85          # where the ground line sits (% of frame height)
HOVER_ALT        = 1.0           # normalised hover altitude
TAKEOFF_SPEED    = 0.02          # how fast the auto-takeoff rises per frame
LANDING_SPEED    = 0.015         # how fast the auto-landing descends per frame
GROUNDED_SIZE    = 30            # small drone when on ground
HOVER_SIZE       = 40            # default size in the air

# 3D DEPTH CONTROL
MIN_DEPTH        = 0.2           # far back (20%)
MAX_DEPTH        = 1.0           # close forward (100%)
DEPTH_SMOOTHING  = 0.15          # smoothing factor for Z-axis

altitude     = 0.0               # 0 = ground, 1 = hover height
fist_start   = None              # timestamp when fist was first detected (for hold-to-land)
FIST_HOLD_TO_LAND = 2.0          # seconds you must hold fist to trigger landing

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Color per state
STATE_COLORS = {
    STATE_GROUNDED: (0, 0, 255),     # Red
    STATE_TAKEOFF:  (0, 165, 255),   # Orange
    STATE_FLYING:   (0, 255, 0),     # Green
    STATE_HOVER:    (0, 255, 255),   # Yellow
    STATE_LANDING:  (255, 100, 0),   # Blue
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for mirror view and get dimensions
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ground_y = int(h * GROUND_Y_RATIO)  # pixel row for the "ground"

    # --- 2. TRACK HANDS ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.time() - start_time) * 1000)                               # Generates timestamp in milliseconds
    results = hand_landmarker.detect_for_video(mp_image, ts)

    gesture = None  # will be "open" or "fist" (or None if no hand)

    if results.hand_landmarks:
        for landmarks in results.hand_landmarks:
            # Draw landmarks and connections
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            for a, b in HAND_CONNECTIONS:
                p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
                p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
                cv2.line(frame, p1, p2, (255, 255, 255), 2)

            wrist = landmarks[0]
            tip   = landmarks[12]
            dist  = math.hypot(tip.x - wrist.x, tip.y - wrist.y)

            if dist < 0.15:
                gesture = "fist"
            elif dist > 0.2:
                gesture = "open"
            
            # Calculate hand size for 3D depth control (palm width)
            # Using distance between thumb base and pinky base
            thumb_base = landmarks[1]
            pinky_base = landmarks[17]
            hand_width = math.hypot(pinky_base.x - thumb_base.x, pinky_base.y - thumb_base.y)
            
            # Map hand width to depth (larger hand = closer, smaller = farther)
            # Typical hand width range: 0.15 (far) to 0.35 (close)
            target_depth = (hand_width - 0.15) / (0.35 - 0.15)
            target_depth = max(MIN_DEPTH, min(MAX_DEPTH, target_depth))  # clamp

    if drone_state == STATE_GROUNDED:
        #  Open hand = take off.
        if gesture == "open":
            drone_state = STATE_TAKEOFF
            fist_start = None

    elif drone_state == STATE_TAKEOFF:
        altitude += TAKEOFF_SPEED
        if altitude >= HOVER_ALT:
            altitude = HOVER_ALT
            drone_state = STATE_FLYING

    elif drone_state == STATE_FLYING:
        #Fist = brake 
        if gesture == "fist":
            drone_state = STATE_HOVER
            fist_start = time.time()   # start the "hold to land" timer
        elif gesture == "open" and results.hand_landmarks:
            # Move toward hand position (3D control)
            for landmarks in results.hand_landmarks:
                target_x = int(landmarks[9].x * w)
                target_y = int(landmarks[9].y * h)
                curr_x = int(curr_x * 0.8 + target_x * 0.2)
                curr_y = int(curr_y * 0.8 + target_y * 0.2)
                
                # 3D DEPTH CONTROL: Update Z-coordinate with smoothing
                thumb_base = landmarks[1]
                pinky_base = landmarks[17]
                hand_width = math.hypot(pinky_base.x - thumb_base.x, pinky_base.y - thumb_base.y)
                target_depth = (hand_width - 0.15) / (0.35 - 0.15)
                target_depth = max(MIN_DEPTH, min(MAX_DEPTH, target_depth))
                curr_z = curr_z * (1 - DEPTH_SMOOTHING) + target_depth * DEPTH_SMOOTHING
                
                # Size based on both hand spread AND depth (perspective)
                wrist = landmarks[0]
                tip   = landmarks[12]
                dist  = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                base_size = int(dist * 300)
                # Apply perspective scaling: closer = bigger, farther = smaller
                depth_scale = 0.5 + curr_z * 0.5  # scale from 0.5x to 1.0x
                target_size = int(base_size * depth_scale)
                drone_size = int(drone_size * 0.9 + target_size * 0.1)
            drone_x = curr_x
            drone_y = curr_y
            drone_z = curr_z

    elif drone_state == STATE_HOVER:
        # Drone holds position.  Open hand = resume flying.
        # Fist held for 2 seconds = intentional landing.
        if gesture == "open":
            drone_state = STATE_FLYING
            fist_start = None
        elif gesture == "fist":
            if fist_start is not None and (time.time() - fist_start) >= FIST_HOLD_TO_LAND:
                drone_state = STATE_LANDING
                fist_start = None
        else:
            # Hand lost while hovering – stay hovering (safe default)
            fist_start = None

    elif drone_state == STATE_LANDING:
        # Auto-descend until we reach ground
        altitude -= LANDING_SPEED
        if altitude <= 0.0:
            altitude = 0.0
            drone_state = STATE_GROUNDED

    # POSITION DRONE BASED ON STATE

    if drone_state == STATE_GROUNDED:
        # Sit on the ground, centered
        drone_x = w // 2
        drone_y = ground_y
        drone_z = 0.5  # middle depth
        drone_size = GROUNDED_SIZE
        curr_x, curr_y, curr_z = drone_x, drone_y, drone_z

    elif drone_state == STATE_TAKEOFF:
        # Rise straight up from ground to hover position
        drone_x = w // 2
        drone_y = int(ground_y - altitude * (ground_y - h * 0.35))
        drone_z = 0.5  # maintain middle depth during takeoff
        drone_size = int(GROUNDED_SIZE + altitude * (HOVER_SIZE - GROUNDED_SIZE))
        curr_x, curr_y, curr_z = drone_x, drone_y, drone_z

    elif drone_state == STATE_LANDING:
        # Descend from current position toward ground
        land_target_y = ground_y
        drone_y = int(land_target_y - altitude * (land_target_y - drone_y))
        drone_x = int(drone_x * 0.95 + (w // 2) * 0.05)   # drift back to center
        drone_z = drone_z * 0.95 + 0.5 * 0.05  # slowly return to middle depth
        drone_size = int(GROUNDED_SIZE + altitude * (HOVER_SIZE - GROUNDED_SIZE))
        curr_x, curr_y, curr_z = drone_x, drone_y, drone_z

  
    # DRAW THE SCENE 

    # Draw ground line
    cv2.line(frame, (0, ground_y), (w, ground_y), (80, 80, 80), 2)
    cv2.putText(frame, "GROUND", (w - 110, ground_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    color = STATE_COLORS.get(drone_state, (255, 255, 255))

    # Draw crosshairs
    cv2.line(frame, (drone_x - 20, drone_y), (drone_x + 20, drone_y), color, 2)
    cv2.line(frame, (drone_x, drone_y - 20), (drone_x, drone_y + 20), color, 2)
    # Draw body
    cv2.circle(frame, (drone_x, drone_y), drone_size, color, 2)
    # Draw "rotors"
    cv2.circle(frame, (drone_x - drone_size, drone_y - drone_size), 10, (255, 255, 255), -1)
    cv2.circle(frame, (drone_x + drone_size, drone_y - drone_size), 10, (255, 255, 255), -1)
    cv2.circle(frame, (drone_x - drone_size, drone_y + drone_size), 10, (255, 255, 255), -1)
    cv2.circle(frame, (drone_x + drone_size, drone_y + drone_size), 10, (255, 255, 255), -1)

   
    # Status line with state name
    status_msgs = {
        STATE_GROUNDED: "GROUNDED  |  Open hand = Take off",
        STATE_TAKEOFF:  "TAKEOFF   |  Auto-ascending...",
        STATE_FLYING:   "FLYING (3D) |  Hand position = X/Y, Hand size = Z depth  |  Fist = Brake",
        STATE_HOVER:    "HOVER     |  Open hand = Resume  |  Hold fist 2s = Land",
        STATE_LANDING:  "LANDING   |  Auto-descending...",
    }
    status = status_msgs[drone_state]
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Altitude bar on the right side
    bar_x = w - 40
    bar_top = int(h * 0.1)
    bar_bot = ground_y
    cv2.line(frame, (bar_x, bar_top), (bar_x, bar_bot), (100, 100, 100), 2)
    alt_y = int(bar_bot - altitude * (bar_bot - bar_top))
    cv2.circle(frame, (bar_x, alt_y), 6, color, -1)
    cv2.putText(frame, "ALT", (bar_x - 15, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Depth bar on the left side (3D forward/backward position)
    depth_bar_x = 40
    depth_bar_top = int(h * 0.1)
    depth_bar_bot = int(h * 0.7)
    cv2.line(frame, (depth_bar_x, depth_bar_top), (depth_bar_x, depth_bar_bot), (100, 100, 100), 2)
    # Map Z from [MIN_DEPTH, MAX_DEPTH] to bar position
    depth_y = int(depth_bar_bot - (drone_z - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * (depth_bar_bot - depth_bar_top))
    cv2.circle(frame, (depth_bar_x, depth_y), 6, color, -1)
    cv2.putText(frame, "DEPTH", (depth_bar_x - 20, depth_bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(frame, "NEAR", (depth_bar_x - 20, depth_bar_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    cv2.putText(frame, "FAR", (depth_bar_x - 15, depth_bar_bot + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # Fist hold-to-land countdown (show during HOVER + fist)
    if drone_state == STATE_HOVER and fist_start is not None:
        elapsed = time.time() - fist_start
        remaining = max(0.0, FIST_HOLD_TO_LAND - elapsed)
        bar_w = int((elapsed / FIST_HOLD_TO_LAND) * 200)
        cv2.rectangle(frame, (20, 60), (20 + 200, 80), (100, 100, 100), 1)
        cv2.rectangle(frame, (20, 60), (20 + min(bar_w, 200), 80), (0, 200, 255), -1)
        cv2.putText(frame, f"Landing in {remaining:.1f}s...", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.imshow("Drone Simulator", frame)
    if cv2.waitKey(1) == ord('q'):
        # Pressing Q while airborne triggers a safe landing first
        if drone_state in (STATE_FLYING, STATE_HOVER, STATE_TAKEOFF):
            drone_state = STATE_LANDING
        else:
            break

    # If we just entered landing via Q, keep looping until grounded
    if drone_state == STATE_GROUNDED and cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()