import cv2
import numpy as np

LOWER_COLOR = np.array([5, 100, 100])
UPPER_COLOR = np.array([25, 255, 255])
REAL_PAD_WIDTH_CM = 10.0
FOCAL_LENGTH_PIXELS = 800.0
ALLOWED_RADIUS_CM = 50.0

feature_params = {"maxCorners": 100, "qualityLevel": 0.3, "minDistance": 7, "blockSize": 7}

lk_params = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}


def calibrate_from_launchpad(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixels_per_cm = None
    altitude_cm = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w > 0 and h > 0:
            pixel_diameter = (w + h) / 2.0
            pixels_per_cm = pixel_diameter / REAL_PAD_WIDTH_CM
            altitude_cm = (REAL_PAD_WIDTH_CM * FOCAL_LENGTH_PIXELS) / pixel_diameter

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Scale: {pixels_per_cm:.1f} px/cm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Altitude: {altitude_cm:.0f} cm",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return frame, pixels_per_cm, altitude_cm


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Point camera at the launchpad. Press 'c' to calibrate.")
print("Press 'r' to reset position to zero.")
print("Press 'q' to quit.")

scale_factor = None
relative_pos_cm = [0.0, 0.0]
old_gray = None
p0 = None
mask = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if scale_factor:
        if mask is None:
            mask = np.zeros_like(display_frame)
        else:
            mask.fill(0)

        frame_gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if p0 is None:
                print("No features found to track. Hold still.")
                continue
            old_gray = frame_gray
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0 and len(good_old) > 0:
                dx_pixels = np.mean(good_old[:, 0] - good_new[:, 0])
                dy_pixels = np.mean(good_old[:, 1] - good_new[:, 1])

                dx_cm = dx_pixels / scale_factor
                dy_cm = dy_pixels / scale_factor

                relative_pos_cm[0] += dx_cm
                relative_pos_cm[1] += dy_cm

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                    display_frame = cv2.circle(display_frame, (a, b), 5, (0, 0, 255), -1)

                display_frame = cv2.add(display_frame, mask)

            old_gray = frame_gray.copy()

            if len(good_new) < 40:
                p0 = None
            else:
                p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = None

        cv2.putText(
            display_frame,
            f"Pos (cm): X={relative_pos_cm[0]:.0f}, Y={relative_pos_cm[1]:.0f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        current_distance_cm = np.sqrt(relative_pos_cm[0] ** 2 + relative_pos_cm[1] ** 2)

        if current_distance_cm > ALLOWED_RADIUS_CM:
            cv2.circle(display_frame, (display_frame.shape[1] - 50, 50), 30, (0, 0, 255), -1)
    else:
        display_frame, _, _ = calibrate_from_launchpad(display_frame)
        cv2.putText(
            display_frame,
            "Press 'c' to lock calibration",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            2,
        )

    cv2.imshow("Tracker View", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c") and scale_factor is None:
        _, scale, alt = calibrate_from_launchpad(frame)
        if scale:
            scale_factor = scale
            print("--- Calibration Complete ---")
            print(f" Major Scale Factor: {scale_factor:.1f} pixels per cm")
            print(f" Major Estimated Altitude: {alt:.0f} cm")
            print("Switched to TRACKING MODE.")
        else:
            print("Calibration failed: Launchpad not detected.")
    elif key == ord("r"):
        relative_pos_cm = [0.0, 0.0]
        print("--- Position Reset to (0, 0) ---")

cap.release()
cv2.destroyAllWindows()
