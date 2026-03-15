import time
#from commands_new import execute
import cv2
import numpy as np
from cv2 import aruco
from srv import Manipulator

# -----------------------------
# Lighting-robust ArUco helpers
# -----------------------------
FILTERS_AFTER_SEC = 5.0  # после этого времени включаем "тяжёлые" фильтры

_PREPROC_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _auto_gamma_from_mean(mean_val: float) -> float:
    # mean > ~170: пересвет -> затемняем (gamma > 1)
    # mean < ~80:  темно -> осветляем (gamma < 1)
    if mean_val > 180:
        return 1.9
    if mean_val > 155:
        return 1.5
    if mean_val < 65:
        return 0.7
    if mean_val < 90:
        return 0.85
    return 1.0


def _apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return gray
    lut = np.array(
        [np.clip((i / 255.0) ** gamma * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, lut)


def preprocess_for_aruco(frame, mode: str):
    gray = _to_gray(frame)
    if gray is None:
        return None

    if mode == "raw":
        return gray

    if mode == "clahe":
        return _PREPROC_CLAHE.apply(gray)

    if mode == "gamma":
        m = float(np.mean(gray))
        g = _auto_gamma_from_mean(m)
        return _apply_gamma(gray, g)

    if mode == "equalize":
        return cv2.equalizeHist(gray)

    if mode == "adapt":
        g = cv2.medianBlur(gray, 3)
        return cv2.adaptiveThreshold(
            g, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 7
        )

    if mode == "adapt_inv":
        b = preprocess_for_aruco(frame, "adapt")
        return cv2.bitwise_not(b) if b is not None else None

    raise ValueError(f"Unknown preprocess mode: {mode}")


def make_detector_presets(aruco_dict, base_min_perimeter: float):
    """3 пресета под разный свет/контраст."""
    dets = []

    # Preset 0: базовый
    p0 = aruco.DetectorParameters()
    p0.minMarkerPerimeterRate = base_min_perimeter
    p0.cornerRefinementWinSize = 5
    p0.cornerRefinementMaxIterations = 30
    p0.cornerRefinementMinAccuracy = 0.1
    dets.append(aruco.ArucoDetector(aruco_dict, p0))

    # Preset 1: harsh light / glare
    p1 = aruco.DetectorParameters()
    p1.minMarkerPerimeterRate = max(0.012, base_min_perimeter * 0.8)
    p1.cornerRefinementWinSize = 5
    p1.cornerRefinementMaxIterations = 30
    p1.cornerRefinementMinAccuracy = 0.1
    p1.adaptiveThreshWinSizeMin = 5
    p1.adaptiveThreshWinSizeMax = 75
    p1.adaptiveThreshWinSizeStep = 10
    p1.adaptiveThreshConstant = 3
    dets.append(aruco.ArucoDetector(aruco_dict, p1))

    # Preset 2: low contrast / mixed shadows
    p2 = aruco.DetectorParameters()
    p2.minMarkerPerimeterRate = max(0.010, base_min_perimeter * 0.6)
    p2.cornerRefinementWinSize = 5
    p2.cornerRefinementMaxIterations = 30
    p2.cornerRefinementMinAccuracy = 0.1
    p2.adaptiveThreshWinSizeMin = 7
    p2.adaptiveThreshWinSizeMax = 101
    p2.adaptiveThreshWinSizeStep = 12
    p2.adaptiveThreshConstant = 10
    dets.append(aruco.ArucoDetector(aruco_dict, p2))

    return dets


def _as_detectors(detector_or_list):
    if isinstance(detector_or_list, (list, tuple)):
        return list(detector_or_list)
    return [detector_or_list]



def detect_markers_best(frame, detector_or_list, budget: str):
    """Вернуть corners, ids с fallback по предобработке и пресетам.
    budget: "fast" (только raw) или "full" (raw+clahe+gamma+equalize+adapt+adapt_inv)
    """
    detectors = _as_detectors(detector_or_list)
    modes = ("raw",) if budget == "fast" else ("raw", "clahe", "gamma", "equalize", "adapt", "adapt_inv")

    best_corners, best_ids = None, None
    best_score = -1

    for mode in modes:
        img = preprocess_for_aruco(frame, mode)
        if img is None:
            continue

        for det in detectors:
            corners, ids, _ = det.detectMarkers(img)
            total = 0 if ids is None else len(ids)
            score = total  # простой скоринг: больше маркеров = лучше
            if score > best_score:
                best_score = score
                best_corners, best_ids = corners, ids

            # ранний выход: в fast режиме raw уже что-то нашёл
            if budget == "fast" and ids is not None and len(ids) > 0:
                return corners, ids

    return best_corners, best_ids


def find_marker_index(ids, marker_id):
    if ids is None:
        return None
    flat = ids.flatten()
    matches = np.where(flat == int(marker_id))[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


#def put_cube(man, index, base, side):

def take_cube(cap, man, index, base, side, up):
    #cap = cv2.VideoCapture(1)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.01
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1
    detectors = make_detector_presets(aruco_dict, base_min_perimeter=0.01)
    take_phase_start = time.time()

    base_ang = base
    side_ang = side
    up_ang = up
    rhand_ang = 0

    t = 0
    final = False
    #execute(man, coords)
    while(True):
        man.Base.SetSyncServoRotation(base_ang)
        man.Side.SetSyncServoRotation(side_ang)
        man.RHand.SetSyncServoRotation(rhand_ang)
        man.Up.SetSyncServoRotation(up_ang)
        ret, frame = cap.read()

        if not ret:
            continue
        h, w = frame.shape[:2]
        frame_center = np.array([w//2, h//2])
        use_full = (time.time() - take_phase_start) >= FILTERS_AFTER_SEC
        budget = "full" if use_full else "fast"
        corners, ids = detect_markers_best(frame, detectors, budget=budget)

        target_idx = find_marker_index(ids, index)
        if target_idx is not None:
            corners = corners[target_idx][0]

            pts = corners.astype(np.float32)
            area = cv2.contourArea(pts)
            #print("Area:", area)
            #print(corners)
            cx = ((corners[0][0] + corners[1][0])/2 + (corners[2][0] + corners[3][0])/2)/2
            cy = ((corners[0][1] + corners[1][1])/2 + (corners[2][1] + corners[3][1])/2)/2
            center = np.array((cx, cy), dtype=np.int32)
            cv2.circle(frame, center, 20, (255, 0, 0), 2)
            dx = frame_center[0] - center[0]
            dy = frame_center[1] - center[1]
            da = area - 56000
            p0 = corners[0]
            p1 = corners[1]
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            angle = abs(angle)
            #print(angle)
            if angle > 90:
                angle = 180 - angle
            if p0[1] < p1[1]:
                angle = -angle
            #print(angle)
            #continue
            done = True
            if not final:
                if abs(dx) > 25:
                    done = False
                    if dx > 0:
                        base_ang += abs(2*dx/100)
                    else:
                        base_ang -= abs(2*dx/100)
                if abs(dy) > 50:
                    done = False
                    if dy > 0:
                        side_ang += abs(4*dy/100)
                    else:
                        side_ang -= abs(4*dy/100)
                if (abs(da)) > 5000:
                    done = False
                    if da > 0:
                        up_ang -= abs(1*da/5000)
                    else:
                        up_ang += abs(1*da/5000)
            print(dx, dy, da, base_ang, side_ang, up_ang)

            #print(angle)
            base_ang = np.clip(round(base_ang), -100, 100)
            side_ang = np.clip(round(side_ang), -100, 100)
            up_ang = np.clip(round(up_ang), -100, 100)
            #print(base_ang, side_ang, up_ang)
            if done:
                final = True
                #if angle > 10:
                rhand_ang = -angle
                man.RHand.SetSyncServoRotation(rhand_ang)
                man.Hand.SetSyncServoRotation(50)
                side_ang -= 90
                up_ang -= 15
                print("target_angle:", angle)
                man.Up.SetSyncServoRotation(up_ang)
                man.Side.SetSyncServoRotation(side_ang)


                up_ang += 55
                man.Up.SetSyncServoRotation(up_ang)
                man.Hand.SetSyncServoRotation(-80)
                up_ang -= 70
                man.Up.SetSyncServoRotation(up_ang)
                man.RHand.SetSyncServoRotation(0)
                time.sleep(3)
                if man.GetHandVoltage() < 0.1:
                    return False

                break
        else:
            t += 1
        # даём алгоритму минимум 12 сек на поиск/дожим,
        # при этом фильтры включатся автоматически после FILTERS_AFTER_SEC
        if (time.time() - take_phase_start) >= 100.0:
            return False

        #man.Base.SetSyncServoRotation(base_ang)
        #man.Side.SetSyncServoRotation(side_ang)
        #man.RHand.SetSyncServoRotation(rhand_ang)
        #man.Up.SetSyncServoRotation(up_ang)

        cv2.imshow('Video', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    #cap.release()

