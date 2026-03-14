import time

import cv2
import cv2.aruco as aruco
import numpy as np
from math import dist
# Если хотите отладку — поставьте True
DEBUG_PRINT = False


def _solve_marker_pose(corner4, marker_length, camera_matrix, dist_coeffs):
    """Оценка позы одного квадратного маркера через solvePnP.

    corner4: (4,2) углы маркера в пикселях (в порядке ArUco: TL, TR, BR, BL)
    marker_length: сторона маркера (в тех же единицах, что и object_points поля)
    """
    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float32)
    else:
        dist = dist_coeffs

    imgp = np.asarray(corner4, dtype=np.float32).reshape(4, 2)

    half = float(marker_length) / 2.0
    objp = np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float32
    )

    # IPPE для квадратов обычно устойчивее (если доступен)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

    ok, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist, flags=flag)
    if not ok:
        return None, None
    return rvec, tvec


def _reproj_mse(obj_pts, img_pts, rvec, tvec, K, dist):
    """Среднеквадратичная ошибка репроекции (в пикселях^2)."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    diff = proj - img_pts
    return float(np.mean(np.sum(diff * diff, axis=1)))


def _best_corner_correspondence(obj4, img4, rvec, tvec, K, dist):
    """Подбираем соответствие углов маркера к obj4:
    - 4 циклических сдвига
    - и 4 сдвига для перевёрнутого обхода (на случай зеркального порядка)
    Возвращаем img4_best (4x2) с минимальной репроекционной ошибкой.
    """
    img4 = np.asarray(img4, dtype=np.float32).reshape(4, 2)

    best = None
    best_err = 1e30

    # 4 циклических сдвига исходного порядка
    for s in range(4):
        img_shift = np.roll(img4, -s, axis=0)
        err = _reproj_mse(obj4, img_shift, rvec, tvec, K, dist)
        if err < best_err:
            best_err = err
            best = img_shift

    # 4 циклических сдвига для обратного порядка (BL, BR, TR, TL) и т.п.
    img_rev = img4[::-1].copy()
    for s in range(4):
        img_shift = np.roll(img_rev, -s, axis=0)
        err = _reproj_mse(obj4, img_shift, rvec, tvec, K, dist)
        if err < best_err:
            best_err = err
            best = img_shift

    return best, best_err


def detect_cubes_multi(
        marker_ids: list[int],
        frame,
        detector: aruco.ArucoDetector,
        real_points: list,
        camera_matrix,
        dist_coeffs,
        marker_length=2.5,
):
    """ВАРИАНТ Б (точнее):
    - один detectMarkers
    - поза поля: solvePnP по центрам + refine по углам (solvePnPRefineLM, если есть)
    - кубы: solvePnP по углам маркера (только нужные ids)

    marker_ids: какие кубы искать (например [11,0,6,8])
    real_points: маркеры поля в формате [[id, x, y, z], ...]

    Возвращает dict:
      { id: (pos_field_xyz, yaw_rad) }
    где pos_field_xyz — (x,y,z) в системе поля, yaw_rad — вокруг Z поля.
    """
    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float32)
    else:
        dist = dist_coeffs

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    ids_flat = [int(x[0]) for x in ids]

    # ------------------------------------------------------------
    # 1) Поза поля относительно камеры:
    #    (а) быстрый старт по центрам
    #    (б) refine по углам маркеров поля (если доступен solvePnPRefineLM)
    # ------------------------------------------------------------
    rp = {int(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in real_points}

    obj_cent = []
    img_cent = []
    field_idxs = []

    for i, mid in enumerate(ids_flat):
        if mid in rp:
            c = corners[i][0].astype(np.float32)
            center = c.mean(axis=0)
            obj_cent.append(rp[mid])
            img_cent.append(center)
            field_idxs.append(i)

    if len(obj_cent) < 4:
        return None

    obj_cent = np.asarray(obj_cent, dtype=np.float32)
    img_cent = np.asarray(img_cent, dtype=np.float32)

    ok, rvec_field, tvec_field = cv2.solvePnP(
        obj_cent,
        img_cent,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # refine по углам маркеров поля (даёт заметно меньший дрейф в pos_field)
    if hasattr(cv2, "solvePnPRefineLM"):
        half = float(marker_length) / 2.0
        obj_ref_list = []
        img_ref_list = []

        # Сначала подбираем соответствия углов под текущую оценку (rvec_field/tvec_field),
        # потом одним пакетом refineLM по всем углам.
        total_err = 0.0
        for i in field_idxs:
            mid = ids_flat[i]
            cx, cy, cz = rp[mid]

            obj4 = np.array(
                [[cx - half, cy + half, cz],
                 [cx + half, cy + half, cz],
                 [cx + half, cy - half, cz],
                 [cx - half, cy - half, cz]],
                dtype=np.float32
            )

            img4 = corners[i][0].astype(np.float32)

            best_img4, err = _best_corner_correspondence(
                obj4, img4, rvec_field, tvec_field, camera_matrix, dist
            )
            total_err += err

            obj_ref_list.append(obj4)
            img_ref_list.append(best_img4)

        obj_ref = np.vstack(obj_ref_list).astype(np.float32)
        img_ref = np.vstack(img_ref_list).astype(np.float32)

        # refine
        rvec_field, tvec_field = cv2.solvePnPRefineLM(
            obj_ref, img_ref, camera_matrix, dist, rvec_field, tvec_field
        )

        if DEBUG_PRINT:
            print("field refineLM total_err:", total_err)

    R_field, _ = cv2.Rodrigues(rvec_field)

    # Инверсия позы поля → камера в системе поля
    R_field_inv = R_field.T
    t_field_inv = -R_field_inv @ tvec_field

    # ------------------------------------------------------------
    # 2) Кубы: solvePnP по углам маркера (только нужные ids)
    # ------------------------------------------------------------
    want = set(int(x) for x in marker_ids)
    out = {}

    for i, mid in enumerate(ids_flat):
        if mid not in want:
            continue

        rvec_cube, tvec_cube = _solve_marker_pose(
            corners[i][0],
            marker_length,
            camera_matrix,
            dist,
        )
        if rvec_cube is None:
            continue

        R_cube, _ = cv2.Rodrigues(rvec_cube)

        # позиция куба в системе поля
        pos_field = (R_field_inv @ tvec_cube.reshape(3, 1) + t_field_inv).flatten()

        # ориентация куба в системе поля
        R_cube_in_field = R_field_inv @ R_cube

        # yaw вокруг оси Z поля
        yaw = np.arctan2(R_cube_in_field[1, 0], R_cube_in_field[0, 0])

        out[mid] = (pos_field, float(yaw))

        if DEBUG_PRINT:
            print("cube", mid, "pos_field", pos_field, "yaw", yaw)

    if len(out) == 0:
        return None

    return out


def detect_cubes(
        marker_id: int,
        frame,
        detector: aruco.ArucoDetector,
        real_points: list,
        camera_matrix,
        dist_coeffs,
        marker_length=2.5,
):
    """Совместимость со старым кодом: вернуть (pos_xy, yaw) для одного marker_id."""
    res = detect_cubes_multi([marker_id], frame, detector, real_points, camera_matrix, dist_coeffs, marker_length)
    if res is None or marker_id not in res:
        return None
    pos_field, yaw = res[marker_id]
    return pos_field[:2], yaw

def base_angle(angle):
    if 0 <= angle < 90:
        angle = 90 - angle
    elif 90 <= angle <= 180:
        angle = -(angle - 90)
    else:
        raise ValueError("angle must be between 0 and 180 degrees")
    return angle*1.3

def decart_to_polar(coords):
    x,y = coords
    d = dist([0,0], [x, y])
    angle = np.degrees(np.atan2(y, x))
    return d,angle


if __name__ == "__main__":
    import os
    #os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    from srv import Manipulator
    #from commands_new import execute, _predict_batch
    from take import take_cube
    cap = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(1)
    if cap.isOpened() is False:
        print("Error opening video stream 1 or file")
        exit(1)
    if cap2.isOpened() is False:
        print("Error opening video stream 2 or file")
        exit(1)
    #cap2 = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.03
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    FRAME_CENTER = np.array((1920 / 2, 1080 / 2))
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    cv_data = np.load("calib_cam1.npz")
    camera_matrix = cv_data["K"]
    dist_coeffs = cv_data["dist"]
    REAL_POINTS = [[7, -18.5, 1.5, 0], [9, 19, 1.5, 0], [12, 19, 19, 0], [10, -19, 18.5, 0]]
    man = Manipulator()
    cnt = 0
    taken = []
    while True:
        if cnt == 4:
            break
        man.Up.SetSyncServoRotation(-90)
        man.Base.SetAsyncServoRotation(-100)
        man.RHand.SetAsyncServoRotation(0)
        man.Side.SetAsyncServoRotation(90)
        time.sleep(2)
        man.Hand.SetSyncServoRotation(40)
        man.Up.SetSyncServoRotation(0)

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        p = detect_cubes_multi([0, 6, 8, 11], frame, detector, REAL_POINTS, camera_matrix, dist_coeffs, marker_length=1)
        cv2.imshow("frame", frame)
        cv2.waitKey(10)
        if p is None:
            print("can't detect marker")
            time.sleep(2)
            continue
        #d, ang = decart_to_polar(p[0][0][:2])
        #t_ang = base_angle(ang)
        #man.Up.SetSyncServoRotation(-90)
        print(p)
        index = -1
        for key, val in p.items():
            if key not in taken:
                taken.append(key)
                index = key
                break
        if index == -1:
            continue
        coords = p[index][0]*10
        angle = np.degrees(np.atan2(coords[1], coords[0]))
        t_angle = base_angle(angle)
        man.Base.SetSyncServoRotation(t_angle)
        man.Side.SetSyncServoRotation(100)
        man.Up.SetSyncServoRotation(-40)
        is_taken = False
        for side in range(100, -100, -15):
            man.Side.SetSyncServoRotation(side)
            time.sleep(1)
            marker_detected = 0

            for i in range(3):
                ret, tmp_frame = cap2.read()
                if not ret:
                    continue
                cv2.imshow("frame", tmp_frame)
                cv2.waitKey(10)
                corners, ids, _ = detector.detectMarkers(tmp_frame)
                if ids is not None and index in ids.flatten():
                    marker_detected += 1
                time.sleep(0.25)
            if marker_detected < 2:
                continue
            print("marker detected ", marker_detected)
            ret, frame2 = cap2.read()
            if not ret:
                print("cannot grab frame2")
                break
            corners, ids, _ = detector.detectMarkers(frame2)
            if ids is None:
                continue
            if index in ids.flatten():
                take_cube(cap2, man, index, t_angle, side-10, -40)
                is_taken = True
                break
        if not is_taken:
            continue
            #print(side)
        if cnt < 4:
            man.Up.SetSyncServoRotation(-90)
            time.sleep(1)
            man.RHand.SetSyncServoRotation(90)
            man.Base.SetAsyncServoRotation(-15 + 30*(cnt < 2))
            man.Side.SetAsyncServoRotation(-67+15*(cnt%2==1))
            time.sleep(2)
            man.Up.SetSyncServoRotation(-42-30*(cnt%2==1))
            time.sleep(2)
            man.Hand.SetSyncServoRotation(40)
            man.Side.SetAsyncServoRotation(-80)
            #man.Up.SetSyncServoRotation(-10)
            man.Side.SetAsyncServoRotation(-65)
            man.Up.SetSyncServoRotation(-90)
        cnt += 1



        if cv2.waitKey(100) & 0xFF == ord('q'):
            break