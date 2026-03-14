import time
#from commands_new import execute
import cv2
import numpy as np
from cv2 import aruco
from srv import Manipulator
def search_for(man, index):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.01
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    FRAME_CENTER = np.array((1920 / 2, 1080 / 2))
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    man.Up.SetSyncServoRotation(-90)
    man.RHand.SetSyncServoRotation(0)
    for side in range(100, -50, -40):
        man.Side.SetSyncServoRotation(side)
        for ang in range(-100, 100, 5):
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 0)
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            man.Base.SetSyncServoRotation(ang)
            if ids is not None:
                if index in ids.flatten():
                    cap.release()
                    cv2.destroyAllWindows()
                    return ang, side
            cv2.imshow("frame", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

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
    detector = aruco.ArucoDetector(aruco_dict, parameters)
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and index in ids.flatten():
            corners = corners[list(ids).index(index)][0]
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
                        side_ang += abs(2*dy/100)
                    else:
                        side_ang -= abs(2*dy/100)
                if (abs(da)) > 5000:
                    done = False
                    if da > 0:
                        up_ang -= abs(2*da/5000)
                    else:
                        up_ang += abs(2*da/5000)
            print(dx, dy, da, base_ang, side_ang, up_ang)
            angle = np.clip(angle, 0, 90)

            #print(angle)
            base_ang = round(base_ang)
            side_ang = round(side_ang)
            up_ang = round(up_ang)
            #print(base_ang, side_ang, up_ang)
            if done:
                final = True
                if angle > 10:
                    rhand_ang = angle
                    man.RHand.SetSyncServoRotation(rhand_ang)
                man.Hand.SetSyncServoRotation(50)
                side_ang -= 90
                up_ang -= 15
                print("target_angle:", angle)
                man.Up.SetSyncServoRotation(up_ang)
                man.Side.SetSyncServoRotation(side_ang)


                up_ang += 60
                man.Up.SetSyncServoRotation(up_ang)
                man.Hand.SetSyncServoRotation(-80)
                up_ang -= 70
                man.Up.SetSyncServoRotation(up_ang)
                man.RHand.SetSyncServoRotation(0)
                time.sleep(3)

                break
        else:
            t += 1
        if t >= 50:
            break
        #man.Base.SetSyncServoRotation(base_ang)
        #man.Side.SetSyncServoRotation(side_ang)
        #man.RHand.SetSyncServoRotation(rhand_ang)
        #man.Up.SetSyncServoRotation(up_ang)

        cv2.imshow('Video', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    #cap.release()

if __name__ == '__main__':
    man = Manipulator()
    base, side = search_for(man, 6)
    take_cube(man, 6, base, side)
