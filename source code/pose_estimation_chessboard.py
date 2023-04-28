import numpy as np
import cv2 as cv

# The given video and calibration data
input_file = './data/chessboard.avi'
K = np.array([[1.68269938e+03, 0.00000000e+00, 9.47470747e+02],
    [0.00000000e+00, 1.64114821e+03, 5.20849555e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([2.38892300e-01, -1.35836436e+00, -3.11281706e-03, -1.65354415e-03, 3.01323337e+00])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_file

# Prepare a 3D box for simple AR
box_lower = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])
box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)


        
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)


        '''
        # Draw the box on the image

        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2) # bottom of the box
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2) # top of the box
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2) # vertical line
        '''
        # Draw a number

        # pt1 ----- pt2
        #            |
        #            |
        #            |
        # pt5 ----- pt6
        #  |
        #  |
        #  |
        # pt4 ----- pt3


        num_col = (255, 255, 255)

        pt1 = (line_upper[0].flatten() + line_lower[0].flatten()) / 2
        pt2 = (line_upper[1].flatten() + line_lower[1].flatten()) / 2

        pt3 = (line_upper[2].flatten() + line_lower[2].flatten()) / 2
        pt4 = (line_upper[3].flatten() + line_lower[3].flatten()) / 2

        pt5 = (pt1 + pt4) / 2
        pt6 = (pt2 + pt3) / 2

        cv.line(img, np.int32(pt1) , np.int32(pt2), num_col, 2)
        cv.line(img, np.int32(pt2) , np.int32(pt6), num_col, 2)
        cv.line(img, np.int32(pt6) , np.int32(pt5), num_col, 2)
        cv.line(img, np.int32(pt5) , np.int32(pt4), num_col, 2)
        cv.line(img, np.int32(pt4) , np.int32(pt3), num_col, 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()