import numpy as np
import cv2 as cv

def select_img_from_video(input_file, board_pattern, select_all=False, wait_msec=10):
    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow('Camera Calibration', display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == 27:                  # 'ESC' key: Exit (Complete image selection)
                break
            elif key == ord(' '):          # 'Space' key: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow('Camera Calibration', display)
                key = cv.waitKey()
                if key == 27: # ESC
                    break
                elif key == ord('\r'):
                    img_select.append(img) # 'Enter' key: Select the image

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be 'np.float32'

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)



if __name__ == '__main__':
    input_file = 'chchch_board.mp4'
    board_pattern = (10, 7)
    board_cellsize = 0.025

    img_select = select_img_from_video(input_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # 영상 변환을 위해 원본 동영상의 프레임의 너비와 높이, 초당 프레임 수(fps) 가져오기
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    # 만화(Cartoon) 스타일로 변환된 동영상을 저장할 파일 설정
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Corrected_chchess.avi', fourcc, fps, (frame_width, frame_height))

    # Run distortion correction
    show_rectify = True
    map1, map2 = None, None
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Rectify geometric distortion (Alternative: cv.undistort() -> 픽셀마다 함수를 실행해야 하는데, 함수를 실행할 때마다 매핑 테이블을 만들어야 하기 때문에 시간이 오래걸린다.)
        info = "Original"
        if show_rectify:
            if map1 is None or map2 is None:
                # 이 단계에서 미리 매핑 테이블을 만들어 놓아, 동영상을 처리하는데 시간을 절약할 수 있다.
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
            img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Rectified"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        out.write(img)

        # Show the image and process the key event
        cv.imshow("Distortion Correction", img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break
        elif key == ord('\t'):
            show_rectify = not show_rectify

    video.release()
    out.release()
    cv.destroyAllWindows()
