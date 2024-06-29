import cv2
import numpy as np
import argparse
import os

def overlay_sunglasses(image, startX, startY, endX, endY, sunglasses, scale, height_scale=0.5):
    (h, w) = endY - startY, endX - startX
    overlay_h = int(height_scale * h)
    overlay_w = int(scale * w)
    if overlay_h > h:
        overlay_h = h

    overlay = cv2.resize(sunglasses, (overlay_w, overlay_h))
    (h_o, w_o) = overlay.shape[:2]

    # 선글라스 위치
    eye_level_y = startY - 30 + (h // 2) + 20
    eye_level_X = startX - 9

    y, x = max(0, eye_level_y - h_o // 2), max(0, eye_level_X + (w - w_o) // 2)
    y1, y2 = max(0, y), min(y + h_o, image.shape[0])
    x1, x2 = max(0, x), min(x + w_o, image.shape[1])

    overlay_crop = overlay[0:(y2 - y1), 0:(x2 - x1)]
    alpha_s = overlay_crop[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (alpha_s * overlay_crop[:, :, c] +
                                  alpha_l * image[y1:y2, x1:x2, c])

def main(args):
    # Caffe 모델과 프로토텍스트 파일을 로드하여 딥러닝 네트워크를 초기화
    print("Loading Caffe model...")
    print("Prototxt file:", args.prototxt)
    print("Model file:", args.model)
    # args.prototxt와 args.model은 프로토텍스트 파일과 사전 학습된 모델 파일의 경로
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    print("Caffe model loaded.")

    # 입력 비디오 소스 초기화
    # args.input이 정수인 경우, 웹캠을 사용
    # 비디오 캡처가 성공적으로 열리지 않으면 오류 메시지를 출력하고 프로그램을 종료
    cap = cv2.VideoCapture(int(args.input))

    if not cap.isOpened():
        print(f"OpenCV: Couldn't read video stream from file \"{args.input}\"")
        return
    
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 선글라스 이미지 로드
    # args.glasses는 선글라스의 이미지 파일 경로
    # 이미지를 로드할 수 없으면 오류 메시지를 출력하고 프로그램을 종료
    sunglasses = cv2.imread(args.glasses, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        print(f"Couldn't load the sunglasses image from \"{args.glasses}\"")
        return
    
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    # 비디오 캡처에서 프레임을 읽음
    # 읽기가 실패하면, 루프를 종료
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        (h, w) = frame.shape[:2]
        # 딥러닝 모델에 입력할 블롭(blob)을 생성
        # 블롭 : 동일한 방식으로 전처리된 동일한 너비, 높이 및 채널 수를 가진 하나 이상의 이미지
        # 블롭을 네트워크에 입력하고 얼굴 감지를 수행
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # 모든 감지된 객체에 대해 신뢰도를 확인
            # 신뢰도가 0.5보다 낮으면 해당 객체를 무시
            if confidence < 0.5:
                continue

            # 얼굴의 경계 상자를 가져와 실제 이미지 크기에 맞게 조정
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 얼굴 위에 선글라스를 오버레이
            overlay_sunglasses(frame, startX, startY, endX, endY, sunglasses, args.scale, args.height_scale)

        cv2.imshow("Frame", frame)
        # e를 누르면 비디오 중단
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
        # s를 누르면 사진 저장
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            filename = os.path.join(output_folder, f"output_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input video file or webcam id")
    parser.add_argument("-o", "--output", required=True, help="path to output folder")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="scale factor for the sunglasses")
    parser.add_argument("-g", "--glasses", required=True, help="path to sunglasses image")
    parser.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    parser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    parser.add_argument("--height_scale", type=float, default=0.2, help="scale factor for the height of the sunglasses")
    args = parser.parse_args()

    main(args)


"""
<실행 명령어>
python dnn_video_sunglasses.py -i 0 -o output_folder -s 1.0 -g sunglasses_image.png \
-p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel
"""