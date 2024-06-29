import cv2
import numpy as np
import argparse
import os

def overlay_sunglasses(image, startX, startY, endX, endY, sunglasses, scale, height_scale=0.5, x_offset=0, y_offset=0):
    (h, w) = endY - startY, endX - startX
    overlay_h = int(height_scale * h)
    overlay_w = int(scale * w)
    if overlay_h > h:
        overlay_h = h

    overlay = cv2.resize(sunglasses, (overlay_w, overlay_h))
    (h_o, w_o) = overlay.shape[:2]

    # 선글라스 위치
    eye_level_y = startY + y_offset + (h // 2) - 5
    eye_level_x = startX + x_offset

    y, x = max(0, eye_level_y - h_o // 2), max(0, eye_level_x + (w - w_o) // 2)
    y1, y2 = max(0, y), min(y + h_o, image.shape[0])
    x1, x2 = max(0, x), min(x + w_o, image.shape[1])

    overlay_crop = overlay[0:(y2 - y1), 0:(x2 - x1)]
    alpha_s = overlay_crop[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (alpha_s * overlay_crop[:, :, c] +
                                  alpha_l * image[y1:y2, x1:x2, c])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-g", "--glasses", required=True, help="path to sunglasses image")
    ap.add_argument("-s", "--scale", type=float, default=1.0, help="scale factor for the sunglasses")
    ap.add_argument("--height_scale", type=float, default=0.2, help="scale factor for the height of the sunglasses")
    ap.add_argument("--x_offset", type=int, default=0, help="horizontal offset for the sunglasses")
    ap.add_argument("--y_offset", type=int, default=0, help="vertical offset for the sunglasses")
    ap.add_argument("-o", "--output", required=False, default="output_folder", help="folder to save the output image")
    args = vars(ap.parse_args())

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # load the input image and construct an input blob for the image
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # load the sunglasses image
    sunglasses = cv2.imread(args["glasses"], cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        print(f"Couldn't load the sunglasses image from \"{args['glasses']}\"")
        exit(1)

    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # filter detections by confidence and keep the one with the highest confidence
    max_confidence = 0
    best_box = None
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > args["confidence"] and confidence > max_confidence:
            max_confidence = confidence
            best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    
    # if a valid detection was found
    if best_box is not None:
        (startX, startY, endX, endY) = best_box.astype("int")
        
        # 얼굴 위에 선글라스를 오버레이
        overlay_sunglasses(image, startX, startY, endX, endY, sunglasses, args["scale"], args["height_scale"], args["x_offset"], args["y_offset"])

    # create the output directory if it does not exist
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    # output 이미지 저장 & 아무 버튼 눌러서 실행 중지
    output_path = os.path.join(args["output"], "output_image.png")
    cv2.imwrite(output_path, image)
    print(f"[INFO] output saved to {output_path}")

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
<실행 명령어>
python dnn_image_sunglasses.py -i input_folder/image.png -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel \
-g sunglasses_image.png -s 1.0 --height_scale 0.2 --x_offset 0 --y_offset -10 -o output_folder
"""