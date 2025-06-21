import torch
import easyocr
from ultralytics import YOLO
import cv2
import numpy as np

# Say Hello
def say_hello():
    print('Hello World' )

# check for GPU
def check_GPU():
    if torch.cuda.is_available() is not True:
        print("Warning! NO GPU DETECTED ")
        return 1
    else:
        print("GPU has been detected")
        return 0


# draw detected boxes
def draw_boxes(image, detections_list,thresh):
    boxes_list = []
    # Process each detected box
    for idx, box in enumerate(detections_list):
        x1, y1, x2, y2, score, class_id = box
        if score > thresh:
            # show score Percentage
            label = f"Plate {(score*100):.2f}"
            # Draw rectangle around detected plate
#            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            # Add text label
 #           cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # Add the box to the list
            boxes_list.append(box)
    return image, boxes_list


class PlateDetection:
    def __init__(self, model_name):
        # Load the trained model
        self.detections = None
        print("Uploading the ANPR model")
        self.license_plate_detector = YOLO(model_name)

        print("Loading completed")

    # Detect Plate in image
    def detect_plate(self, image, thresh = 0.7):
        # Perform detection
        self.detections = self.license_plate_detector.predict(image, verbose=False)[0]

        # get the image and list of boxes for the OCR
        image, boxes_list = draw_boxes(image, self.detections.boxes.data.tolist(),thresh)
        return image,boxes_list


class PlateEasyOCR:
    def __init__(self):
        # initializing OCR Object
        print('initializing OCR Object')
        self.reader = easyocr.Reader(['en'], gpu = True)
        print('OCR Object is ready')

    # process Type 1 plate
    def process_type_1(self, img):
        img = cv2.resize(img, (109, 57))

        # Sharpen filter
        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

        # Sharpen and apply threshold filters
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.filter2D(src=img, ddepth=-1, kernel=sharp_kernel)
        img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]

        # Process Admin Region
        admin_region = img[5:25, 20:55]
        admin_text = self.reader.readtext(admin_region, allowlist='0123456789')
        if admin_text:
            admin_num, admin_conf = admin_text[0][1], admin_text[0][2]
            if len(admin_num) < 1 or len(admin_num) > 2 or admin_conf < 0.78:
                return

            # Process Serial Region
            serial_region = img[25:60, 20:100]
            serial_text = self.reader.readtext(serial_region, allowlist='0123456789')
            if serial_text:
                serial_num, serial_conf = serial_text[0][1], serial_text[0][2]
                if (len(serial_num) == 4 or len(serial_num) == 5) and serial_conf >= 0.78:
                    print(admin_num + "-" + serial_num)
                    return admin_num, serial_num
        return


    # Process Type 2 plate
    def process_type_2(self, img):
        img = cv2.resize(img, (218, 39))

        # Sharpen filter
        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

        # Sharpen and apply threshold filters
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.filter2D(src=img, ddepth=-1, kernel=sharp_kernel)
        img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]

        # Process Admin Region
        admin_region = img[:, 75:120]
        admin_text = self.reader.readtext(admin_region, allowlist='0123456789')
        if admin_text:
            admin_num, admin_conf = admin_text[0][1], admin_text[0][2]
            if len(admin_num) < 1 or len(admin_num) > 2 or admin_conf < 0.78:
                return

            # Process Serial Region
            serial_region = img[:, 130:]
            serial_text = self.reader.readtext(serial_region, allowlist='0123456789')
            if serial_text:
                serial_num, serial_conf = serial_text[0][1], serial_text[0][2]
                if (len(serial_num) == 4 or len(serial_num) == 5) and serial_conf >= 0.78:
                    print(admin_num + "-" + serial_num)
                    return admin_num, serial_num
        return


    # Easy OCR Function for Reading The Plate Generally
    def read_plate(self,boxes_list, detection_image, thresh = 0.4):
        # list for all results
        results_list = []

        # process the boxes, idx could be needed later
        for idx, box in enumerate(boxes_list):
            x1, y1, x2, y2, score, class_id = box
            roi = detection_image[int(y1):int(y2), int(x1):int(x2)]
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, image_thresh = cv2.threshold(gray_image, 64,255, cv2.THRESH_BINARY)

            ocr_result  = self.reader.readtext(image_thresh)

            if ocr_result:
                ocr_text, ocr_confidence = ocr_result[0][1], ocr_result[0][2]
                if ocr_confidence > thresh:
                    results_list.append(ocr_text.upper())
                    print(ocr_text)
                    file_name = './saved_images/'+ ocr_text + '.jpg'

                    cv2.imwrite(file_name, roi )
        return results_list

    # For Reading Kuwaiti Plates
    def read_KW_plate(self, boxes_list, detection_image):
        # list for all OCR results
        results_list = []

        for idx, box in enumerate(boxes_list):
            x1, y1, x2, y2, score, class_id = box
            roi = detection_image[int(y1):int(y2), int(x1):int(x2)]
            # get the ratio of the plate
            ratio = roi.shape[0] / roi.shape[1]

            # if the width to length ratio is higher than 0.43 then process the plate is type 1
            if ratio >= 0.43:
                    self.process_type_1(roi)
            else: # else it is type 2
                    self.process_type_2(roi)
        return results_list

    def cvocr(self,boxes_list, detection_image, thresh = 0.4):
        # crop license plate
        results_list = []

        for idx, box in enumerate(boxes_list):
            x1, y1, x2, y2, score, class_id = box

            license_plate_crop = detection_image[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            license_plate_text  = self.reader.readtext(license_plate_crop_thresh)
            if license_plate_text:
                text, score = license_plate_text[0][1] , license_plate_text[0][2]
                print(text.upper())
                results_list.append((text.upper()))
        return  results_list