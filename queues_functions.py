from utils import *
import queue

# resize parameters
scale_percent = 40  # percent of original size
detection_thresh = 0.4 # plate detection threshold, if the score below this the model will not detect

stop_threads = False

frame_q = queue.Queue()
OCR_q = queue.Queue()

def display():
    # print("Start Displaying")
    global stop_threads
    while not stop_threads:
        if not frame_q.empty():
            frame = frame_q.get()
            frame = cv2.resize(frame, (720, 480))
            cv2.imshow("Display'", frame)
            # print ('Done Displaying')
            # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            cv2.destroyAllWindows()
            break
    print('quit display')

def plate_detect():
    model_name = 'ANPR.pt'
    plate_detector = PlateDetection(model_name)

    # initialize cap object and first frame

    cap = cv2.VideoCapture('license_plate.mp4')
    # Get the original resolution of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #cap = cv2.VideoCapture('station28.mp4')
    #cap = cv2.VideoCapture(0)
    global stop_threads
    # start processing
    while not stop_threads:
        ret, frame = cap.read()     # check availability of a frame
        if ret is False:                    # break if there is no frame
            stop_threads = True
            print('quit detect')
            break


        # resize image
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # Detect Plates in the video
        image_with_detections, boxes_list = plate_detector.detect_plate(frame, detection_thresh)
        frame_q.put(image_with_detections)
        OCR_q.put((boxes_list,image_with_detections))

def read_plate():
    # Read The Plates
        global stop_threads
        reader = PlateEasyOCR()
        while not stop_threads:
            try:
                # Try to get the data from the queue with a timeout
                boxes_list, image_np_with_detections = OCR_q.get(timeout=3)
                #reader.read_KW_plate(boxes_list, image_np_with_detections)
                #reader.read_plate(boxes_list, image_np_with_detections)
                reader.cvocr(boxes_list, image_np_with_detections)

            except queue.Empty:
                # If the queue is empty and times out, check `stop_threads` again
                if stop_threads:
                    break



