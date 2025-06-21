import threading
from queues_functions import *


if __name__ == '__main__':
    say_hello()
    check_GPU()
    print('Building Threads')
    # start the real processing
    p1 = threading.Thread(target=plate_detect)
    p2 = threading.Thread(target=read_plate)
    p3 = threading.Thread(target=display)
    p1.start()
    p2.start()
    p3.start()
    print('All threads started')
    p1.join()
    p2.join()
    p3.join()
    print("All threads has been terminated!")



