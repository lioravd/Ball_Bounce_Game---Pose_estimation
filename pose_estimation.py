import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import cv2
_NUM_KEYPOINTS = 17
def det_pose(input):
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    img = Image.fromarray(inp)
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)


    print(pose)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for i in range(0, _NUM_KEYPOINTS):
        draw.ellipse(
            xy=[
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill=(255, 0, 0))

        for i in range(9,11):
            draw.ellipse(
            xy=[
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill=(0, 255, 0))
    #img.save(args.output)
    #img.save(args.output)
    #print('Done. Results saved at', args.output)
    img.save("outo.jpg")
    return np.array(img)


class WebcamStream:
    # initialization method
    def __init__(self, stream_id=1):
        self.stream_id = stream_id  # default is 1 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True


# initializing and starting multi-threaded webcam input stream
webcam_stream = WebcamStream(stream_id=1) # 0 id for main camera
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0

while (True):

    # Capture the video frame
    # by frame

    if webcam_stream.stopped is True :
        break
    else:
        inp = webcam_stream.read()
    # Display the resulting frame
    cv2.imshow('output',det_pose(inp))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()