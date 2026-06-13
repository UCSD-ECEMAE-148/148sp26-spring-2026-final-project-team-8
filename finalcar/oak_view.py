import cv2
import depthai as dai

# Resolution sent over USB — smaller = less lag. Try (640, 360) or (320, 180).
STREAM_SIZE = (4033, 300)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera)
    cam.build(dai.CameraBoardSocket.CAM_A)

    video_queue = cam.requestOutput(
        size=STREAM_SIZE,
        type=dai.ImgFrame.Type.BGR888p,
        fps=15,
    ).createOutputQueue()

    pipeline.start()

    cv2.namedWindow("OAK-D Preview", cv2.WINDOW_NORMAL)

    while pipeline.isRunning():
        frame_msg = video_queue.get()
        frame = frame_msg.getCvFrame()

        cv2.imshow("OAK-D Preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()

