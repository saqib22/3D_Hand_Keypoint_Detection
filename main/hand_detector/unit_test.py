import cv2
from YoloAkash import YOLO_AKASH

detector = YOLO_AKASH('best.pt')

cap = cv2.VideoCapture('GROUND-HEAD_24-Sept_1256-1305_test_plain_video_8aefa919-9143-43e1-bcb4-156428fbd14b_2020-09-24T12:56:10+05:00_2020-09-24T12:56:10+05:00.avi')

write_w = int(cap.get(3))
write_h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

writer = cv2.VideoWriter('inference/output/video.mp4', fourcc, 15, (write_w, write_h))

counter = 0
while(True):
	print(counter)
	ret,frame = cap.read()
	if not ret:
		break

	counter += 1
	bbox, confs, labels = detector.detect_image(frame)

	for box in bbox:
		frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

	writer.write(frame)

cap.release()
writer.release()