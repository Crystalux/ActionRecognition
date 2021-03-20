import matplotlib.pyplot as plt
import cv2
from preprocess.preprocess import get_frames

# visualising frames
path_to_video = './data/UCF11_updated_mpg/biking/v_biking_04/v_biking_04_03.mpg'


cap = cv2.VideoCapture(path_to_video)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Example', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

frames, len_v = get_frames(path_to_video, n_frames=16)

# Show frames extracted
plt.figure(figsize=(10, 10))
for idx, img in enumerate(frames):
    print(idx)
    plt.subplot(4, 4, idx+1)
    plt.imshow(img)
plt.title('Frames extracted')
plt.show()

