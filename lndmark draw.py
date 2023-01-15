import cv2
import mediapipe as m

camera= cv2.VideoCapture(0)
hand_rec=m.solutions.hands.Hands()
draw = m.solutions.drawing_utils  #land mark gula draw korar function
while True:
    _,video = camera.read()
    video=cv2.flip(video,1)
    rgb_video=cv2.cvtColor(video,cv2.COLOR_BGR2RGB)
    hand_detector=hand_rec.process(rgb_video)
    landmark=hand_detector.multi_hand_landmarks
    print(landmark)
#######################################################################
    if landmark:  #jodi landmark thake
        for i in landmark:   #jotogula land mark ase totobar cholbe loop
            draw.draw_landmarks(video,i)   #landmark draw kore "video" te print

############################################################

    cv2.imshow('Monitor',video)

    cv2.waitKey(1)
