import cv2
import mediapipe as m
hand_rec= m.solutions.hands.Hands()    #media pipe er hand recognizer function call
camera=cv2.VideoCapture(0)


while True:
    _,video=camera.read()
    video=cv2.flip(video,1)
    rgb_video=cv2.cvtColor(video, cv2.COLOR_BGR2RGB)   #video color RGB te nisi
    hand_detector= hand_rec.process(rgb_video)   #rgb video k process kore hand detect kortese
    landmark= hand_detector.multi_hand_landmarks  #hand theke land mark gula findout
    print(landmark)
    cv2.imshow('Monitor',video)
    cv2.waitKey(1)
