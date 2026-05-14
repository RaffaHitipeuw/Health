import cv2
import numpy as np

class AdaptiveROIManager:


    def __init__(self):
        self.perfusion_map = None
        self.skin_mask = None
        
    def segment_skin(self, frame_bgr):


        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        

        skin_ycrcb = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
        
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        

        skin_hsv = (H >= 0) & (H <= 25) & (S >= 20) & (S <= 150)
        
        self.skin_mask = (skin_ycrcb | skin_hsv).astype(np.uint8) * 255
        return self.skin_mask

    def update_perfusion_mapping(self, frame_bgr, face_landmarks, h, w):


        if self.skin_mask is None:
            self.segment_skin(frame_bgr)
            


        return self.skin_mask

    def get_adaptive_roi(self, frame_bgr, landmarks, landmark_ids, h, w):


        pts = []
        for idx in landmark_ids:
            pt = landmarks.landmark[idx]
            pts.append([int(pt.x * w), int(pt.y * h)])
        pts = np.array(pts, dtype=np.int32)
        

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        

        if self.skin_mask is None:
            self.segment_skin(frame_bgr)
            
        refined_mask = cv2.bitwise_and(poly_mask, self.skin_mask)
        

        if cv2.countNonZero(refined_mask) < 10:
            return None
            
        mean_val = cv2.mean(frame_bgr, mask=refined_mask)

        return mean_val[2], mean_val[1], mean_val[0], refined_mask
