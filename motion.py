import cv2
import math
import numpy as np
import time
from datetime import datetime
import mediapipe as mp

class AdvancedFingerCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.setup_camera()
        
        # Initialisation MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Variables de suivi
        self.last_motion_time = time.time()
        self.alert_delay = 5  # secondes
        self.finger_history = []
        self.history_length = 10
        self.stable_count = 0
        self.frame_shape = (720, 1280)  # Dimensions par défaut
        
    def setup_camera(self):
        """Configure la caméra pour une meilleure détection"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
        
    def detect_hands_mediapipe(self, frame):
        """Détection des mains avec MediaPipe (très précis)"""
        self.frame_shape = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        total_fingers = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Dessiner les landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Compter les doigts pour cette main
                fingers, finger_tips = self.count_fingers_mediapipe(hand_landmarks)
                total_fingers += fingers
                
                hand_label = handedness.classification[0].label
                hand_data = {
                    'landmarks': hand_landmarks,
                    'fingers': fingers,
                    'label': hand_label,
                    'finger_tips': finger_tips
                }
                hands_data.append(hand_data)
                self.draw_hand_info(frame, hand_data)
        
        # ✅ Vérification que multi_hand_landmarks n'est pas None
        hand_detected = bool(results.multi_hand_landmarks) if results.multi_hand_landmarks is not None else False
        
        return total_fingers, hands_data, hand_detected
    
    def count_fingers_mediapipe(self, landmarks):
        """Compte les doigts levés avec MediaPipe"""
        finger_tips = [4, 8, 12, 16, 20]  # Points des bouts des doigts
        finger_pips = [3, 6, 10, 14, 18]  # Points des articulations (PIP)
        finger_mcps = [2, 5, 9, 13, 17]   # Points des articulations (MCP)
        
        fingers = 0
        raised_fingers = []
        finger_tip_positions = []
        
        # Pouce
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        if landmarks.landmark[5].x < landmarks.landmark[17].x:  # Main droite
            thumb_raised = thumb_tip.x < thumb_ip.x
        else:  # Main gauche
            thumb_raised = thumb_tip.x > thumb_ip.x
        
        if thumb_raised:
            fingers += 1
            raised_fingers.append("Pouce")
            finger_tip_positions.append((int(thumb_tip.x * self.frame_shape[1]), int(thumb_tip.y * self.frame_shape[0])))
        
        # Autres doigts
        for i in range(1, 5):
            tip = landmarks.landmark[finger_tips[i]]
            pip = landmarks.landmark[finger_pips[i]]
            
            if tip.y < pip.y:
                fingers += 1
                finger_names = ["Index", "Majeur", "Annulaire", "Auriculaire"]
                raised_fingers.append(finger_names[i-1])
                finger_tip_positions.append((int(tip.x * self.frame_shape[1]), int(tip.y * self.frame_shape[0])))
        
        return fingers, finger_tip_positions
    
    def detect_hand_movement(self, current_hands_data, previous_hands_data=None):
        """Détecte le mouvement de la main entre les frames"""
        if previous_hands_data is None or not current_hands_data:
            return True, 0
        
        try:
            movement_score = 0
            current_landmarks = current_hands_data[0]['landmarks']
            previous_landmarks = previous_hands_data[0]['landmarks']
            
            for i in range(21):  # 21 points par main
                curr_x = current_landmarks.landmark[i].x
                curr_y = current_landmarks.landmark[i].y
                prev_x = previous_landmarks.landmark[i].x
                prev_y = previous_landmarks.landmark[i].y
                distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                movement_score += distance
            
            motion_detected = movement_score > 0.005
            return motion_detected, movement_score * 1000
            
        except (IndexError, AttributeError):
            return True, 0
    
    def draw_hand_info(self, frame, hand_data):
        landmarks = hand_data['landmarks']
        fingers = hand_data['fingers']
        label = hand_data['label']
        
        wrist = landmarks.landmark[0]
        text_x = int(wrist.x * frame.shape[1])
        text_y = int(wrist.y * frame.shape[0]) - 50
        
        cv2.rectangle(frame, (text_x-10, text_y-25), (text_x+150, text_y+10), (50, 50, 50), -1)
        cv2.putText(frame, f"{label}: {fingers} doigts", (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        for i, (x, y) in enumerate(hand_data['finger_tips']):
            cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)
            cv2.putText(frame, str(i+1), (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def draw_enhanced_overlay(self, frame, fingers, hands_data, motion_detected, immobility_time, movement_score, hand_detected):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "🤚 COMPTEUR DE DOIGTS AVANCE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
        status_text = "MAIN ACTIVE" if motion_detected else "MAIN IMMOBILE"
        cv2.putText(frame, f"Doigts leves: {fingers}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Statut: {status_text}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Immobilite: {immobility_time:.1f}s", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouvement: {movement_score:.1f}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        detection_status = "MAIN DETECTEE" if hand_detected else "PAS DE MAIN"
        detection_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"Detection: {detection_status}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
        
        alert_progress = min(immobility_time / self.alert_delay, 1.0)
        bar_width = 300
        bar_color = (0, 255, 0) if alert_progress < 0.7 else (0, 255, 255) if alert_progress < 0.9 else (0, 0, 255)
        cv2.rectangle(frame, (20, 200), (20 + int(bar_width * alert_progress), 215), bar_color, -1)
        cv2.rectangle(frame, (20, 200), (20 + bar_width, 215), (255, 255, 255), 1)
        if immobility_time >= self.alert_delay:
            cv2.putText(frame, "⚠️ MAIN IMMOBILE! BOUGEZ!", (150, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (130, 55), 15, (0, 0, 255), -1)
    
    def update_finger_history(self, fingers):
        self.finger_history.append(fingers)
        if len(self.finger_history) > self.history_length:
            self.finger_history.pop(0)
        if len(self.finger_history) == self.history_length:
            if len(set(self.finger_history)) == 1:
                self.stable_count += 1
            else:
                self.stable_count = 0
    
    def get_stable_finger_count(self):
        if len(self.finger_history) == 0:
            return 0
        if self.stable_count >= 3:
            return max(set(self.finger_history), key=self.finger_history.count)
        else:
            return self.finger_history[-1] if self.finger_history else 0
    
    def run(self):
        print("🤚 COMPTEUR DE DOIGTS AVANCE")
        print("=" * 50)
        previous_hands_data = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                fingers, hands_data, hand_detected = self.detect_hands_mediapipe(frame)
                
                self.update_finger_history(fingers)
                stable_fingers = self.get_stable_finger_count()
                
                motion_detected, movement_score = self.detect_hand_movement(hands_data, previous_hands_data)
                previous_hands_data = hands_data if hands_data else None
                
                current_time = time.time()
                if motion_detected and hand_detected:
                    self.last_motion_time = current_time
                immobility_time = current_time - self.last_motion_time
                
                self.draw_enhanced_overlay(frame, stable_fingers, hands_data, motion_detected, immobility_time, movement_score, hand_detected)
                
                cv2.imshow("🤚 Compteur de Doigts Avance - MediaPipe", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Programme termine")

def main():
    finger_counter = AdvancedFingerCounter()
    finger_counter.run()

if __name__ == "__main__":
    main()
