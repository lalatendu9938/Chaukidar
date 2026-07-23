import cv2
from ultralytics import YOLO
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

print("Project Chaukidar: YOLO26 Edition Loading...")


# USED MODEL FOR PROJECT 
model = YOLO('yolo_best.pt') 


#  AUDIO SETUP (SPECIES-SPECIFIC)
pygame.mixer.init()
SOUND_FILES = {
    "elephant": "Sounds/bee_swarm.mp3",
    "monkey": "Sounds/tiger growl.mp3",   
    "wild_boar": "Sounds/dogs.mp3",       
    "cow": "Sounds/siren.mp3",            
    "leopard": "Sounds/lion_roar.mp3"
    
}

ALARM_SOUNDS = {}
print(" Loading Audio Files...")
for animal, file_path in SOUND_FILES.items():
    if os.path.exists(file_path):
        ALARM_SOUNDS[animal] = pygame.mixer.Sound(file_path)
        print(f"  [+] Loaded sound for: {animal.upper()}")
    else:
        print(f"  [!] Missing sound for: {animal.upper()} (Path: {file_path})")

if ALARM_SOUNDS:
    print(" Audio System Ready!")
else:
    print(" ERROR: No sound files found!")

# 🎥 CAMERA & MONITORING (YOLO26 SPEED MODE)
cap = cv2.VideoCapture(0)
print("Camera Started! Scanning for threats with YOLO26...")

current_playing_animal = None 
frame_count = 0
last_results = None 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    if frame_count % 2 == 0:
        
        # imgsz=640 for high precision
        results = model.predict(frame, conf=0.75, imgsz=640, verbose=False)
        last_results = results 
        
        detected_animals = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                # Colab names: ['Elephant', 'Monkey', 'Wild_Boar', 'Cow', 'Deer', 'Leopard']
                animal_name = model.names[class_id].lower() 
                detected_animals.append(animal_name)

        # SMART DECISION LOGIC (Threat vs Non-Threat)

        THREAT_ANIMALS = ["elephant", "monkey", "wild_boar", "leopard", "cow"]
        NON_THREAT_ANIMALS = ["deer"]

        if detected_animals:
            primary_animal = detected_animals[0]

            # CASE 1: RED ALERT (Threats)
            if primary_animal in THREAT_ANIMALS:
                if primary_animal in ALARM_SOUNDS and current_playing_animal != primary_animal:
                    if current_playing_animal and current_playing_animal in ALARM_SOUNDS:
                        ALARM_SOUNDS[current_playing_animal].stop()

                    print(f"\n YOLO26 ALERT: {primary_animal.upper()} detected! Playing alarm...")
                    ALARM_SOUNDS[primary_animal].play(-1)
                    current_playing_animal = primary_animal

            # CASE 2: SAFE ZONE (Deer)
            elif primary_animal in NON_THREAT_ANIMALS:
                if current_playing_animal and current_playing_animal in ALARM_SOUNDS:
                    ALARM_SOUNDS[current_playing_animal].stop()
                    current_playing_animal = None

                print(f"YOLO26 INFO: {primary_animal.upper()} (Non-Threat). All safe.")

        else:
            #  CASE 3: ALL CLEAR
            if current_playing_animal:
                print(" Status: All Clear...")
                if current_playing_animal in ALARM_SOUNDS:
                    ALARM_SOUNDS[current_playing_animal].stop()
                current_playing_animal = None

    # DISPLAY (YOLO26 is naturally smooth)

    if last_results is not None:
        cv2.imshow('Chaukidar AI - YOLO26 Ultra Fast', last_results[0].plot())
    else:
        cv2.imshow('Chaukidar AI - YOLO26 Ultra Fast', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if current_playing_animal and current_playing_animal in ALARM_SOUNDS:
            ALARM_SOUNDS[current_playing_animal].stop()
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("\n System Shut Down. Great Job Lead!")