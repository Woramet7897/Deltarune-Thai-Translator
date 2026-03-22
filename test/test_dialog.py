import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

img_path = 'test_game4.png'
original_frame = cv2.imread(img_path)
results = model.predict(source=original_frame, conf=0.5)

def remove_white_border(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    col_mean = np.mean(gray, axis=0)
    row_mean = np.mean(gray, axis=1)
    
    # print ดูค่าบนล่างว่าเส้นม่วงมี mean เท่าไหร่
    print(f"row_mean first 5: {row_mean[:5].round(1).tolist()}")
    print(f"row_mean last  5: {row_mean[-5:].round(1).tolist()}")
    
    threshold = 20
    
    cols = np.where(col_mean <= threshold)[0]
    rows = np.where(row_mean <= threshold)[0]
    
    if len(cols) == 0 or len(rows) == 0:
        return crop
    
    x1, x2 = cols[0], cols[-1] + 1
    y1, y2 = rows[0], rows[-1] + 1
    
    return crop[y1:y2, x1:x2]

for i, r in enumerate(results):
    for j, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # ภาพดิบ
        crop_img = original_frame[y1:y2, x1:x2]
        crop_name = f'dialog_crop_{i}_{j}.png'
        cv2.imwrite(crop_name, crop_img)
        print(f'Saved: {crop_name}')
        
        # ภาพหลังลบขอบขาว
        clean_img = remove_white_border(crop_img)
        clean_name = f'dialog_clean_{i}_{j}.png'
        cv2.imwrite(clean_name, clean_img)
        print(f'Saved: {clean_name}')
        
        # แสดงเปรียบเทียบ
        cv2.imshow('Crop (Original)', crop_img)
        cv2.imshow('Clean (Border Removed)', clean_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()