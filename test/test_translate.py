import easyocr
from deep_translator import GoogleTranslator

# 1. เตรียมตัวอ่าน (ใช้ภาษาอังกฤษ)
reader = easyocr.Reader(['en'])

# 2. อ่านตัวหนังสือจากรูปที่คุณ Crop มา
result = reader.readtext('dialog_crop_0_0.png', detail=0)
text_en = " ".join(result)
print(f"English: {text_en}")

# 3. ส่งไปแปลเป็นไทย
if text_en:
    translated = GoogleTranslator(source='en', target='th').translate(text_en)
    print(f"Thai: {translated}")