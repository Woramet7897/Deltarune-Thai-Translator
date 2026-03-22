import ollama

def test_vlm():
    print("🚀 ปลุก Moondream ให้ตื่นมาดูรูป...")
    try:
        # ลองส่งรูปที่คุณตัดไว้ให้มันอ่าน
        response = ollama.generate(
            model='moondream',
            prompt='What does the text in this image say?',
            images=['dialog_crop_0_0.png'] # ตรวจสอบว่าชื่อไฟล์ถูกต้อง
        )
        print("\n--- ผลลัพธ์จาก Moondream ---")
        print(response['response'])
        print("---------------------------")
    except Exception as e:
        print(f"❌ พังครับอาจารย์: {e}")

if __name__ == "__main__":
    test_vlm()