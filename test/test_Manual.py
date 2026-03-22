# main
import cv2
import torch
import json
import easyocr
import re
import ollama
import hashlib
import sys
import time
import mss
import keyboard
import tkinter as tk
import threading
import queue
import numpy as np
from ultralytics import YOLO
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from thefuzz import process as fuzzy_process
from spellchecker import SpellChecker

spell = SpellChecker()

# --- 1. Setup Models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO('best.pt')

# ✅ อนุญาตให้มีวงเล็บ () เพื่อดักจับบริบทของ Deltarune
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,:'-* ()$"
reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

last_hash = None
popup_queue = queue.Queue()
popup_thread_started = False

def load_lore():
    try:
        with open('lore.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

LORE_DATA = load_lore()

# --- 2. Temmie dialect lexicon ---
TEMMIE_DIALECT: dict = {
    "awa":  "I am",
    "sorz": "sorry",
    "s0rz": "sorry",
    "ver":  "very",
    "yec":  "very",   
    "tem":  "Temmie",
    "hav":  "have",
    "luv":  "love",
    "r":    "are",
}

# --- 3. Whitelist สำหรับ Context-Aware Fix ---
IS_FOLLOWERS = {
    "upstairs", "downstairs", "here", "there", "that", "this",
    "not", "the", "a", "an", "what", "where", "gone", "ready",
    "over", "in", "on", "it", "my", "your",
    "good", "bad", "true", "false", "just", "so", "too", "very",
    "dangerous", "impossible", "fun", "wrong", "right", "time",
    "enough", "fine", "okay", "ok", "nothing", "everything",
    "mine", "yours", "his", "hers", "ours",
    "coming", "going", "happening", "possible"
}

# ✅ OCR Patches เฉพาะ pattern ที่ปลอดภัยทุกบริบท
OCR_PATCHES = {
    # You/your variants
    "Vou": "You", "vou": "you",
    "Vov": "you", "vov": "you",
    "4ou": "you", "4Wu": "you", "4wu": "you",
    "Yov": "you", "yov": "you",

    # What/want
    "vVhat": "What", "vvhat": "what", "Nhat": "What",
    "vvant": "want", "vVant": "want",

    # short token corrections
    "9o": "go", "9o,": "go,",
    "Al!": "All",
    "Wil!": "Will",
    "wil!": "will",
    "S4o4": "$40!",
    "S4o": "$40",
    "84o": "$40",
    "84o!": "$40!",
    "8 4o": "$40",

    # Our variants
    "Qur": "Our", "qur": "our", "Uur": "Our",
    "Mur": "Our", "mur": "our",
    "0ur": "our", "0Ur": "Our",

    # same variants
    "Samp": "same", "samp": "same",
    "samo": "same", "sam8": "same",
    "samc": "same", "samе": "same",

    # look variants
    "lowked": "looked", "lopked": "looked",
    "lookcd": "looked", "lookeد": "looked",

    # They
    "Theu": "They", "theu": "they",
    "Thev": "They", "thev": "they",

    # about
    "ahout": "about", "ahont": "about",

    # punctuation noise
    "upstairsl": "upstairs!", "upstairs1": "upstairs!",
    "upstairsy": "upstairs!",
    "uS": "us", "US,": "us,",

    # welcome variants
    "Wleloome": "Welcome",
    "wleloome": "welcome",
    "Welcorne": "Welcome",  
    "Wellcome": "Welcome",  

    # contraction recovery
    "('s": "(It's",

}

# ✅ OCR fixes เฉพาะเกม/บริบท โหลดจาก lore.json เพื่อแก้ได้โดยไม่ต้องแตะโค้ด
LORE_OCR_FIXES = LORE_DATA.get("ocr_fixes", {}) if isinstance(LORE_DATA, dict) else {}

def build_proper_noun_lexicon(lore_data: dict) -> list:
    names = ["Kris", "Ralsei", "Susie", "Lancer", "Jevil", "Spamton", "Temmie", "Noelle", "Berdly", "Rouxls", "Darkner", "Lightner", "Queen", "Seam", "Anya", "Rudinn"]
    for en_key in lore_data.get("characters", {}).keys():
        for word in en_key.split():
            if len(word) >= 4 and word[0].isupper(): names.append(word)
    return list(dict.fromkeys(names))

PROPER_NOUN_LEXICON = build_proper_noun_lexicon(LORE_DATA)

SPELL_IGNORE = {
    "hathy", "hatty", "kris", "ralsei", "susie", "lancer",
    "jevil", "spamton", "temmie", "noelle", "berdly", "rouxls",
    "darkner", "lightner", "seam", "rudinn", "hee"
}


def context_aware_fix(text: str) -> str:
    tokens = re.findall(r"[\w']+|[^\w']+", text)  
    result = []
    
    for i, tok in enumerate(tokens):
        if re.fullmatch(r'15|I5|l5', tok):
            next_word = ""
            for j in range(i+1, len(tokens)):
                w = tokens[j].strip().lower().rstrip('.,!?y')
                if w:
                    next_word = w
                    break
            print(f"  [15 debug] next_word='{next_word}' in IS_FOLLOWERS={next_word in IS_FOLLOWERS}")
            if next_word in IS_FOLLOWERS:
                result.append("is")
            else:
                result.append(tok)  
                
        elif re.fullmatch(r'Ho|ho', tok):
            prev_meaningful = ""
            for j in range(i-1, -1, -1):
                w = tokens[j].strip()
                if w:
                    prev_meaningful = w
                    break
            if prev_meaningful in {"*", ",", ".", ""} or i == 0:
                result.append("No" if tok == "Ho" else "no")
            else:
                result.append(tok)
                
        elif re.fullmatch(r'Mu|mu', tok):
            next_word = ""
            for j in range(i+1, len(tokens)):
                w = tokens[j].strip().lower().rstrip('.,!?')
                if w:
                    next_word = w
                    break
            
            MY_FOLLOWERS = {"heart", "name", "friend", "life", "soul", "eyes"}
            YOU_FOLLOWERS = {"felt", "are", "have", "can", "should", "will", "about"}
            
            if next_word in MY_FOLLOWERS:
                result.append("My" if tok[0].isupper() else "my")
            elif next_word in YOU_FOLLOWERS:
                result.append("You" if tok[0].isupper() else "you")
            else:
                result.append(tok)
        else:
            result.append(tok)
            
    return "".join(result)


def spell_fix(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    result = []
    for tok in tokens:
        if not re.fullmatch(r"[A-Za-z]+", tok) or len(tok) < 4:
            result.append(tok)
            continue
        if tok.lower() in SPELL_IGNORE:
            result.append(tok)
            continue
        if tok[0].isupper():
            result.append(tok)
            continue
        candidate = spell.correction(tok.lower())
        if candidate and candidate != tok.lower():
            result.append(candidate)
        else:
            result.append(tok)
    return "".join(result)


def fuzzy_fix_ocr(raw_text: str) -> str:
    text = raw_text.replace("|", "I").replace("#", "*").replace("0", "o")
    text = text.replace("kindly-", "kindly.")
    text = re.sub(r"It'\s+(\d)", r"It's", text)  # It' 5 -> It's
    text = re.sub(r"it'\s+(\d)", r"it's", text)
    text = re.sub(r"It'\s+\$", "It's", text)  # It' $ -> It's
    text = re.sub(r"it'\s+\$", "it's", text)
    text = re.sub(r'\bAl!\b', 'All', text)
    text = re.sub(r'^\d+\s*', '', text)  # ลบตัวเลขที่ขึ้นต้นประโยค
    text = re.sub(r'([a-zA-Z])l!([a-zA-Z])', r'\1ll\2', text)  
    text = re.sub(r'([a-zA-Z])l!(\s|$|\))', r'\1ll\2', text)  

    # แก้ v/V ที่นำหน้าสระ -> มักเป็น y/Y จาก pixel font
    text = re.sub(r'\bv([aeiou])', r'y\1', text)
    text = re.sub(r'\bV([aeiou])', r'Y\1', text)
    
    # ✅ THE MASTERPIECE REGEX: เปลี่ยน : ท้ายคำให้กลายเป็น .
    text = re.sub(r'([a-z])\:(\s|$|\))', r'\1.\2', text)
    
    # ลบเลข 1 โดดๆ ยกเว้นถ้าเป็นตัวเลขในประโยค
    text = re.sub(r'(?<![0-9])\b1\b(?![0-9])', '', text)
    # ลบกลุ่มตัวอักษรสั้นๆ ที่เป็น noise เช่น J, E, JE (คง I/i ไว้)
    text = re.sub(r'(?<![A-Za-z])\b[KkTtLlJjEe]{1,2}\b(?![A-Za-z])', '', text)
    
    text = re.sub(r'^1\s+([A-Z])', r'N\1', text) 
    text = re.sub(r'([a-z])[1l](\s|$)', r'\1!\2', text) 

    text = context_aware_fix(text)

    for wrong, right in OCR_PATCHES.items():
        escaped_wrong = re.escape(wrong)
        if re.fullmatch(r"[A-Za-z0-9']+", wrong):
            text = re.sub(rf'\b{escaped_wrong}\b', right, text)
        else:
            # Punctuation-containing patterns (e.g., "Al!", "guess 's")
            # are safer with direct escaped replacement.
            text = re.sub(escaped_wrong, right, text)

    # Apply lore-driven fixes for game-specific OCR noise.
    for wrong, right in LORE_OCR_FIXES.items():
        text = text.replace(wrong, right)

    tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9']+", text)
    step2 = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z0-9']+", tok):
            step2.append(TEMMIE_DIALECT.get(tok.lower(), tok))
        else:
            step2.append(tok)
    text = "".join(step2)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']*|[^A-Za-z]+", text)
    step3 = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9']*", tok) and len(tok) >= 4:
            hit = fuzzy_process.extractOne(tok, PROPER_NOUN_LEXICON)
            step3.append(hit[0] if hit and hit[1] >= 82 else tok)
        else:
            step3.append(tok)
    text = "".join(step3)

    text = spell_fix(text)

    return re.sub(r"\s+", " ", text).strip()


def build_ocr_variants(crop):
    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted = cv2.bitwise_not(otsu) 
    return [gray, inverted] 


# ลบขอบสว่าง (กรอบ dialog) โดยตัดจากขอบเข้ามาเรื่อยๆ จนเจอแถวที่มืดกว่า threshold
# ใช้ edge-trimming แทน filter ทั้งภาพ เพื่อไม่ตัดตัวอักษรสว่างข้างใน
def remove_white_border(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    row_mean = np.mean(gray, axis=1)
    col_mean = np.mean(gray, axis=0)

    border_threshold = 30  # ลด threshold เพื่อกินเส้นขาวแนวตั้งที่ยังค้าง

    top = 0
    while top < len(row_mean) and row_mean[top] > border_threshold:
        top += 1

    bottom = len(row_mean) - 1
    while bottom > top and row_mean[bottom] > border_threshold:
        bottom -= 1
    bottom += 1

    left = 0
    while left < len(col_mean) and col_mean[left] > border_threshold:
        left += 1

    right = len(col_mean) - 1
    while right > left and col_mean[right] > border_threshold:
        right -= 1
    right += 1

    if top >= bottom or left >= right:
        return crop

    return crop[top:bottom, left:right]


def split_into_lines(crop):
    crop = remove_white_border(crop)
    
    h, w = crop.shape[:2]
    
    # scan เฉพาะกลางภาพ 60% เพื่อหลบกรอบข้างๆ
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.05)
    
    if h <= margin_y * 2 or w <= margin_x * 2:
        inner = crop
        margin_x = 0
        margin_y = 0
    else:
        inner = crop[margin_y:h-margin_y, margin_x:w-margin_x]
    
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    row_sums = binary.sum(axis=1)
    threshold = 255 * int(inner.shape[1] * 0.01)
    
    in_text = False
    line_starts, line_ends = [], []
    
    for y, s in enumerate(row_sums):
        if s > threshold and not in_text:
            in_text = True
            line_starts.append(y)
        elif s <= threshold and in_text:
            in_text = False
            line_ends.append(y)
    if in_text:
        line_ends.append(len(row_sums))

    print(f"[Split] {len(line_starts)} segments after border removal {h}x{w}")
    
    lines = []
    min_line_height = max(10, int(inner.shape[0] * 0.02))
    
    for y1, y2 in zip(line_starts, line_ends):
        if y2 - y1 > min_line_height:
            # map กลับพิกัดเดิม + เอาความกว้างเต็ม
            y1_orig = max(0, y1 + margin_y - 2)
            y2_orig = min(h, y2 + margin_y + 2)
            lines.append(crop[y1_orig:y2_orig, :])
    
    return lines if lines else [crop]


def ocr_best_text(crop, box_index=0):
    inner_box = remove_white_border(crop)
    inner_box = cv2.resize(inner_box, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    temp_path = f"vlm_input_{box_index}.png"
    cv2.imwrite(temp_path, inner_box)

    try:
        response = ollama.generate(
            model='moondream',
            prompt="""Read ALL the text in this game dialogue box from top to bottom.
Include every line of text you see.
Output only the text, nothing else.""",
            images=[temp_path]
        )
        text_en = response.get('response', '').strip()
        text_en = re.sub(r'^[\s\*\?\"\'\.\,]+', '', text_en)
        text_en = text_en.replace('"', '').strip()

        has_real_words = bool(re.search(r'[A-Za-z]{2,}', text_en))
        if not text_en or not has_real_words:
            print(f"  [Moondream empty → fallback EasyOCR]")
            lines = split_into_lines(crop)
            all_texts = []
            for line_crop in lines:
                pad = 10
                line_padded = cv2.copyMakeBorder(line_crop, 0, 0, pad, 0,
                                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for variant in build_ocr_variants(line_padded):
                    result = reader.readtext(variant, detail=1, paragraph=False,
                                             allowlist=OCR_ALLOWLIST)
                    if result:
                        result = sorted(result, key=lambda r: r[0][0][0])
                        texts = [r[1] for r in result if len(r) >= 3]
                        all_texts.append(" ".join(texts))
                        break
            text_en = " ".join(all_texts)

        print(f"  [VLM OCR]: {text_en}")
        return text_en, 0.95

    except Exception as e:
        print(f"VLM Error: {e}")
        return "", 0


def translate_nllb(text_en: str) -> str:
    clean_en = fuzzy_fix_ocr(text_en)

    # Replace lore terms before NLLB translation to preserve intended names/phrases.
    for category in LORE_DATA.values():
        if isinstance(category, dict):
            for en, th in category.items():
                if len(en) >= 4:
                    clean_en = clean_en.replace(en, th)

    inputs = tokenizer(clean_en, return_tensors="pt").to(device)

    with torch.inference_mode():
        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("tha_Thai"),
            max_length=128,
            no_repeat_ngram_size=3,
            do_sample=False,
            num_beams=2,  
            early_stopping=True,
        )

    text_th = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    for category in LORE_DATA.values():
        if isinstance(category, dict):
            for en, th in category.items():
                text_th = text_th.replace(en, th)
            
    # ปรับ Lore ไทยให้เข้ากับบริบทเกม
    lore_fix = { 
        "เพื่อน": "คู่หู", 
        "พาร์ทเนอร์": "คู่หู", 
        "คู่แข่ง": "คู่หู", 
        "เทมมี่": "เทม",
        "เทพ": "พระเจ้า",
        "เทพเจ้า": "พระเจ้า"
    }
    for wrong, right in lore_fix.items(): 
        text_th = text_th.replace(wrong, right)
        
    return text_th


def get_image_hash(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


def _popup_worker():
    root = tk.Tk()
    root.title("Deltarune Translator")
    root.attributes('-topmost', True)
    root.configure(bg='black')

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"500x150+{sw-520}+{sh-200}")

    en_label = tk.Label(
        root,
        text="EN:",
        fg='white',
        bg='black',
        font=('Courier', 10),
        wraplength=480,
        justify='left'
    )
    en_label.pack(pady=5, padx=10, anchor='w')

    th_label = tk.Label(
        root,
        text="TH:",
        fg='yellow',
        bg='black',
        font=('Tahoma', 12, 'bold'),
        wraplength=480,
        justify='left'
    )
    th_label.pack(pady=5, padx=10, anchor='w')

    def poll_updates():
        latest = None
        while True:
            try:
                latest = popup_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            en_text, th_text = latest
            en_label.config(text=f"EN: {en_text}")
            th_label.config(text=f"TH: {th_text}")
            root.deiconify()
            root.lift()

        root.after(100, poll_updates)

    # กดปิดหน้าต่างแล้วซ่อนแทน เพื่อเปิดซ้ำได้ตอนกด F9
    root.protocol("WM_DELETE_WINDOW", root.withdraw)
    poll_updates()
    root.mainloop()


def show_popup(en_text: str, th_text: str):
    """แสดง popup แปลภาษาแบบค้างไว้จนกว่าจะมีการอัปเดตครั้งถัดไป"""
    global popup_thread_started

    if not popup_thread_started:
        threading.Thread(target=_popup_worker, daemon=True).start()
        popup_thread_started = True

    popup_queue.put((en_text, th_text))


def process_screen_image(img: np.ndarray, use_dedup: bool = True):
    global last_hash

    if use_dedup:
        current_hash = get_image_hash(img)
        if current_hash == last_hash:
            print("[Skip] ภาพเหมือนเดิม ข้ามการแปล")
            return

        last_hash = current_hash

    results = yolo_model.predict(source=img, conf=0.5, verbose=False)
    found = False

    for r in results:
        sorted_boxes = sorted(r.boxes, key=lambda b: b.xyxy[0][1].item())

        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            text_en, conf = ocr_best_text(crop, box_index=i)

            if text_en.strip() and conf >= 0.35:
                found = True
                text_th = translate_nllb(text_en)
                en_fix = fuzzy_fix_ocr(text_en)

                print(f"EN: {en_fix}")
                print(f"TH: {text_th}")
                print("-" * 40)

                show_popup(en_fix, text_th)

    if not found:
        print("ไม่พบกล่องข้อความ")


def process_static_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Image not found: {image_path}")
        return

    print(f"[Input] {image_path}")
    process_screen_image(img, use_dedup=False)


def start_realtime_translator():
    print("=========================================")
    print("Deltarune Real-time Translator Ready!")
    print("Press 'F9' to translate text on screen")
    print("Hold 'ESC' to exit")
    print("=========================================")

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            if keyboard.is_pressed('f9'):
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                process_screen_image(img, use_dedup=True)

                time.sleep(1)
                print("Waiting for next command... (press F9 to translate)")

            elif keyboard.is_pressed('esc'):
                print("Shutting down translator...")
                break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_static_image(sys.argv[1])
    else:
        start_realtime_translator()