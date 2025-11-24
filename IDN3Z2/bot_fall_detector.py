import cv2
import numpy as np
import argparse
import time
import os
import imutils
from collections import deque
from telegram import Bot
from telegram.error import TelegramError

# ----------------- Налаштування детектора MobileNet-SSD -----------------

PROTO_PATH = "mobilenet_ssd_deploy.prototxt.txt"
MODEL_PATH = "mobilenet_ssd_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# ----------------- Конфігурація логіки виявлення падіння -----------------
DEFAULTS = {
    "min_confidence": 0.5,
    "fall_drop_px": 80,            # різкий спад центроїда Y в пікселях
    "aspect_ratio_thresh": 0.6,   # якщо h/w < thresh -> лежить (менше означає широкі/низькі bbox)
    "hold_time_seconds": 3.0,     # після падіння має пройти стільки секунд без руху щоб підтвердити
    "history_len": 32,            # число попередніх центрів для smoothing
    "debounce_seconds": 20.0,     # мінімальний інтервал між повідомленнями
}

# ----------------- Telegram (скласти свої значення) -----------------
TELEGRAM_TOKEN = "8522752517:AAG8PWG8isif6fBPtrcKs5QPTy0JQ7jEbq8"
CHAT_ID = "2062880220"

# ----------------- Допоміжні функції -----------------
def ensure_model_files():
    if not os.path.exists(PROTO_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Не знайдено модельні файли. Потрібні: {PROTO_PATH}, {MODEL_PATH}.\n"
            "Завантажте їх з репозиторію MobileNet-SSD і покладіть у ту саму папку."
        )

def load_net():
    ensure_model_files()
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    return net

def detect_people(net, frame, min_confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        idx = int(detections[0, 0, i, 1])
        if confidence > min_confidence and CLASSES[idx] == "person":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                "bbox": (startX, startY, endX, endY),
                "confidence": confidence
            })
    return results

# ----------------- Основна логіка -----------------
def main(args):
    global TELEGRAM_TOKEN, CHAT_ID
    TELEGRAM_TOKEN = args.telegram_token
    CHAT_ID = args.chat_id

    config = {
        "min_confidence": args.min_confidence or DEFAULTS["min_confidence"],
        "fall_drop_px": args.fall_drop_px or DEFAULTS["fall_drop_px"],
        "aspect_ratio_thresh": args.aspect_ratio_thresh or DEFAULTS["aspect_ratio_thresh"],
        "hold_time_seconds": args.hold_time_seconds or DEFAULTS["hold_time_seconds"],
        "history_len": DEFAULTS["history_len"],
        "debounce_seconds": DEFAULTS["debounce_seconds"],
    }

    # Ініціалізуємо телеграм-бота
    bot = Bot(token=TELEGRAM_TOKEN)

    # Завантажуємо нейронку
    print("[INFO] Завантаження мережі...")
    net = load_net()
    print("[INFO] Доступ до відеопотоку:", args.source)

    cap = cv2.VideoCapture(args.source if isinstance(args.source, int) or args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("[ERROR] Не можу відкрити джерело:", args.source)
        return

    last_alert_time = 0.0

    centroid_history = deque(maxlen=config["history_len"])
    bbox_history = deque(maxlen=config["history_len"])
    last_movement_time = time.time()
    potential_fall_time = None
    fall_confirmed = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Кінець відеопотоку або помилка")
            break
        frame_idx += 1
        frame = imutils.resize(frame, width=600)
        orig = frame.copy()
        now = time.time()

        people = detect_people(net, frame, min_confidence=config["min_confidence"])
        # Якщо людей немає — розчистимо історію
        if len(people) == 0:
            centroid_history.clear()
            bbox_history.clear()
            potential_fall_time = None
            fall_confirmed = False
            # відображення
            cv2.putText(frame, "No person detected", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # показ вікна (при локальному тестуванні)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # Обираємо найбільший bbox (найближча/найважливіша людина)
        people_sorted = sorted(people, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True)
        p = people_sorted[0]
        (sx, sy, ex, ey) = p["bbox"]
        w = ex - sx
        h = ey - sy
        cx = int(sx + w/2)
        cy = int(sy + h/2)
        centroid_history.append((cx, cy, now))
        bbox_history.append((sx, sy, ex, ey, now))

        # Намалюємо bbox
        cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"Conf: {p['confidence']:.2f}", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Аналіз падіння:
        fall_detected_now = False
        # 1) Перевіримо різкий спад центроїда: порівняємо поточний cy з середнім за попередні N кадрів
        if len(centroid_history) >= 6:
            # беремо середнє по попереднім кадрам (окрім останнього)
            xs, ys, ts = zip(*centroid_history)
            # середнє Y попередніх 4-5 точок (окрім останньої)
            prev_ys = list(ys)[:-1]
            if len(prev_ys) >= 3:
                avg_prev_y = sum(prev_ys[-4:]) / len(prev_ys[-4:])
                dy = cy - avg_prev_y
                if dy > config["fall_drop_px"]:
                    # різке падіння вниз
                    fall_detected_now = True
                    reason = f"rapid_drop dy={dy:.1f}"
                else:
                    reason = f"no_rapid_drop dy={dy:.1f}"
            else:
                reason = "not_enough_history"
        else:
            reason = "short_history"

        # 2) Перевіримо співвідношення h/w — якщо стало дуже маленьким (лежачий)
        aspect = h / float(w+1e-6)
        if aspect < config["aspect_ratio_thresh"]:
            fall_detected_now = True
            reason = (reason + f"; flat_aspect {aspect:.2f}")

        # Якщо виявлено потенційне падіння — починаємо таймер підтвердження
        if fall_detected_now and not potential_fall_time:
            potential_fall_time = now
            print(f"[INFO] Potential fall at frame {frame_idx}, reason: {reason}")

        # Якщо потенційне падіння в минулому — перевіряємо чи є рух після нього
        if potential_fall_time:
            # вирахуємо середню швидкість руху центроїда за останні кілька кадрів
            if len(centroid_history) >= 6:
                # швидкість = зміна положення / час
                (x_old, y_old, t_old) = centroid_history[0]
                (x_new, y_new, t_new) = centroid_history[-1]
                speed = np.hypot(x_new-x_old, y_new-y_old) / max(1e-6, (t_new - t_old))
                # якщо швидкість низька — значить людина лежить без руху
                if speed < 5.0:  # px/sec — поріг можна налаштувати
                    # перевірка часу утримання
                    if now - potential_fall_time >= config["hold_time_seconds"]:
                        # впевненіше в падінні — підтверджуємо
                        if not fall_confirmed and (now - last_alert_time) > config["debounce_seconds"]:
                            print(f"[ALERT] Fall confirmed at {time.ctime(now)}; speed={speed:.2f}")
                            # відправка повідомлення в Telegram
                            try:
                                # збережемо кадр
                                fname = f"alert_{int(now)}.jpg"
                                cv2.imwrite(fname, orig)
                                text = (f"⚠️ Підтверджено падіння!\nЧас: {time.ctime(now)}\n"
                                        f"Причина: {reason}\nAspect={aspect:.2f}, speed={speed:.2f}")
                                bot.send_message(chat_id=CHAT_ID, text=text)
                                with open(fname, "rb") as f:
                                    bot.send_photo(chat_id=CHAT_ID, photo=f)
                                last_alert_time = now
                                fall_confirmed = True
                            except TelegramError as e:
                                print("[ERROR] Telegram send error:", e)
                                # продовжимо роботу локально
                        else:
                            # або debounce не дозволяє сповіщення
                            pass
                else:
                    # якщо є рух — це можливо помилкове спрацювання (людина піднялася)
                    potential_fall_time = None
                    fall_confirmed = False
            # якщо ще дуже мало часу з моменту потенційного падіння — чекаємо
            if now - potential_fall_time > 10.0 and not fall_confirmed:
                # анульовуємо якщо не підтверджено довго
                potential_fall_time = None

        # Позначимо інформаційний текст
        info = f"AR={aspect:.2f} potential_fall={'yes' if potential_fall_time else 'no'}"
        cv2.putText(frame, info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

        # Показуємо відео (корисно при локальному тестуванні)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Парсинг аргументів -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-token", default=TELEGRAM_TOKEN, help="Telegram bot token")
    parser.add_argument("--chat-id", default=CHAT_ID, help="Telegram chat_id to send alerts")
    parser.add_argument("--source", default="0", help="Video source: device index or URL (default 0)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULTS["min_confidence"])
    parser.add_argument("--fall-drop-px", dest="fall_drop_px", type=int, default=DEFAULTS["fall_drop_px"])
    parser.add_argument("--aspect-ratio-thresh", dest="aspect_ratio_thresh", type=float, default=DEFAULTS["aspect_ratio_thresh"])
    parser.add_argument("--hold-time-seconds", dest="hold_time_seconds", type=float, default=DEFAULTS["hold_time_seconds"])
    args = parser.parse_args()
    # нормалізуємо source
    try:
        # якщо цифра -> int
        if args.source.isdigit():
            args.source = int(args.source)
    except Exception:
        pass
    main(args)

