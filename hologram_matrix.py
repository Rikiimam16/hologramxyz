# Petunjuk Instalasi:
# Buka terminal dan jalankan perintah berikut:
# pip install opencv-python mediapipe numpy pillow

import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageGrab

class HologramVFX:
    def __init__(self):
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # EMA Smoothing — peredam getaran koordinat
        self.prev_pts = None
        self.smooth_alpha = 0.25
        
        # State untuk Matrix Digital Rain (posisi jatuh tiap kolom binary code)
        self.rain_drops = [np.random.randint(-40, 0) for _ in range(100)]
        
        # State untuk perekaman layar laptop
        self.recording = False
        self.video_writer = None
        
        # State Timer 4 Detik (Non-blocking)
        self.screenshot_time_requested = None
        self.recording_time_requested = None
        self.last_countdown_print = -1

    # ──────────────────────────────────────────
    # GESTURE DETECTION
    # ──────────────────────────────────────────
    def is_fist(self, lm):
        """Mendeteksi kepalan tangan."""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        closed = sum(1 for t, p in zip(tips, pips) if lm.landmark[t].y > lm.landmark[p].y)
        return closed >= 4

    def is_open_palm(self, lm):
        """Mendeteksi telapak tangan terbuka lebar."""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        opened = sum(1 for t, p in zip(tips, pips) if lm.landmark[t].y < lm.landmark[p].y)
        if lm.landmark[4].y < lm.landmark[3].y:
            opened += 1
        return opened >= 5

    # ──────────────────────────────────────────
    # EMA SMOOTHING
    # ──────────────────────────────────────────
    def smooth_coordinates(self, curr_pts):
        if self.prev_pts is None:
            self.prev_pts = np.array(curr_pts, dtype=np.float32)
            return curr_pts
        curr_arr = np.array(curr_pts, dtype=np.float32)
        smoothed = self.smooth_alpha * curr_arr + (1.0 - self.smooth_alpha) * self.prev_pts
        self.prev_pts = smoothed
        return [(int(p[0]), int(p[1])) for p in smoothed]

    # ──────────────────────────────────────────
    # NEW DESIGN: RETRO GREEN MATRIX DIGITAL RAIN
    # ──────────────────────────────────────────
    def apply_matrix_rain_filter(self, roi):
        """Filter hologram bergaya Matrix / Binary Rain berwarna hijau retro."""
        rh, rw = roi.shape[:2]
        
        # 1. Konversi ke Grayscale & Efek Pikselasi Digital (Pixelate)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Tingkatkan kontras
        
        pixel_w, pixel_h = max(8, rw // 6), max(8, rh // 6)
        small = cv2.resize(gray, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        
        # 2. Pewarnaan Hijau Monokrom Matrix
        green_tint = np.zeros(roi.shape, dtype=np.uint8)
        green_tint[:, :, 1] = pixelated
        
        result = cv2.addWeighted(roi, 0.08, green_tint, 0.92, 0)
        
        # 3. Render Aliran Kode Binary (Falling Binary Rain / Green Stream)
        col_width = 18
        cols = min(100, rw // col_width)
        
        for c in range(cols):
            col_x = c * col_width + 4
            
            # Gerakkan posisi hujan binary ke bawah secara acak
            self.rain_drops[c] += np.random.randint(6, 12)
            if self.rain_drops[c] > rh:
                self.rain_drops[c] = np.random.randint(-40, 0)
                
            y = self.rain_drops[c]
            
            # Gambar 6 karakter ekor binary (0 & 1) memudar ke atas
            for i in range(6):
                char_y = y - i * 14
                if 0 < char_y < rh:
                    char = "1" if np.random.rand() > 0.5 else "0"
                    
                    if i == 0:
                        color = (255, 255, 255) # Putih (BGR)
                    elif i < 3:
                        color = (180, 255, 180) # Hijau muda
                    else:
                        color = (0, 180, 0) # Hijau gelap
                        
                    cv2.putText(result, char, (col_x, char_y), 
                                cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1, cv2.LINE_AA)
                    
        # 4. HUD Teks Status Pemindaian Retro
        cv2.putText(result, "STREAMS: ON", (12, 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(result, "DEC.CODE: SECURE", (12, 35), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        return result

    # ──────────────────────────────────────────
    # RENDER: Hologram area tangan
    # ──────────────────────────────────────────
    def render_hand_hologram(self, frame, tl, tr, br, bl):
        """Render hologram hijau matrix di antara 4 titik jari."""
        h, w, _ = frame.shape
        x_coords = [tl[0], tr[0], br[0], bl[0]]
        y_coords = [tl[1], tr[1], br[1], bl[1]]
        min_x = max(0, min(x_coords) - 5)
        max_x = min(w, max(x_coords) + 5)
        min_y = max(0, min(y_coords) - 5)
        max_y = min(h, max(y_coords) + 5)
        if max_x <= min_x or max_y <= min_y:
            return frame

        roi = frame[min_y:max_y, min_x:max_x].copy()
        holo = self.apply_matrix_rain_filter(roi)

        # Masking polygon
        mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        pts_local = np.array(
            [[pt[0]-min_x, pt[1]-min_y] for pt in [tl, tr, br, bl]], np.int32)
        cv2.fillPoly(mask, [pts_local], 255, cv2.LINE_AA)

        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(holo, holo, mask=mask)
        frame[min_y:max_y, min_x:max_x] = cv2.add(bg, fg)

        # Bingkai Neon Hijau Matrix Berlapis
        pts = np.array([tl, tr, br, bl], np.int32)
        cv2.polylines(frame, [pts], True, (0, 100, 0),     12, cv2.LINE_AA)  # Glow hijau tua luar
        cv2.polylines(frame, [pts], True, (0, 255, 0),     4, cv2.LINE_AA)  # Core hijau menyala
        cv2.polylines(frame, [pts], True, (220, 255, 220), 1, cv2.LINE_AA)  # Inti putih-hijau

        # Corner brackets HUD
        self._draw_corner_brackets(frame, [tl, tr, br, bl], (0, 255, 0))
        return frame

    # ──────────────────────────────────────────
    # RENDER: Hologram full screen
    # ──────────────────────────────────────────
    def render_fullscreen_hologram(self, frame):
        """Terapkan efek hologram matrix ke seluruh layar."""
        frame[:] = self.apply_matrix_rain_filter(frame)

        # Border neon di tepi layar
        h, w = frame.shape[:2]
        pad = 12
        cv2.rectangle(frame, (pad, pad), (w-pad, h-pad), (0, 100, 0),     12, cv2.LINE_AA)
        cv2.rectangle(frame, (pad, pad), (w-pad, h-pad), (0, 255, 0),     4, cv2.LINE_AA)
        cv2.rectangle(frame, (pad, pad), (w-pad, h-pad), (220, 255, 220), 1, cv2.LINE_AA)

        # Corner brackets di empat sudut layar
        corners = [(pad, pad), (w-pad, pad), (w-pad, h-pad), (pad, h-pad)]
        self._draw_corner_brackets(frame, corners, (0, 255, 0), size=40, thickness=3)
        return frame

    # ──────────────────────────────────────────
    # UTILITY: Corner brackets HUD sci-fi
    # ──────────────────────────────────────────
    def _draw_corner_brackets(self, frame, pts, color, size=22, thickness=2):
        tl, tr, br, bl = pts
        for pt, dx, dy in [(tl, 1, 1), (tr, -1, 1), (br, -1, -1), (bl, 1, -1)]:
            px, py = pt
            cv2.line(frame, (px, py), (px + dx * size, py), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * size), color, thickness, cv2.LINE_AA)

    # ──────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Pengaturan ukuran jendela (window) kamera agar bisa dikecilkan
        window_name = "Retro Green Matrix Hologram"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 360) # Mengubah ukuran jendela tampilan jadi lebih kecil (640x360)

        print("====== RETRO GREEN MATRIX HOLOGRAM ======")
        print("Gestur:")
        print("  • Bingkai (Jempol+Telunjuk 2 tangan) → Hologram matrix antar tangan")
        print("  • Kedua telapak terbuka lebar        → Hologram matrix full layar")
        print("  • Kepalan (satu/dua tangan)          → Hologram hilang")
        print("Tombol Keyboard (Timer 4 Detik):")
        print("  • Tekan 'S' untuk Mengambil Foto Layar Laptop (Tunggu 4 detik)")
        print("  • Tekan 'R' untuk Mulai Merekam Layar Laptop (Tunggu 4 detik)")
        print("  • Tekan 'Q' untuk keluar.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            current_time = time.time()

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            hands_data     = []
            fist_count     = 0
            open_palm_count = 0

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:

                    # Gambar Skeleton Tangan (Hijau neon)
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(
                            color=(0, 255, 80), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(
                            color=(0, 180, 50), thickness=2)
                    )

                    # Deteksi gestur
                    if self.is_fist(hand_lm):
                        fist_count += 1
                    if self.is_open_palm(hand_lm):
                        open_palm_count += 1

                    # Kumpulkan data posisi tangan
                    wrist_x   = hand_lm.landmark[0].x
                    thumb_pt  = (int(hand_lm.landmark[4].x * w),
                                 int(hand_lm.landmark[4].y * h))
                    index_pt  = (int(hand_lm.landmark[8].x * w),
                                 int(hand_lm.landmark[8].y * h))
                    hands_data.append({'wrist_x': wrist_x, 'thumb': thumb_pt, 'index': index_pt})

            # ── LOGIKA GESTUR ──
            if fist_count >= 1:
                # Kepalan → Hologram hilang, skeleton tetap muncul
                self.prev_pts = None

            elif open_palm_count == 2:
                # Kedua telapak terbuka → Hologram full layar
                frame = self.render_fullscreen_hologram(frame)
                self.prev_pts = None

            elif len(hands_data) == 2:
                # Dua tangan + bukan kepalan → Hologram antar tangan
                hands_data.sort(key=lambda x: x['wrist_x'])
                left  = hands_data[0]
                right = hands_data[1]

                raw_pts = [
                    left['index'],   # tl
                    right['index'],  # tr
                    right['thumb'],  # br
                    left['thumb'],   # bl
                ]
                tl, tr, br, bl = self.smooth_coordinates(raw_pts)
                frame = self.render_hand_hologram(frame, tl, tr, br, bl)

            else:
                self.prev_pts = None

            cv2.imshow("Retro Green Matrix Hologram", frame)
            
            # ── PROSES TIMER & CAPTURE SCREEN ──
            # 1. Hitung Mundur Screenshot Laptop
            if self.screenshot_time_requested is not None:
                elapsed = current_time - self.screenshot_time_requested
                remaining = 4 - int(elapsed)
                if remaining > 0:
                    if int(elapsed) != self.last_countdown_print:
                        print(f"[FOTO] Mengambil foto dalam {remaining}...")
                        self.last_countdown_print = int(elapsed)
                else:
                    try:
                        screenshot = ImageGrab.grab()
                        filename = f"screenshot_laptop_{int(current_time)}.png"
                        screenshot.save(filename)
                        print(f"[FOTO] Tangkapan layar laptop disimpan: {filename}")
                    except Exception as e:
                        print("Gagal mengambil screenshot laptop:", e)
                    self.screenshot_time_requested = None
                    self.last_countdown_print = -1

            # 2. Hitung Mundur Mulai Merekam Layar
            if self.recording_time_requested is not None:
                elapsed = current_time - self.recording_time_requested
                remaining = 4 - int(elapsed)
                if remaining > 0:
                    if int(elapsed) != self.last_countdown_print:
                        print(f"[REKAM] Mulai merekam dalam {remaining}...")
                        self.last_countdown_print = int(elapsed)
                else:
                    try:
                        self.recording = True
                        screen_size = ImageGrab.grab().size
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        filename = f"rekaman_layar_{int(current_time)}.avi"
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 12.0, screen_size)
                        print(f"[REKAM] Mulai merekam layar laptop. Disimpan ke: {filename}")
                    except Exception as e:
                        print("Gagal memulai perekaman layar laptop:", e)
                        self.recording = False
                    self.recording_time_requested = None
                    self.last_countdown_print = -1

            # 3. Tulis frame rekaman saat sedang aktif merekam
            if self.recording and self.video_writer is not None:
                try:
                    screen = ImageGrab.grab()
                    screen_np = np.array(screen)
                    screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(screen_bgr)
                except Exception as e:
                    print("Error ketika merekam frame layar:", e)

            # Penanganan Input Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                if self.screenshot_time_requested is None:
                    self.screenshot_time_requested = current_time
                    self.last_countdown_print = -1
                    print("[FOTO] Timer 4 detik dimulai...")
            elif key == ord('r') or key == ord('R'):
                if not self.recording:
                    if self.recording_time_requested is None:
                        self.recording_time_requested = current_time
                        self.last_countdown_print = -1
                        print("[REKAM] Timer perekaman 4 detik dimulai...")
                else:
                    # Matikan perekaman langsung tanpa delay
                    self.recording = False
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None
                    print("[REKAM] Perekaman layar laptop berhenti.")

        if self.video_writer is not None:
            self.video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HologramVFX()
    app.run()
