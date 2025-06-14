# doorbell.py
import os
import pickle
import cv2
import torch
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys, pathlib, types

# 0) CONFIGURAÇÃO
DAT_FILE     = "known_faces.dat"
WEIGHTS_PATH = "face_recognition_best.pt"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
# VIDEO_SRC    = 2  # seu índice de webcam (0,1,2…) ou caminho de vídeo
VIDEO_SRC    = 'enzo_video.mp4'  # seu índice de webcam (0,1,2…) ou caminho de vídeo

# hack para yolov5 hubconf
fake = types.ModuleType("pathlib._local")
fake.Path      = pathlib.Path
fake.PosixPath = pathlib.PosixPath
sys.modules["pathlib._local"] = fake

# 1) carrega DB
known_embs, known_meta = pickle.load(open(DAT_FILE, "rb"))
print(f"[+] {len(known_embs)} embeddings carregados de {DAT_FILE}")

# 2) carrega YOLOv5
print("🔁 carregando YOLOv5…")
yolo = torch.hub.load(
    "./yolov5", "custom",
    path=WEIGHTS_PATH,
    source="local",
    force_reload=True
).to(DEVICE)
yolo.conf = 0.5

# 3) carrega Facenet
from facenet_pytorch import InceptionResnetV1
print("🔁 carregando InceptionResnetV1…")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# 4) extrai embedding e normaliza
def extract_embedding(face_bgr):
    face = cv2.resize(face_bgr, (160,160), interpolation=cv2.INTER_LINEAR)
    rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    t    = torch.tensor(rgb, device=DEVICE).permute(2,0,1).unsqueeze(0).float()
    t    = (t/255.0 - 0.5) / 0.5
    with torch.no_grad():
        emb = resnet(t).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

# 5) compara com o banco
def lookup(emb, thr=1.5):
    dists = np.linalg.norm(np.stack(known_embs) - emb, axis=1)
    idx   = np.argmin(dists)
    best  = dists[idx]
    print(f"[DEBUG] melhor distância para {known_meta[idx]['name']}: {best:.3f}")
    if best < thr:
        m = known_meta[idx]
        now = datetime.now()
        if now - m["last_seen"] > timedelta(minutes=2):
            m["seen_count"] += 1
        m["last_seen"] = now
        return m
    return None

# 6) abre vídeo
cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    raise RuntimeError("Não consegui abrir a fonte de vídeo.")

while True:
    ret, frame = cap.read()
    if not ret: break

    H, W = frame.shape[:2]

    # 6.1) detecta todas as faces
    dets = yolo(frame).xyxy[0].cpu().numpy()
    dets = dets[dets[:,4] >= 0.5]  # conf ≥ 0.5

    # 6.2) filtra boxes ≥20px e ≤90% da tela
    good = []
    for x1,y1,x2,y2,conf,cls in dets:
        bw, bh = x2-x1, y2-y1
        if bw<20 or bh<20 or bw>0.9*W or bh>0.9*H:
            continue
        good.append((x1,y1,x2,y2,conf))
    if not good:
        cv2.putText(frame, "No face", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Campainha", frame)
        if cv2.waitKey(1)&0xFF==ord("q"): break
        continue

    # 6.3) pega box de maior confiança
    x1,y1,x2,y2,conf = max(good, key=lambda b: b[4])

    # 6.4) recorte quadrado centrado + 10% de margem
    x1i, y1i = max(0,int(x1)), max(0,int(y1))
    x2i, y2i = min(W,int(x2)),   min(H,int(y2))
    bw, bh    = x2i-x1i, y2i-y1i
    size      = max(bw,bh)
    cx, cy    = x1i + bw//2, y1i + bh//2
    mrg       = int(0.1*size)
    size     += mrg
    x1s = max(0, cx-size//2)
    y1s = max(0, cy-size//2)
    x2s = min(W, cx+size//2)
    y2s = min(H, cy+size//2)
    face = frame[y1s:y2s, x1s:x2s]

    # 6.5) lookup
    emb = extract_embedding(face)
    m   = lookup(emb)

    # 6.6) desenha
    if m:
        label, col = f"{m['name']}, Visitas realizadas: ({m['seen_count']})", (0,255,0)
    else:
        label, col = "Unknown", (0,0,255)

    cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), col, 2)
    cv2.putText(frame, label, (x1i, y1i-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    cv2.imshow("Campainha", frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
