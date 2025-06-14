# face_register.py
import os, pickle, cv2, torch
from datetime import datetime
from pathlib import Path
import numpy as np

# caminho para seu YOLO treinado
WEIGHTS = "face_recognition_best.pt"
GALLERY  = "gallery"
DAT_FILE = "known_faces.dat"
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

import sys, pathlib, types
# CRIA um módulo fake para que o hubconf do yolov5 encontre pathlib._local
fake = types.ModuleType("pathlib._local")
fake.Path      = pathlib.Path
fake.PosixPath = pathlib.PosixPath
sys.modules["pathlib._local"] = fake

# modelos
print("🔁 carregando YOLOv5…")
yolo = torch.hub.load('./yolov5', 'custom', path=WEIGHTS, source='local', force_reload=True).to(DEVICE)
yolo.conf = 0.5

from facenet_pytorch import InceptionResnetV1
print("🔁 carregando InceptionResnetV1…")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# memória
known_embs = []
known_meta = []

def load_db():
    global known_embs, known_meta
    if Path(DAT_FILE).exists():
        known_embs, known_meta = pickle.load(open(DAT_FILE,'rb'))
        print(f"[+] {len(known_embs)} faces carregadas do banco.")
    else:
        print("[!] começando banco vazio")

def save_db():
    pickle.dump((known_embs, known_meta), open(DAT_FILE,'wb'))
    print(f"[✓] salvo {len(known_embs)} embeddings em {DAT_FILE}")

def extract_embedding(face_bgr):
    # Face deve ter tamanho mínimo 1×1, mas Inception exige ≥160×160
    # Redimensiona sempre para 160×160
    face_resized = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_LINEAR)

    # Converte BGR → RGB
    rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Tensor [1,3,160,160], normaliza conforme Facenet
    img_t = torch.tensor(rgb, device=DEVICE).permute(2,0,1).unsqueeze(0).float()
    img_t = (img_t / 255.0 - 0.5) / 0.5

    with torch.no_grad():
        emb = resnet(img_t).cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)  # Normaliza o vetor de embedding

    return emb

def register(name, face_bgr):
    emb = extract_embedding(face_bgr)
    known_embs.append(emb)
    known_meta.append({
        "name": name,
        "first_seen": datetime.now(),
        "last_seen":  datetime.now(),
        "seen_count": 1,
        "face_image": face_bgr
    })
    print(f"[+] {name} registrado")

def process_gallery():
    load_db()
    for fn in os.listdir(GALLERY):
        if not fn.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        name = Path(fn).stem
        img = cv2.imread(os.path.join(GALLERY, fn))
        if img is None:
            print(f"[!] falha ao ler {fn}")
            continue

        # detecta com YOLO
        dets = yolo(img).xyxy[0].cpu().numpy()
        if dets.size == 0:
            print(f"[!] sem face em {fn}")
            continue

        # pega bbox de maior confiança
        x1, y1, x2, y2, conf, _ = max(dets, key=lambda x: x[4])

        # ajusta e recorta a ROI para evitar out-of-bounds
        x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
        x1i, x2i = sorted((x1i, x2i))
        y1i, y2i = sorted((y1i, y2i))
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i = min(img.shape[1], x2i)
        y2i = min(img.shape[0], y2i)

        face = img[y1i:y2i, x1i:x2i]
        if face is None or face.size == 0:
            print(f"[!] ROI vazia para {fn}: bbox ({x1i},{y1i},{x2i},{y2i}) — pulando")
            continue

        register(name, face)

    save_db()

if __name__ == "__main__":
    process_gallery()
