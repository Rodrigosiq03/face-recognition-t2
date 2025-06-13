import face_recognition
import cv2
import os
import pickle
from datetime import datetime
import numpy as np
from datetime import timedelta

# Caminho para o diretório da galeria (onde as imagens das pessoas estão armazenadas)
GALLERY_DIR = 'gallery'
KNOWN_FACES_FILE = 'known_faces.dat'

# Listas para armazenar as codificações das faces e os metadados
known_face_encodings = []
known_face_metadata = []

def save_known_faces():
    """Salva as faces registradas no arquivo .dat"""
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump([known_face_encodings, known_face_metadata], f)
    print(f"Faces salvas em {KNOWN_FACES_FILE}")

def load_known_faces():
    """Carrega as faces previamente registradas do arquivo .dat"""
    global known_face_encodings, known_face_metadata
    try:
        with open(KNOWN_FACES_FILE, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
        print(f"Faces carregadas de {KNOWN_FACES_FILE}")
    except FileNotFoundError:
        print(f"Arquivo {KNOWN_FACES_FILE} não encontrado. Começando com uma lista em branco.")

def register_face(image_path, name):
    """Registra uma nova face na galeria"""
    # Carregar a imagem e encontrar a face
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]

    # Adicionar a codificação da face à lista de codificações
    known_face_encodings.append(face_encoding)
    
    # Registrar o metadado da face
    known_face_metadata.append({
        "name": name,
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": cv2.imread(image_path)  # Adiciona a imagem da face
    })

    # Salvar as faces após o registro
    save_known_faces()

def process_gallery():
    """Processa todas as imagens da galeria e registra as faces"""
    for image_name in os.listdir(GALLERY_DIR):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(GALLERY_DIR, image_name)
            name = os.path.splitext(image_name)[0]  # Nome da pessoa (sem a extensão)
            print(f"Registrando {name} a partir de {image_path}")
            register_face(image_path, name)

def lookup_known_face(face_encoding):
    """Procura uma face conhecida e retorna o metadado se encontrada"""
    metadata = None
    if len(known_face_encodings) == 0:
        return metadata
    
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:  # Limiar de correspondência, ajustável
        metadata = known_face_metadata[best_match_index]
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=2):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1
    return metadata

if __name__ == "__main__":
    # Carregar faces previamente registradas
    load_known_faces()

    # Processar a galeria (se necessário)
    process_gallery()

    # Exemplo de teste de uma face (substitua com sua imagem de teste)
    test_image_path = 'digao_teste.jpg'
    test_image = face_recognition.load_image_file(test_image_path)
    test_face_encoding = face_recognition.face_encodings(test_image)[0]

    metadata = lookup_known_face(test_face_encoding)
    if metadata:
        print(f"Pessoa reconhecida: {metadata['name']}, Visitas: {metadata['seen_count']}")
    else:
        print("Pessoa desconhecida.")
    
    # save_known_faces()
