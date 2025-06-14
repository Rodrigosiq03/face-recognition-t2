import os
import re
from PIL import Image

DATA_DIR = os.path.join("data", "WIDER")

def delete_invalid_image(img_path):
    """Deleta a imagem se não for válida"""
    try:
        os.remove(img_path)
        print(f"[Deletado] Imagem inválida removida: {img_path}")
    except Exception as e:
        print(f"[Erro] Não foi possível deletar {img_path}: {e}")

def main():
    
    for split in ["train", "val"]:
        label_dir = os.path.join(DATA_DIR, f"labels/{split}")
        os.makedirs(label_dir, exist_ok=True)

        
        annot_file = os.path.join(DATA_DIR, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
        
        
        if not os.path.exists(annot_file):
            print(f"[Erro] Arquivo de anotações não encontrado: {annot_file}")
            continue

        with open(annot_file) as f:
            lines = f.readlines()

        print(f"Processando {annot_file} ...")

        i = 0
        while i < len(lines):
            img_rel_path = lines[i].strip()  
            
            img_path = os.path.join(DATA_DIR, f"WIDER_{split}", "images", img_rel_path)

            if not os.path.exists(img_path):
                print(f"[Aviso] Imagem não encontrada: {img_path} - Pulando esta imagem.")
                i += 2  
                continue

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[Erro] Não foi possível abrir a imagem {img_rel_path}: {e} - Deletando imagem.")
                
                delete_invalid_image(img_path)
                i += 2  
                continue
            
            try:
                n_faces = int(lines[i+1].strip())  
            except ValueError:
                print(f"[Aviso] Número de faces não encontrado ou linha malformada em {img_rel_path}. Pulando esta imagem.")
                i += 2  
                continue

            label_txt = os.path.join(DATA_DIR, f"labels/{split}", re.sub(r"\.(jpg|jpeg|png)$", ".txt", img_rel_path, flags=re.IGNORECASE))
            os.makedirs(os.path.dirname(label_txt), exist_ok=True)
            
            with open(label_txt, "w") as lt:
                for j in range(n_faces):
                    comps = list(map(int, lines[i+2+j].split()[:4]))  
                    convert_annotation_line = f"0 {(comps[0]+comps[2]/2)/img_w:.6f} {(comps[1]+comps[3]/2)/img_h:.6f} {(comps[2]/img_w):.6f} {(comps[3]/img_h):.6f}\n"
                    lt.write(convert_annotation_line)

            i += 2 + n_faces  

        print(f"[✓] Conversão concluída para {split}.")
    
    print("[✓] Processamento finalizado.")

if __name__ == "__main__":
    main()
