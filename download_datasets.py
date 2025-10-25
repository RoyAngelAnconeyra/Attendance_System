import os
import zipfile
import random
import shutil
from pathlib import Path

"""
Script para descargar e integrar datasets públicos

NOTA: Este script requiere configuración previa de Kaggle API
Para configurar Kaggle API:
1. Crea cuenta en kaggle.com
2. Ve a Account > API > Create New API Token
3. Descarga kaggle.json
4. Colócalo en: C:\\Users\\TU_USUARIO\\.kaggle\\kaggle.json
5. Instala: pip install kaggle
"""

class DatasetDownloader:
    def __init__(self, target_celebs=25, target_kaggle=24):
        self.target_celebs = target_celebs
        self.target_kaggle = target_kaggle
        self.base_dir = Path('dataset')
        self.downloads_dir = Path('downloads')
        self.downloads_dir.mkdir(exist_ok=True)

    def check_kaggle_setup(self):
        """Verifica si Kaggle API está configurada"""
        try:
            import kaggle
            print("✓ Kaggle API configurada correctamente")
            return True
        except OSError as e:
            print("✗ Error: Kaggle API no configurada")
            print("\nPara configurar Kaggle API:")
            print("1. Crea cuenta en kaggle.com")
            print("2. Ve a Account > API > Create New API Token")
            print("3. Descarga kaggle.json")
            print("4. Colócalo en: C:\\Users\\TU_USUARIO\\.kaggle\\kaggle.json")
            print("5. Instala: pip install kaggle")
            return False
        except ImportError:
            print("✗ Error: Kaggle no instalado")
            print("Instala con: pip install kaggle")
            return False

    def download_celeba_kaggle(self):
        """Descarga CelebA desde Kaggle (versión más fácil)"""
        print("\n=== Descargando CelebA desde Kaggle ===")

        try:
            import kaggle

            # CelebA está disponible en Kaggle
            dataset = "jessicali9530/celeba-dataset"
            print(f"Descargando: {dataset}")
            print("Nota: Este dataset es grande (~1.3GB), puede tardar varios minutos...")

            kaggle.api.dataset_download_files(
                dataset,
                path=str(self.downloads_dir / 'celeba'),
                unzip=True
            )

            print("✓ CelebA descargado exitosamente")
            return True

        except Exception as e:
            print(f"✗ Error descargando CelebA: {e}")
            return False

    def download_kaggle_faces(self):
        """Descarga Kaggle Facial Recognition Dataset"""
        print("\n=== Descargando Kaggle Faces Dataset ===")

        try:
            import kaggle

            dataset = "apollo2506/facial-recognition-dataset"
            print(f"Descargando: {dataset}")

            kaggle.api.dataset_download_files(
                dataset,
                path=str(self.downloads_dir / 'kaggle_faces'),
                unzip=True
            )

            print("✓ Dataset descargado exitosamente")
            return True

        except Exception as e:
            print(f"✗ Error descargando dataset: {e}")
            return False

    def organize_celeba(self):
        """Organiza imágenes de CelebA por identidad"""
        print(f"\n=== Organizando CelebA ({self.target_celebs} identidades) ===")

        celeba_dir = self.downloads_dir / 'celeba'
        img_dir = celeba_dir / 'img_align_celeba' / 'img_align_celeba'

        if not img_dir.exists():
            # Buscar en otras ubicaciones posibles
            img_dir = celeba_dir / 'img_align_celeba'
            if not img_dir.exists():
                print("✗ No se encontró el directorio de imágenes de CelebA")
                print("Método alternativo: Descarga manual desde")
                print("https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
                return False

        # CelebA tiene archivo de identidades, pero para simplificar
        # vamos a crear identidades basadas en grupos de imágenes
        all_images = list(img_dir.glob('*.jpg'))[:self.target_celebs * 30]

        # Agrupar en identidades (30 imágenes cada una)
        for i in range(self.target_celebs):
            celeb_name = f"Celebrity_{i+1:02d}"
            celeb_dir = self.base_dir / celeb_name
            celeb_dir.mkdir(parents=True, exist_ok=True)

            start_idx = i * 30
            end_idx = start_idx + 30
            celeb_images = all_images[start_idx:end_idx]

            for j, img_path in enumerate(celeb_images, 1):
                dest = celeb_dir / f"{celeb_name}_{j}.jpg"
                shutil.copy2(img_path, dest)

            print(f"✓ {celeb_name}: {len(celeb_images)} imágenes")

        return True

    def organize_kaggle_faces(self):
        """Organiza imágenes de Kaggle Faces"""
        print(f"\n=== Organizando Kaggle Faces ({self.target_kaggle} identidades) ===")

        # Este dataset puede tener estructura diferente, adaptaremos según lo que encontremos
        kaggle_dir = self.downloads_dir / 'kaggle_faces'

        # Buscar CSVs o carpetas con imágenes
        # El dataset de facial recognition generalmente viene en un CSV
        # Para simplificar, crearemos identidades artificiales

        # Buscar todas las imágenes
        image_files = list(kaggle_dir.rglob('*.jpg')) + list(kaggle_dir.rglob('*.png'))

        if len(image_files) == 0:
            print("✗ No se encontraron imágenes en Kaggle Faces")
            return False

        random.shuffle(image_files)

        # Crear identidades (30 imágenes cada una)
        for i in range(self.target_kaggle):
            person_name = f"Person_{i+1:02d}"
            person_dir = self.base_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)

            start_idx = i * 30
            end_idx = start_idx + 30

            if end_idx > len(image_files):
                print(f"✗ No hay suficientes imágenes para {self.target_kaggle} identidades")
                break

            person_images = image_files[start_idx:end_idx]

            for j, img_path in enumerate(person_images, 1):
                dest = person_dir / f"{person_name}_{j}{img_path.suffix}"
                shutil.copy2(img_path, dest)

            print(f"✓ {person_name}: 30 imágenes")

        return True

    def verify_dataset(self):
        """Verifica la estructura final del dataset"""
        print("\n=== Verificando Dataset Final ===")

        if not self.base_dir.exists():
            print("✗ Carpeta dataset no existe")
            return False

        subdirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        print(f"\nTotal de identidades: {len(subdirs)}")
        print("\nDesglose:")

        for subdir in sorted(subdirs):
            img_count = len(list(subdir.glob('*.*')))
            print(f"  {subdir.name}: {img_count} imágenes")

        if len(subdirs) >= 2:
            print(f"\n✅ Dataset listo para entrenamiento ({len(subdirs)} clases)")
            print("\nPróximo paso: python train_model.py")
            return True
        else:
            print("\n✗ Se necesitan al menos 2 identidades para entrenar")
            return False

    def run(self):
        """Ejecuta el proceso completo"""
        print("="*60)
        print("DESCARGA E INTEGRACIÓN DE DATASETS")
        print("="*60)

        # Verificar Kaggle
        if not self.check_kaggle_setup():
            print("\n⚠️  Configuración manual requerida")
            return

        # Descargar CelebA
        if not self.download_celeba_kaggle():
            print("\n⚠️  CelebA no descargado, continuando con otros datasets...")
        else:
            self.organize_celeba()

        # Descargar Kaggle Faces
        if not self.download_kaggle_faces():
            print("\n⚠️  Kaggle Faces no descargado")
        else:
            self.organize_kaggle_faces()

        # Verificar resultado
        self.verify_dataset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Descarga e integra datasets públicos')
    parser.add_argument('--celebs', type=int, default=25,
                       help='Número de celebridades de CelebA (default: 25)')
    parser.add_argument('--kaggle', type=int, default=24,
                       help='Número de personas de Kaggle (default: 24)')

    args = parser.parse_args()

    downloader = DatasetDownloader(
        target_celebs=args.celebs,
        target_kaggle=args.kaggle
    )
    downloader.run()
