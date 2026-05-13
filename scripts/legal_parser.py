import os
import re
import fitz  # PyMuPDF
import logging
from pathlib import Path

# ==========================================
# Configuración de Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class LegalParser:
    """
    Clase especializada en la extracción, limpieza y estructuración 
    de documentos legales en formato PDF a Markdown.
    """
    
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text):
        """
        Aplica limpieza profunda al texto usando Regex.
        Elimina encabezados, pies de página, números y avisos de reforma.
        """
        # 1. Eliminar avisos de reformas (ej. 'DOF 20-04-2026' o 'Reforma DOF ...')
        text = re.sub(r'(?i)(?:reforma|publicado en el|última reforma|fe de erratas).*?DOF\s+\d{2}-\d{2}-\d{4}', '', text)
        
        # 2. Eliminar encabezados y pies de página institucionales comunes
        # Se agregan variaciones detectadas en el documento de prueba
        patterns_to_remove = [
            r'CÁMARA DE DIPUTADOS DEL H\. CONGRESO DE LA UNIÓN',
            r'CONSTITUCIÓN POLÍTICA DE LOS ESTADOS UNIDOS MEXICANOS',
            r'Secretaría General',
            r'Secretaría de Servicios Parlamentarios',
            r'www\.diputados\.gob\.mx',
            r'Texto Vigente',
            r'Nueva Ley .*?',
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
            
        # 3. Eliminar números de página
        # Formatos: "1 de 100", "Página 5", o números aislados con posibles espacios
        text = re.sub(r'^\s*\d+\s+de\s+\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Página\s+\d+\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # 4. Normalización de saltos de línea (Unir oraciones cortadas)
        # Identifica líneas que terminan en letra/coma y la siguiente empieza con minúscula
        text = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑ,;])\n(?=[a-zñáéíóú])', r'\1 ', text)
        
        # 5. Eliminar múltiples saltos de línea y espacios excesivos
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def structure_markdown(self, text):
        """
        Convierte la estructura legal (Títulos, Capítulos, Artículos) a jerarquía Markdown.
        """
        # TÍTULO -> # TÍTULO (Soporta Romanos, Números y Palabras como Primero, Segundo...)
        text = re.sub(r'^(TÍTULO\s+(?:[IVXLCDM\d]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO).*?)$', 
                     r'# \1', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # CAPÍTULO -> ## CAPÍTULO
        text = re.sub(r'^(CAPÍTULO\s+(?:[IVXLCDM\d]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO).*?)$', 
                     r'## \1', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # SECCIÓN -> ### SECCIÓN
        text = re.sub(r'^(SECCIÓN\s+(?:[IVXLCDM\d]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO).*?)$', 
                     r'### \1', text, flags=re.MULTILINE | re.IGNORECASE)

        # Artículo -> ### Artículo 
        # Asegura capturar variantes como "Artículo 1o." o "Artículo 1."
        text = re.sub(r'^(Artículo\s+\d+\s*[o°\.]?.*?)$', r'### \1', text, flags=re.MULTILINE)
        
        return text

    def process_file(self, pdf_path):
        """
        Flujo principal: Extracción -> Limpieza -> Estructuración -> Guardado.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logging.error(f"Archivo no encontrado: {pdf_path}")
                return

            # Extracción con PyMuPDF
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            logging.info(f"Procesando '{pdf_path.name}' | Páginas: {num_pages}")
            
            raw_text = ""
            for page in doc:
                # Se utiliza el modo "text" para mantener el flujo de palabras
                raw_text += page.get_text("text") + "\n"
            
            doc.close()

            # Procesamiento de texto
            cleaned = self.clean_text(raw_text)
            markdown_content = self.structure_markdown(cleaned)
            
            # Guardado
            output_filename = self.output_dir / f"{pdf_path.stem}.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            logging.info(f"Exito: Guardado en '{output_filename}'")
            print(f"\n--- Resumen de procesamiento ---")
            print(f"Archivo: {pdf_path.name}")
            print(f"Páginas procesadas: {num_pages}")
            print(f"Salida: {output_filename}\n")

        except Exception as e:
            logging.error(f"Error crítico procesando {pdf_path.name}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    parser = LegalParser()
    
    # Soporte para argumentos de línea de comandos o carpeta default /data
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            parser.process_file(arg)
    else:
        # Si no hay argumentos, busca en la carpeta data
        data_path = Path("data")
        if data_path.exists():
            files = list(data_path.glob("*.pdf"))
            if not files:
                logging.warning("No hay archivos .pdf en la carpeta 'data/'.")
            for f in files:
                parser.process_file(f)
        else:
            logging.info("Uso: python legal_parser.py ruta/al/archivo.pdf")
            logging.warning("No se encontró la carpeta 'data/' para búsqueda automática.")
