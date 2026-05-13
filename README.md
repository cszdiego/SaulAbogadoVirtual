# Saul — Detector de Cláusulas Abusivas en Contratos Laborales Mexicanos

Saul es un sistema de análisis legal basado en IA que detecta violaciones a la **Ley Federal del Trabajo (LFT)** y a la **Constitución Política de los Estados Unidos Mexicanos (CPEUM)** en contratos laborales individuales.

El proyecto combina **RAG** (Retrieval-Augmented Generation) sobre las leyes mexicanas con un modelo de lenguaje **fine-tuneado con LoRA** para producir análisis legales precisos con citas de artículos específicos.

---

## Demo

![Saul UI](https://via.placeholder.com/800x400?text=Saul+Demo)

El sistema entrega:
- **Puntaje de abusividad** (0–100) con gauge visual
- **Cláusulas problemáticas** ordenadas por severidad (alta / media / baja)
- **Artículo exacto** de la LFT vulnerado en cada cláusula
- **Veredicto legal** con recomendación

---

## Arquitectura

```
PDF / TXT
    │
    ▼
PyMuPDF ──► texto del contrato
    │
    ├──► ChromaDB (RAG) ──► 5 artículos LFT relevantes
    │         ▲
    │    LFT.md + CPEUM.md
    │    (1,450+ artículos indexados)
    │
    └──► Claude claude-haiku-4-5 API
              │
              ▼
         JSON estructurado
         ┌─────────────────────┐
         │ resumen             │
         │ clausulas_abusivas  │
         │ puntaje_abusividad  │
         │ veredicto           │
         └─────────────────────┘
              │
              ▼
         Streamlit UI (Saul)
```

---

## Stack

| Capa | Tecnología |
|---|---|
| UI | Streamlit + CSS (Playfair Display / Cormorant Garamond) |
| LLM | Claude claude-haiku-4-5 (Anthropic API) |
| RAG | ChromaDB + sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` |
| Fine-Tuning | LLaMA-3-8B + LoRA (PEFT + TRL) en Google Colab T4 |
| Infraestructura | Docker Compose (ChromaDB + PostgreSQL) |
| PDF | PyMuPDF |

---

## Instalación

### Requisitos previos
- Python 3.10+
- Docker Desktop
- Una API key de [Anthropic](https://console.anthropic.com/)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/saul.git
cd saul
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 4. Levantar la infraestructura

```bash
docker compose up -d
```

### 5. Indexar las leyes mexicanas en ChromaDB

```bash
python scripts/ingest_laws.py
```

Esto indexa ~1,450 artículos de la LFT y la CPEUM. Tarda ~5 minutos la primera vez (descarga el modelo de embeddings).

### 6. Iniciar la app

```bash
streamlit run app.py
```

Abre [http://localhost:8501](http://localhost:8501)

---

## Scripts disponibles

| Script | Descripción |
|---|---|
| `scripts/ingest_laws.py` | Indexa LFT y CPEUM en ChromaDB |
| `scripts/prepare_dataset.py` | Genera `data/train.jsonl` para fine-tuning |
| `scripts/train.py` | Fine-tuning de LLaMA-3-8B con LoRA (requiere GPU) |
| `scripts/eval_baseline.py` | Evalúa el modelo sin fine-tuning |
| `scripts/eval_finetuned.py` | Evalúa el modelo fine-tuneado |
| `scripts/export_to_gguf.py` | Exporta el modelo a GGUF para Ollama (Colab) |
| `scripts/compare.py` | Comparativa baseline vs fine-tuned en un contrato |

---

## Fine-Tuning

El modelo fue entrenado sobre **10 contratos laborales mexicanos** (5 con violaciones LFT, 5 que cumplen la ley) usando **LoRA** con los siguientes hiperparámetros:

| Parámetro | Valor |
|---|---|
| Modelo base | `unsloth/llama-3-8b-instruct-bnb-4bit` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| Max steps | 60 |
| Batch efectivo | 8 |

Los adaptadores entrenados están en `models/abogado_virtual_lora/`.

Para ejecutar el fine-tuning en Google Colab (GPU T4):

```python
# Colab
!pip install trl datasets peft accelerate bitsandbytes transformers -q
!python scripts/train.py
```

---

## Comparativa de resultados

Sobre el contrato *"Contrato Logístico con Cláusulas de Complejidad Semántica"*:

| Dimensión | Baseline | Fine-Tuned |
|---|---|---|
| Cita artículos LFT | Vago / genérico | Arts. 5, 61, 66, 67, 87, 110 exactos |
| Detecta "salario integral" | Parcialmente | Nulidad + cuantificación del daño |
| Detecta jornada ilegal | Porcentaje incorrecto | 100% / 200% extras correcto |
| Detecta descuentos ilegales | No detecta | Art. 110 LFT + análisis del caso |
| Cita jurisprudencia | No | SCJN sobre salario integral |
| Cuantifica daño económico | No | Tabla con monto anual sustraído |

---

## Estructura del proyecto

```
saul/
├── app.py                          # Aplicación Streamlit (Saul)
├── docker-compose.yml              # ChromaDB + PostgreSQL
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml                 # Tema Legal Noir
├── data/
│   ├── processed/
│   │   ├── LFT.md                  # Ley Federal del Trabajo
│   │   ├── CPEUM.md                # Constitución
│   │   └── Contracts/              # 10 contratos de entrenamiento
│   ├── raw/                        # PDFs originales
│   └── train.jsonl                 # Dataset de fine-tuning (ShareGPT)
├── models/
│   └── abogado_virtual_lora/       # Adaptadores LoRA entrenados
├── scripts/
│   ├── ingest_laws.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── eval_baseline.py
│   ├── eval_finetuned.py
│   ├── export_to_gguf.py
│   └── compare.py
└── reports/                        # Análisis generados
```

---

## Licencia

MIT License — ver [LICENSE](LICENSE)
