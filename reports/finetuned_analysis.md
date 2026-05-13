        # Análisis Post Fine-Tuning — Detección de Cláusulas Abusivas

        **Fecha:** 2026-04-29 22:20:08
        **Modelo base:** `unsloth/llama-3-8b-instruct-bnb-4bit`
        **Adaptadores LoRA:** `C:\Projects\Procesamiento\proyect\models\abogado_virtual_lora`
        **Contrato analizado:** `contratp_malas_practicas.md`
        **RAG:** Artículos recuperados de ChromaDB (Docker local)

        ---

        ## Artículos Utilizados

        **1. [LFT]** Artículo 471.- (Se deroga).



> ### Artículo 471.- (Se deroga).
>
> Artículo derogado DOF 04-01-1980  
> LEY FEDERAL DEL TRABAJO  
> TITULO NOVENO
> Riesgos de Trabajo...

**2. [LFT]** Artículo 134.- Son obligaciones de los trabajadores:

> ### Artículo 134.- Son obligaciones de los trabajadores:
>
> I.-
> Cumplir las disposiciones de las normas de trabajo que les sean aplicables;  
> II.
> Observar las disposiciones contenidas en el reglamento y las normas oficiales mexicanas en
> materia de seguridad, salud y medio ambiente de trabajo, así com...

**3. [CPEUM]** (sin número)

> # Título Sexto
>
> Del Trabajo y de la Previsión Social...

        ---

        ## Análisis del Modelo Fine-Tuneado (Post LoRA)

        **Análisis del contrato**

### CLÁUSULA 1. OBJETO DEL CONTRATO

- **Válido**: El contrato define con precisión el objeto del pacto, es decir, los servicios que EL TRABAJADOR prestará a EL PATRÓN.
- **LFT: Art. 8** (Se cumple): La Ley Federal del Trabajo establece en su artículo 8 que el contrato individual de trabajo debe contener, entre otras cosas, la descripción del puesto o las funciones que el trabajador deberá desempeñar.

### CLÁUSULA 2. DURACIÓN DE LA RELACIÓN LABORAL

- **Válido**: El contrato se celebra por tiempo indeterminado, lo que significa que la relación laboral puede prolongarse indefinidamente.
- **LFT: Art. 35** (Se cumple): La Ley Federal del Trabajo establece en su artículo 35 que el contrato por tiempo indeterminado se celebrará por tiempo ilimitado, salvo que las partes lo convengan en contrario.

### CLÁUSULA 3. PUESTO Y FUNCIONES

#### A) DENOMINACIÓN DEL PUESTO

- **Válido**: EL TRABAJADOR desempeña el puesto de SUPERVISOR DE PISO DE VENTAS.
- **LFT: Art. 8** (Se cumple): La Ley Federal del Trabajo establece en su artículo 8 que el contrato debe contener la descripción del puesto o las funciones que el trabajador deberá desempeñar.

#### B) RESPONSABILIDADES

- **Válido**: EL TRABAJADOR se obliga a realizar todas las actividades inherentes al puesto de SUPERVISOR DE PISO DE VENTAS, incluyendo la supervisión del personal de ventas y el control del inventario de productos.
- **LFT: Art. 124** (Se cumple): La Ley Federal del Trabajo establece en su artículo 124 que el trabajador está obligado a desempeñar las funciones inherentes al puesto para el cual ha sido contratado.

#### C) CAPACITACIÓN Y ASISTENCIA

- **Válido**: EL PATRÓN se obliga a proporcionar la capacitación y asistencia técnica necesaria a EL TRABAJADOR para que desempeñe con éxito sus funciones.

---

        ---

        ## Cómo Comparar con el Baseline

        Abre en paralelo:
        - `reports/baseline_analysis.md`  — modelo sin entrenar
        - `reports/finetuned_analysis.md` — este reporte (modelo fine-tuneado)

        Señales de mejora esperadas:
        - Cita artículos específicos de la LFT (Art. 5, 87, 80, 110, 56)
        - Identifica las 3 violaciones del contrato de prueba
        - Respuesta más estructurada y precisa legalmente
