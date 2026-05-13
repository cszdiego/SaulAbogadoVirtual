        # Análisis Baseline — Detección de Cláusulas Abusivas

        **Fecha:** 2026-04-29 11:59:30
        **Modelo:** `llama3.2:3b` (backend: OLLAMA)
        **Contrato analizado:** `contratp_malas_practicas.md`
        **Colección RAG:** `leyes_mexicanas` (5 artículos recuperados)

        ---

        ## Artículos Recuperados por RAG

        **1. [LFT]** Artículo 471.- (Se deroga).

> ### Artículo 471.- (Se deroga).
Artículo derogado DOF 04-01-1980  
LEY FEDERAL DEL TRABAJO  
TITULO NOVENO
Riesgos de Trabajo...

**2. [LFT]** Artículo 134.- Son obligaciones de los trabajadores:

> ### Artículo 134.- Son obligaciones de los trabajadores:  
I.-
Cumplir las disposiciones de las normas de trabajo que les sean aplicables;  
II.
Observar las disposiciones contenidas en el reglamento y las normas oficiales mexicanas en
materia de seguridad, salud y medio ambiente de trabajo, así com...

**3. [CPEUM]** (sin número)

> # Título Sexto
Del Trabajo y de la Previsión Social...

**4. [LFT]** Artículo 888.- Se deroga.

> ### Artículo 888.- Se deroga.  
LEY FEDERAL DEL TRABAJO  
Artículo , 30-11-2012. Derogado DOF 01-05-2019...

**5. [LFT]** Artículo 337.- Los patrones tienen las obligaciones especiales siguientes:

> ### Artículo 337.- Los patrones tienen las obligaciones especiales siguientes:  
I.
Guardar consideración al trabajador del hogar, absteniéndose de todo mal trato de palabra o de
obra.
Fracción  
II.
Proporcionar al trabajador habitación cómoda e higiénica, alimentación sana y suficiente y
condicion...


        ---

        ## Análisis del Modelo (Pre Fine-Tuning)

        A continuación, te presento las irregularidades encontradas en el contrato:

**Cláusula 1: OBJETO DEL CONTRATO**

* Cláusula del contrato: PRIMERA. OBJETO DEL CONTRATO
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se especifica qué tipo de objeto se está contrataando (mayorista, por ejemplo).
* Explicación: La cláusula 1 del contrato establece el objetivo del contrato, pero no se proporciona una definición clara y concisa de lo que se está contratando. Esto puede generar confusión y malentendidos en el futuro.

**Cláusula 2: DURACIÓN DE LA RELACIÓN LABORAL**

* Cláusula del contrato: SEGUNDA. DURACIÓN DE LA RELACIÓN LABORAL
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se especifica qué tipo de duración se está contratando (tiempo indefinido, por ejemplo).
* Explicación: La cláusula 2 del contrato establece la duración de la relación laboral como tiempo indefinido, pero no se proporciona una definición clara y concisa de lo que significa "tiempo indefinido" en el contexto de la Ley Federal del Trabajo.

**Cláusula 3: PUESTO Y FUNCIONES**

* Cláusula del contrato: TERCERA. PUESTO Y FUNCIONES
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se proporcionan detalles suficientes sobre las funciones y responsabilidades del puesto.
* Explicación: La cláusula 3 del contrato establece el puesto de supervisión de piso de ventas, pero no se proporcionan detalles suficientes sobre las funciones y responsabilidades específicas del puesto. Esto puede generar confusión y malentendidos en el futuro.

**Cláusula 4: JORNADA DE TRABAJO**

* Cláusula del contrato: CUARTA. JORNADA DE TRABAJO
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se especifican los turnos y horarios de trabajo.
* Explicación: La cláusula 4 del contrato establece la jornada de trabajo como mixta, pero no se proporcionan detalles suficientes sobre los turnos y horarios de trabajo. Esto puede generar confusión y malentendidos en el futuro.

**Cláusula 5: DISPONIBILIDAD OPERATIVA**

* Cláusula del contrato: C) DISPONIBILIDAD OPERATIVA
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se especifican las condiciones bajo las cuales se extenderá la jornada.
* Explicación: La cláusula 4 del contrato establece que la jornada podrá extenderse en casos de contingencias operativas, pero no se proporcionan detalles suficientes sobre las condiciones bajo las cuales se extenderá la jornada. Esto puede generar confusión y malentendidos en el futuro.

**Cláusula 6: PERIODO DE ALIMENTOS**

* Cláusula del contrato: D) PERIODO DE ALIMENTOS
* Artículo de la LFT vulnerado: No se encuentra un artículo específico que se vulnera, pero se puede argumentar que la cláusula no es clara o completa, ya que no se especifica el tiempo exacto en que se tomarán los alimentos.
* Explicación: La cláusula 6 del contrato establece que EL TRABAJADOR dispondrá de 30 minutos para tomar sus alimentos, pero no se proporciona un tiempo exacto en que se tomarán los alimentos. Esto puede generar confusión y malentendidos en el futuro.

En resumen, no se encontraron vulneraciones directas de la LFT en el contrato, pero sí se identificaron algunas cláusulas que podrían ser claras o completas para evitar confusiones y malentendidos en

        ---

        *Este reporte sirve como línea base (baseline) antes del fine-tuning con LoRA.*
        *Comparar con el análisis post-entrenamiento para medir la mejora del modelo.*
