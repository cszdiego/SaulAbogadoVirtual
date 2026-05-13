"""
Genera data/train.jsonl para fine-tuning del Abogado Virtual.

Formato de salida: ShareGPT (conversaciones con roles system/human/gpt),
compatible con Unsloth y HuggingFace TRL SFTTrainer.

Las etiquetas (análisis) están escritas por un experto y son la fuente
de verdad del entrenamiento — no se generan automáticamente.

Ejecución (desde la raíz del proyecto):
    python scripts/prepare_dataset.py
"""
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = BASE_DIR / "data" / "processed" / "Contracts"
OUTPUT_PATH = BASE_DIR / "data" / "train.jsonl"

SYSTEM_PROMPT = (
    "Eres un experto legal mexicano especializado en derecho laboral. "
    "Tu tarea es analizar contratos individuales de trabajo y detectar "
    "violaciones a la Ley Federal del Trabajo (LFT) o a la Constitución "
    "Política de los Estados Unidos Mexicanos (CPEUM). "
    "Para cada irregularidad debes citar: (1) la cláusula del contrato, "
    "(2) el artículo de la LFT que se vulnera, y (3) una explicación precisa. "
    "Si el contrato cumple con la normativa vigente, indícalo con la misma precisión."
)

USER_TEMPLATE = (
    "Analiza este contrato laboral y detecta posibles violaciones a la "
    "Ley Federal del Trabajo (LFT) o confirma su cumplimiento:\n\n{contract_text}"
)

# ---------------------------------------------------------------------------
# Catálogo de contratos
# Estructura: filename → {label, analysis}
# label: "abusivo" | "cumple"
# analysis: respuesta experta que el modelo debe aprender a generar
# ---------------------------------------------------------------------------

CONTRACT_CATALOG = {

    # -----------------------------------------------------------------------
    # CONTRATOS CON VIOLACIONES
    # -----------------------------------------------------------------------

    "contratp_malas_practicas.md": {
        "label": "abusivo",
        "analysis": """## Análisis Legal — GRUPO COMERCIAL MAYORISTA DEL SURESTE

Este contrato presenta **tres violaciones directas** a la Ley Federal del Trabajo:

---

### ❌ VIOLACIÓN 1 — Cláusula Séptima B): Salario Integral que Absorbe Prestaciones Irrenunciables

**Cláusula:** "El salario pactado tiene carácter integral [...] comprende componentes ordinarios y una porción distributiva de gratificaciones anuales anticipadas."

**Artículos violados:**
- **Art. 5 LFT**: Las disposiciones de esta Ley son de orden público y los derechos que otorga son irrenunciables. Cualquier cláusula que prive al trabajador de sus prestaciones es nula de pleno derecho.
- **Art. 82 LFT**: El salario es la retribución que debe pagar el patrón al trabajador por su trabajo, distinta e independiente de las prestaciones de ley.
- **Art. 87 LFT**: Los trabajadores tendrán derecho a un aguinaldo anual de 15 días de salario como mínimo. Esta prestación no puede subsumirse ni prorratearse dentro del salario ordinario.
- **Art. 80 LFT**: El trabajador tiene derecho a una prima vacacional no menor del 25% sobre los salarios del período de vacaciones, que tampoco puede anticiparse ni absorberse.

**Consecuencia:** La cláusula es nula. El trabajador conserva su derecho íntegro a recibir aguinaldo y prima vacacional en las fechas y montos que establece la ley, independientemente del "salario integral" pactado.

---

### ❌ VIOLACIÓN 2 — Cláusula Décima Quinta B): Descuentos por Faltantes de Caja

**Cláusula:** "EL TRABAJADOR participará proporcionalmente en la reposición del faltante mediante descuentos quincenales."

**Artículos violados:**
- **Art. 110 LFT**: Los descuentos en los salarios de los trabajadores están estrictamente prohibidos salvo en los casos taxativamente enumerados (pago de deudas al patrón por anticipo de salarios, errores, pérdidas, averías o adquisición de artículos producidos por la empresa). Los faltantes de caja por diferencias de arqueo NO están contemplados en ese listado.
- **Art. 18 LFT**: En la interpretación de las normas de trabajo, en caso de duda, prevalecerá la interpretación más favorable al trabajador.

**Consecuencia:** El descuento es ilegal aunque el trabajador lo haya "aceptado expresamente". Su consentimiento no valida una cláusula contraria a la ley. El patrón que aplique estos descuentos comete una violación sancionable con multa y devolución.

---

### ❌ VIOLACIÓN 3 — Cláusula Tercera C): Cambio Unilateral de Centro de Trabajo

**Cláusula:** "EL TRABAJADOR prestará sus servicios [...] o en cualquier otra sucursal que EL PATRÓN determine conforme a sus necesidades operativas."

**Artículos violados:**
- **Art. 51 fracción II LFT**: El trabajador podrá separarse de su trabajo sin incurrir en responsabilidad cuando el patrón reduzca su salario o cambie unilateralmente las condiciones de trabajo en su perjuicio.
- **Art. 56 LFT**: Las condiciones de trabajo no podrán ser inferiores a las fijadas en la ley y deberán ser proporcionales a la importancia de los servicios e iguales para trabajos iguales.
- **Art. 9 LFT**: El lugar de prestación del servicio es un elemento esencial del contrato y su modificación unilateral constituye un cambio de condiciones de trabajo.

**Consecuencia:** Un cambio unilateral de sucursal sin consentimiento del trabajador constituye una modificación ilegal de las condiciones de trabajo que puede fundar una rescisión imputable al patrón con derecho a indemnización constitucional (3 meses de salario + 20 días por año).

---

**Veredicto:** Este contrato contiene cláusulas abusivas de nulidad absoluta. Se recomienda su revisión completa antes de la firma.""",
    },

    "contrato.md": {
        "label": "abusivo",
        "analysis": """## Análisis Legal — OPERADORA COMERCIAL RETAIL S.A. DE C.V.

Este contrato presenta **dos violaciones graves** disfrazadas de cláusulas técnicas:

---

### ❌ VIOLACIÓN 1 — Cláusula Tercera: Jornada Extendida Sin Pago de Horas Extraordinarias

**Cláusula:** "la jornada se extenderá y adaptará según los flujos operativos [...] sin que esto amerite un cálculo independiente por tiempo extraordinario."

**Artículos violados:**
- **Art. 61 LFT**: La duración máxima de la jornada diurna es de ocho horas, y la semana laboral no podrá exceder 48 horas. Este límite es absoluto y no puede pactarse su extensión ordinaria.
- **Art. 66 LFT**: Podrá también prolongarse la jornada de trabajo por circunstancias extraordinarias, sin que pueda exceder nunca de tres horas diarias ni de tres veces en una semana.
- **Art. 67 LFT**: Las horas de trabajo extraordinario se pagarán con un ciento por ciento más del salario que corresponda a las horas de la jornada. Cuando las horas extras excedan de nueve a la semana, se pagarán con un doscientos por ciento adicional.
- **Art. 5 fracción III LFT**: Será nula cualquier estipulación que establezca jornadas de trabajo notoriamente excesivas.

**Consecuencia:** La cláusula es nula. Las horas trabajadas más allá de la jornada legal deben pagarse como horas extraordinarias con los recargos de ley, independientemente del puesto o de lo pactado en contrato.

---

### ❌ VIOLACIÓN 2 — Cláusula Cuarta: Salario Integral que Absorbe Aguinaldo y Prima Vacacional

**Cláusula:** "el salario integrado [...] incluye la parte proporcional correspondiente a gratificaciones anuales anticipadas y primas vacacionales, dándose por cubiertas dichas prestaciones."

**Artículos violados:**
- **Art. 5 LFT**: Las prestaciones de la LFT son irrenunciables e inembargables. Una cláusula contractual no puede suprimir ni absorber derechos mínimos.
- **Art. 87 LFT**: El aguinaldo mínimo de 15 días de salario debe pagarse antes del 20 de diciembre. Prorratear su importe en el salario mensual no cumple con esta obligación.
- **Art. 80 LFT**: La prima vacacional del 25% debe pagarse al momento de disfrutar las vacaciones, no de forma anticipada ni prorrateada.

**Consecuencia:** El trabajador tiene derecho a recibir aguinaldo y prima vacacional en los términos y fechas de la ley. El patrón que argue haber "pagado anticipado" mediante el salario integral deberá acreditarlo documentalmente o volver a pagar.

---

**Veredicto:** Contrato con cláusulas abusivas que ocultan violaciones a la LFT mediante redacción técnica. Nulas de pleno derecho.""",
    },

    "contrato_empleado.md": {
        "label": "abusivo",
        "analysis": """## Análisis Legal — SEGURIDAD PRIVADA CORPORATIVA ÉLITE S.A. DE C.V.

Este contrato presenta **una violación estructural** relativa a la jornada de trabajo:

---

### ❌ VIOLACIÓN ÚNICA — Cláusula Tercera: Jornada de Guardia Sin Límite y Absorbida en Salario

**Cláusula:** "la jornada se extenderá de manera flexible según los flujos operativos [...] integrando estas horas de permanencia a la compensación global pactada [...] por lo que dichas extensiones operativas se consideran saldadas integralmente."

**Artículos violados:**
- **Art. 61 LFT**: Ningún tipo de contrato puede establecer como cláusula ordinaria una jornada de duración indefinida. El máximo es 48 horas semanales para jornada diurna o 42 horas para jornada nocturna.
- **Art. 65 LFT**: En los casos de siniestro o riesgo inminente que pongan en peligro la vida del trabajador, de sus compañeros o del patrón, o la existencia de la empresa, la jornada de trabajo podrá prolongarse el tiempo estrictamente indispensable. Este es el único supuesto de extensión legítima, y aplica solo en emergencias, no como práctica ordinaria.
- **Art. 66 LFT**: Las horas extras deben pagarse, aun en puestos de vigilancia y custodia.
- **Art. 5 fracción III LFT**: Es nula cualquier estipulación que establezca jornadas notoriamente excesivas.

**Contexto especial del sector:** Los trabajadores de vigilancia y seguridad privada tienen derecho pleno al pago de horas extraordinarias. La Suprema Corte de Justicia de la Nación ha reiterado que la naturaleza del servicio no exime al patrón de cumplir con los límites de jornada ni con el pago de tiempo extra.

**Consecuencia:** Las horas trabajadas que excedan los límites legales deben pagarse con un 100% adicional (primeras 9 horas semanales) y un 200% adicional (excedente). El patrón no puede absorber estas horas en el salario ordinario sin violar la ley.

---

**Veredicto:** Contrato con cláusula abusiva de nulidad absoluta en materia de jornada. Afecta especialmente a trabajadores de turno nocturno y rotativo.""",
    },

    "indv_contract.md": {
        "label": "abusivo",
        "analysis": """## Análisis Legal — LOGÍSTICA Y DISTRIBUCIÓN EXPRESS S.A. DE C.V.

Este contrato presenta **dos violaciones graves** que afectan jornada y prestaciones:

---

### ❌ VIOLACIÓN 1 — Cláusula Tercera: Jornada Indefinida Condicionada al Cumplimiento de Ruta

**Cláusula:** "la jornada diaria se considerará concluida de manera efectiva una vez que se complete al cien por ciento la ruta de entrega asignada [...] sin que el tiempo de espera o conclusión de ruta genere un cálculo independiente por tiempo extraordinario."

**Artículos violados:**
- **Art. 58 LFT**: La jornada de trabajo es el tiempo durante el cual el trabajador está a disposición del patrón para prestar su trabajo. El tiempo en ruta —incluyendo esperas, imprevistos de tráfico y conclusión de entregas— es tiempo a disposición del patrón.
- **Art. 61 LFT**: La jornada máxima diurna es de 8 horas diarias y 48 horas semanales. Una jornada cuya duración depende de factores externos (tráfico, distancia de ruta) puede exceder ilimitadamente estos topes.
- **Art. 67 LFT**: Todo tiempo trabajado más allá de la jornada ordinaria es tiempo extraordinario que debe pagarse con el cien por ciento adicional.

**Consecuencia:** La cláusula transfiere al trabajador el riesgo operativo de la empresa (retrasos, rutas largas). Este tipo de cláusula ha sido declarada nula reiteradamente en laudos arbitrales y sentencias laborales mexicanas.

---

### ❌ VIOLACIÓN 2 — Cláusula Cuarta: Salario Integral que Absorbe Prestaciones Legales

**Cláusula:** "este esquema de remuneración integral incluye la parte proporcional de gratificaciones anuales anticipadas y primas vacacionales, dándose por cubiertas y liquidadas dichas prestaciones de ley de manera prorrateada."

**Artículos violados:**
- **Art. 5 LFT**: Son nulas las cláusulas que impliquen renuncia de derechos laborales. El aguinaldo y la prima vacacional son derechos irrenunciables.
- **Art. 87 LFT**: El aguinaldo es una prestación que debe pagarse en diciembre, no puede anticiparse ni prorratearse en el salario diario.
- **Art. 80 LFT**: La prima vacacional del 25% debe pagarse al inicio del período de vacaciones efectivo.
- **Art. 84 LFT**: El salario se integra con los pagos hechos en efectivo por cuota diaria, más las demás prestaciones. Las prestaciones de ley no sustituyen ni se confunden con el salario base.

**Agravante:** El salario pactado ($5,200.00 catorcenal = ~$371.43 diarios) se acerca peligrosamente al salario mínimo general vigente. Si el "salario integral" pretende absorber prestaciones en un monto tan bajo, el trabajador queda en situación de sub-remuneración efectiva.

---

**Veredicto:** Contrato con doble violación de nulidad absoluta. Especialmente grave por combinar jornada indeterminada con salario cercano al mínimo.""",
    },

    "dataset_clausulas_abusivas_entrenamiento.md": {
        "label": "abusivo",
        "analysis": """## Análisis Legal — SOLUCIONES TECNOLÓGICAS DEL GOLFO S.A. DE C.V.
## ⚠️ Documento de Referencia con Cláusulas Abusivas Etiquetadas

Este documento contiene múltiples ejemplos de cláusulas ilegales con su análisis de nulidad. Se identifican las siguientes violaciones:

---

### ❌ VIOLACIÓN 1 — Jornada Excesiva Sin Compensación

**Cláusula:** "jornada semanal de 69 horas [...] no tendrá derecho al pago de tiempo extraordinario."

**Artículos violados:**
- **Art. 61 LFT**: Máximo 48 horas semanales (jornada diurna) o 45 horas (jornada mixta).
- **Art. 66 LFT**: Límite de 3 horas extras diarias y máximo 3 veces por semana.
- **Art. 67 LFT**: Horas extras se pagan al doble (primeras 9 hrs/semana) y al triple (excedente).
- **Art. 5 fracción III LFT**: Nula toda cláusula que establezca jornadas notoriamente excesivas.

**Consecuencia:** Jornada de 69 horas excede en 21 horas el máximo legal. La cláusula es nula y el patrón debe pago retroactivo de todas las horas laboradas fuera de la jornada ordinaria.

---

### ❌ VIOLACIÓN 2 — Renuncia Expresa a Prestaciones Irrenunciables

**Cláusula:** "EL TRABAJADOR renuncia expresamente y de forma voluntaria a percibir: Aguinaldo anual, Prima vacacional, PTU, Prima dominical, Días de descanso obligatorio adicionales."

**Artículos violados:**
- **Art. 5 LFT**: Cualquier estipulación que establezca renuncia de derechos irrenunciables es nula.
- **Art. 87 LFT**: Aguinaldo mínimo de 15 días — irrenunciable.
- **Art. 80 LFT**: Prima vacacional del 25% — irrenunciable.
- **Art. 117-131 LFT**: Participación en Utilidades (PTU) — derecho constitucional (Art. 123 CPEUM fracción IX), irrenunciable.
- **Art. 73 LFT**: Prima dominical del 25% por trabajar el día de descanso semanal — irrenunciable.
- **Art. 74 LFT**: Días de descanso obligatorio con goce de salario — irrenunciables.

**Consecuencia:** La renuncia es completamente nula aunque esté "voluntariamente" firmada. El consentimiento no puede validar la renuncia a derechos de orden público. El trabajador conserva todos estos derechos íntegramente.

---

### ❌ VIOLACIÓN 3 — Multas y Descuentos Disciplinarios Ilegales

Si el contrato incluye cláusulas de multas internas o descuentos por faltas disciplinarias (tardanzas, errores, daños), estas violan:
- **Art. 110 LFT**: Solo se permiten descuentos expresamente listados (anticipos, IMSS, INFONAVIT, créditos de caja de ahorro, pension alimenticia). Las multas disciplinarias NO están incluidas.
- **Art. 107 LFT**: El salario mínimo es inembargable e intocable mediante descuento.

---

**Veredicto:** Este contrato contiene cláusulas de nulidad absoluta en múltiples dimensiones. Ninguna es salvable mediante acuerdo de partes. El trabajador que haya firmado conserva todos sus derechos y puede reclamarlos ante los Tribunales Laborales.""",
    },

    # -----------------------------------------------------------------------
    # CONTRATOS QUE CUMPLEN
    # -----------------------------------------------------------------------

    "contract.md": {
        "label": "cumple",
        "analysis": """## Análisis Legal — CONSTRUCTORA Y EDIFICACIONES INDUSTRIALES S.A. DE C.V.

Tras revisar el contrato en su integridad, se concluye que **cumple con la normativa laboral vigente**:

---

### ✅ Jornada de Trabajo — Art. 61 LFT

La jornada diurna de lunes a viernes (08:00-17:00) y sábados (08:00-13:00) totaliza **48 horas semanales**, equivalente exactamente al máximo legal establecido en el artículo 61 LFT. La hora de comida está correctamente excluida del cómputo de la jornada efectiva.

Las horas extraordinarias requieren autorización expresa por escrito y se pagarán conforme a los artículos 66 y 67 LFT, lo que garantiza el derecho al pago de tiempo extra.

---

### ✅ Vacaciones Dignas — Art. 76 LFT

El contrato otorga **12 días de vacaciones en el primer año**, cumpliendo con la reforma de "Vacaciones Dignas" (2023) que elevó el mínimo de 6 a 12 días anuales. La escala de incremento es conforme a la tabla del artículo 76 LFT reformado.

---

### ✅ Prima Vacacional — Art. 80 LFT

Se reconoce la prima vacacional del **25%** sobre el salario de vacaciones, en estricto cumplimiento del artículo 80 LFT.

---

### ✅ Aguinaldo — Art. 87 LFT

El aguinaldo equivale a **15 días de salario**, pagadero antes del 20 de diciembre, exactamente como lo exige el artículo 87 LFT. Se reconoce el pago proporcional para trabajadores con menos de un año.

---

### ✅ Jurisdicción Laboral — Art. 604 LFT

La sumisión a los Tribunales Laborales Federales o Locales es correcta y no representa una renuncia a derechos procesales del trabajador.

---

**Veredicto:** Este contrato cumple íntegramente con la Ley Federal del Trabajo. No se detectan cláusulas abusivas ni intentos de absorber prestaciones irrenunciables. Es un contrato laboral legalmente válido.""",
    },

    "contrato_indiv.md": {
        "label": "cumple",
        "analysis": """## Análisis Legal — SOLUCIONES FINANCIERAS INTEGRALES S.A. DE C.V.

Tras revisar el contrato en su integridad, se concluye que **cumple y supera los mínimos legales**:

---

### ✅ Jornada de Trabajo — Art. 61 LFT

La jornada de **40 horas semanales** (lunes a viernes, 09:00-18:00 con hora de comida excluida) es notablemente inferior al máximo legal de 48 horas, lo que constituye una condición más favorable para el trabajador conforme al artículo 56 LFT.

Las horas extraordinarias requieren autorización previa por escrito y se pagarán con los recargos del 100% (primeras 9 horas semanales) y 200% (excedente), en estricto cumplimiento de los artículos 66, 67 y 68 LFT.

---

### ✅ Aguinaldo Superior a la Ley — Art. 87 LFT

El contrato establece un aguinaldo de **30 días de salario**, duplicando el mínimo legal de 15 días. Esta prestación superior es plenamente válida conforme al artículo 56 LFT, que permite condiciones más favorables para el trabajador.

---

### ✅ Prima Vacacional — Art. 80 LFT

La prima vacacional del **25%** cumple exactamente con el artículo 80 LFT. Al ser otorgada como prestación separada del salario ordinario, no existe confusión con ningún esquema de "salario integral".

---

### ✅ Prestaciones Legales Independientes

Las prestaciones (aguinaldo, prima vacacional, días de descanso) se establecen de forma expresa e independiente del salario base, sin ningún intento de absorción o prorrateo, respetando el artículo 5 LFT sobre irrenunciabilidad.

---

**Veredicto:** Contrato que cumple la LFT y otorga prestaciones superiores al mínimo legal. No se detecta ninguna violación. Es un ejemplo de contrato laboral con condiciones favorables para el trabajador.""",
    },

    "contrato_individual.md": {
        "label": "cumple",
        "analysis": """## Análisis Legal — CONSTRUCTORA E INMOBILIARIA PENINSULAR S.A. DE C.V.

Tras revisar el contrato en su integridad, se concluye que **cumple con la normativa laboral vigente**:

---

### ✅ Jornada de Trabajo — Arts. 58, 59, 60 y 61 LFT

La jornada **diurna de 40 horas semanales** (lunes a viernes) respeta ampliamente el máximo legal de 48 horas establecido en el artículo 61 LFT. La determinación explícita del tipo de jornada (diurna: entre las 6:00 y las 20:00 horas) cumple con la definición del artículo 60 LFT y garantiza claridad sobre las condiciones de trabajo.

La reducción respecto al máximo legal (8 horas vs máximo 48 semanales) representa una condición más favorable para la trabajadora conforme al artículo 56 LFT.

---

### ✅ Precisión en Condiciones de Trabajo

El contrato especifica con claridad el puesto, horario y funciones, cumpliendo con el requisito del artículo 25 LFT sobre los elementos mínimos del contrato individual de trabajo.

---

### ✅ Ausencia de Cláusulas Abusivas

No se detectan intentos de:
- Absorber prestaciones irrenunciables en el salario (Art. 5 LFT).
- Extender la jornada sin pago de tiempo extraordinario (Arts. 66-67 LFT).
- Imponer descuentos no autorizados (Art. 110 LFT).
- Modificar unilateralmente condiciones de trabajo (Art. 56 LFT).

---

**Veredicto:** Este contrato cumple íntegramente con la Ley Federal del Trabajo. No se identifican irregularidades. Las condiciones pactadas son iguales o superiores a los mínimos legales.""",
    },

    "contrato_individual_trabajo.md": {
        "label": "cumple",
        "analysis": """## Análisis Legal — TECNOLOGÍAS DEL SURESTE S.A. DE C.V.

Tras revisar el contrato en su integridad, se concluye que **cumple con la normativa laboral vigente**:

---

### ✅ Jornada de Trabajo — Art. 61 LFT

La jornada de **40 horas semanales** (lunes a viernes, 09:00-18:00 con hora de comida de 14:00-15:00 excluida del cómputo) cumple con el límite legal. La posibilidad de modificación de horario queda condicionada a no exceder las 48 horas semanales y a notificación con 7 días de anticipación, lo que respeta el artículo 56 LFT sobre modificación de condiciones.

---

### ✅ Prima Vacacional — Art. 80 LFT

La prima vacacional del **25%** se establece como prestación independiente pagadera antes del período vacacional, cumpliendo literalmente con el artículo 80 LFT. No hay confusión con el salario ordinario.

---

### ✅ Días de Descanso Obligatorio — Art. 74 LFT

El contrato reconoce expresamente los días de descanso obligatorio establecidos en el artículo 74 LFT, incluyendo la referencia a los días electorales, con pleno goce de salario.

---

### ✅ Confidencialidad Dentro del Marco Legal

La obligación de confidencialidad se fundamenta en los artículos 134 fracción XIII y 135 fracción VI LFT, que son las disposiciones legales específicas para la protección de secretos comerciales y técnicos. Al estar referenciada en la ley, no excede las facultades del patrón.

---

**Veredicto:** Contrato laboral válido que cumple con la LFT. No se detectan cláusulas abusivas. La jornada, las prestaciones y las obligaciones del trabajador están dentro del marco legal vigente.""",
    },

    "contrato_mejores_practicas.md": {
        "label": "cumple",
        "analysis": """## Análisis Legal — COMERCIALIZADORA DE ALIMENTOS DEL PACÍFICO S.A. DE C.V.

Tras revisar el contrato en su integridad, se concluye que **constituye un modelo de mejores prácticas en contratación laboral**:

---

### ✅ Tiempo Indeterminado con Inicio Preciso — Art. 35 LFT

El contrato es por tiempo indeterminado desde el 1 de junio de 2025, sin condiciones ni términos que limiten artificialmente su duración, cumpliendo con el artículo 35 LFT que establece la preferencia por contratos indefinidos.

---

### ✅ Jornada Claramente Definida — Arts. 58-61 LFT

La jornada está explícitamente definida en horas y días, con los límites legales correctamente citados. No existen cláusulas de "disponibilidad" ni de "jornada flexible" que pudieran vulnerar el artículo 61 LFT.

---

### ✅ Prestaciones Separadas e Irrenunciables — Arts. 5, 80, 87 LFT

Las prestaciones de ley (aguinaldo, prima vacacional, vacaciones) se establecen de forma independiente del salario base. No existe ningún "salario integral" ni "compensación global" que pretenda absorber derechos irrenunciables.

---

### ✅ Capacidad Económica Declarada — Art. 25 LFT

El patrón declara expresamente contar con los recursos necesarios para cumplir todas las obligaciones laborales, incluyendo seguridad social. Esta declaración fortalece la posición del trabajador en caso de controversia.

---

### ✅ Datos de Seguridad Social Completos — Arts. 15 LSS y 29 LINFONAVIT

Se detallan el NSS, RFC y CURP del trabajador, así como el Número Patronal ante IMSS e INFONAVIT. El patrón se obliga a registrar al trabajador dentro de los plazos legales.

---

### ✅ Ausencia Total de Cláusulas Abusivas

Revisión exhaustiva confirma que no existen:
- Absorciones de prestaciones en salario (Art. 5 LFT).
- Jornadas indeterminadas o extensiones sin pago (Arts. 61, 66-67 LFT).
- Descuentos ilegales (Art. 110 LFT).
- Traslados unilaterales (Art. 51-56 LFT).
- Renuncias anticipadas a derechos (Art. 5 LFT).

---

**Veredicto:** Este contrato es un modelo ejemplar de contratación laboral conforme a derecho. Supera los estándares mínimos de la LFT y puede usarse como referencia de buenas prácticas.""",
    },
}

# ---------------------------------------------------------------------------
# Lógica de construcción del dataset
# ---------------------------------------------------------------------------

def load_contract(filename: str) -> str | None:
    path = CONTRACTS_DIR / filename
    if not path.exists():
        print(f"  [ADVERTENCIA] No encontrado: {filename}")
        return None
    return path.read_text(encoding="utf-8").strip()


def make_example(contract_text: str, analysis: str) -> dict:
    """
    Formato ShareGPT — compatible con Unsloth y HuggingFace TRL SFTTrainer.

    Unsloth espera: {"conversations": [{"from": "system"|"human"|"gpt", "value": "..."}]}
    TRL SFTTrainer con apply_chat_template también acepta este formato.
    """
    return {
        "conversations": [
            {
                "from": "system",
                "value": SYSTEM_PROMPT,
            },
            {
                "from": "human",
                "value": USER_TEMPLATE.format(contract_text=contract_text),
            },
            {
                "from": "gpt",
                "value": analysis,
            },
        ]
    }


def main() -> None:
    sep = "=" * 54
    print(sep)
    print("  Preparación del Dataset de Fine-Tuning")
    print(sep)

    examples = []
    stats = {"cumple": 0, "abusivo": 0, "no_encontrado": 0}

    for filename, meta in CONTRACT_CATALOG.items():
        contract_text = load_contract(filename)
        if contract_text is None:
            stats["no_encontrado"] += 1
            continue

        example = make_example(contract_text, meta["analysis"])
        examples.append(example)
        stats[meta["label"]] += 1

        label_icon = "✓" if meta["label"] == "cumple" else "✗"
        print(f"  {label_icon} [{meta['label']:8}] {filename}")

    # Guardar JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Verificación rápida
    written = sum(1 for _ in OUTPUT_PATH.open(encoding="utf-8"))

    print(f"\n{sep}")
    print(f"  Ejemplos generados  : {len(examples)}")
    print(f"    Contratos buenos  : {stats['cumple']}")
    print(f"    Contratos abusivos: {stats['abusivo']}")
    print(f"    No encontrados    : {stats['no_encontrado']}")
    print(f"  Líneas escritas     : {written}")
    print(f"  Archivo de salida   : {OUTPUT_PATH}")
    print(sep)

    if stats["no_encontrado"] > 0:
        print("\n  [ADVERTENCIA] Algunos archivos no se encontraron.")
        print("  Verifica que los nombres en CONTRACT_CATALOG coincidan")
        print(f"  con los archivos en: {CONTRACTS_DIR}")


if __name__ == "__main__":
    main()
