"""
Saul — Análisis Legal de Contratos Laborales Mexicanos
"""
import os
from typing import Literal

import anthropic
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CHROMA_HOST    = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT    = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION     = "leyes_mexicanas"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K          = 5

MODEL          = "claude-haiku-4-5"
MAX_TOKENS     = 8192
MAX_CONTRACT   = 8000   # chars enviados al modelo

# Sistema de costos claude-haiku-4-5 (USD por token)
COST_INPUT     = 1.00 / 1_000_000
COST_CACHED    = 0.10 / 1_000_000   # 0.1x al leer del caché
COST_OUTPUT    = 5.00 / 1_000_000

# ---------------------------------------------------------------------------
# System prompt (largo para superar mínimo de caché de 2048 tokens en Haiku)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Eres un abogado laboral mexicano con más de 20 años de experiencia \
detectando violaciones a la Ley Federal del Trabajo (LFT) y a los derechos constitucionales \
de los trabajadores establecidos en el Artículo 123 de la CPEUM. Tu análisis es SEVERO, \
PRECISO y siempre cita el artículo exacto que se vulnera.

## PRINCIPIOS IRRENUNCIABLES DE LA LFT QUE SIEMPRE DEBES APLICAR

**Art. 5 — Irrenunciabilidad absoluta:**
Las prestaciones de ley son irrenunciables. Cualquier cláusula donde el trabajador "renuncia \
expresamente" a aguinaldo, prima vacacional, PTU, prima dominical o días de descanso es NULA \
DE PLENO DERECHO aunque esté firmada, aunque diga "salario integral" o "compensación global". \
El consentimiento no puede validar la renuncia a derechos de orden público.

**Arts. 58-61 — Jornada máxima:**
Diurna: 8 horas diarias / 48 horas semanales.
Mixta: 7.5 horas diarias / 45 horas semanales.
Nocturna: 7 horas diarias / 42 horas semanales.
Cualquier extensión de la jornada bajo el pretexto de "necesidades operativas", "jornada flexible" \
o "puestos de confianza" SIN pago de tiempo extraordinario viola los artículos 66 y 67 LFT.
Los puestos de confianza SÍ tienen derecho a pago de horas extras (Art. 9 y 66 LFT).

**Arts. 66-68 — Horas extraordinarias:**
Máximo 3 horas extras diarias, 3 veces por semana.
Primeras 9 horas semanales: 100% adicional al salario ordinario.
Excedente de 9 horas semanales: 200% adicional.
Una cláusula que diga que el salario "incluye" las horas extras es ilegal.

**Art. 76 (Vacaciones Dignas, reforma 2023):**
Año 1: 12 días. Año 2: 14 días. Año 3: 16 días. Año 4: 18 días. Años 5-9: 20 días.
Incremento de 2 días cada 5 años posteriores.
Cualquier contrato que indique menos de 12 días en el primer año viola la reforma.

**Art. 80 — Prima vacacional:**
Mínimo 25% sobre el salario durante el período vacacional.
NO puede anticiparse ni prorratearse en el salario mensual.

**Art. 87 — Aguinaldo:**
Mínimo 15 días de salario anuales, pagados ANTES del 20 de diciembre.
NO puede prorratearse mensualmente ni absorberse en el "salario integral".
El trabajador con menos de un año tiene derecho proporcional.

**Art. 110 — Descuentos prohibidos:**
Solo se permiten los descuentos taxativamente establecidos: anticipos de salario, \
errores de pago, pérdidas por descuido del trabajador (con límite del 30% del excedente \
al salario mínimo), cuotas del IMSS, INFONAVIT, y pensiones alimenticias judiciales.
Los descuentos por "faltantes de caja", "diferencias en arqueos", "daños a mercancía" \
o "responsabilidad compartida" son ilegales salvo condiciones muy específicas y con \
autorización de la Junta de Conciliación.

**Art. 56 — Modificación unilateral prohibida:**
El patrón NO puede modificar unilateralmente condiciones esenciales de trabajo \
(salario, jornada, lugar de trabajo, funciones) en perjuicio del trabajador.
Una cláusula que permita al patrón reubicar al trabajador a "cualquier sucursal o ciudad" \
sin su consentimiento puede fundar una rescisión imputable al patrón.

**Arts. 39-A y 39-F — Períodos de prueba:**
Máximo 30 días para trabajadores en general.
Máximo 180 días para trabajadores de dirección, gerenciales o con funciones especializadas.
Un período de prueba que exceda estos límites es ilegal.

**Art. 153-A — Capacitación:**
La capacitación debe impartirse DURANTE la jornada laboral.
Si se imparte fuera de la jornada, el tiempo adicional debe pagarse como tiempo extraordinario.

**Confidencialidad post-laboral:**
La obligación de confidencialidad que se extiende indefinidamente DESPUÉS de terminar la \
relación laboral puede ser abusiva. Debe limitarse razonablemente en tiempo y alcance. \
Plazos mayores a 2 años generalmente son considerados excesivos por los tribunales laborales.

**PTU — Participación en Utilidades (Art. 123 CPEUM y Arts. 117-131 LFT):**
Es un derecho constitucional irrenunciable. Ninguna cláusula puede eximir al patrón de pagar \
PTU ni hacer que el trabajador renuncie a ella, incluso si el salario es "superior al mercado".

## PATRONES DE FRAUDE A LA LEY MÁS FRECUENTES

1. **Salario integral fraudulento:** El contrato dice que el salario mensual "incluye" aguinaldo, \
prima vacacional, PTU y otros. Esto viola el Art. 5 LFT porque estas prestaciones son irrenunciables \
y tienen épocas específicas de pago establecidas en la ley.

2. **Jornada ilimitada disfrazada:** Frases como "la jornada se extenderá según los flujos \
operativos", "el puesto requiere disponibilidad total", "se labora hasta completar la tarea del día". \
Estas cláusulas son nulas (Art. 5 fracc. III LFT) porque establecen jornadas notoriamente excesivas.

3. **Descuento por faltantes de caja disfrazado:** "Responsabilidad compartida en el resguardo de \
activos", "reposición proporcional de faltantes", "descuentos administrativos por diferencias". \
Son ilegales si no cumplen exactamente los requisitos del Art. 110 LFT.

4. **Traslado unilateral:** "El trabajador prestará servicios en el domicilio que el patrón \
determine", "podrá ser reubicado a cualquier sucursal a nivel nacional". Viola el Art. 51 \
fracc. II LFT (causa de rescisión imputable al patrón).

5. **Período de prueba excesivo o renovable:** Períodos de prueba mayores a 30 días para \
trabajadores ordinarios, o cláusulas que permiten renovar el período de prueba indefinidamente.

6. **Capacitación no remunerada:** "La capacitación se realizará en el horario que designe la \
empresa, incluyendo fines de semana". Si es fuera de jornada, debe pagarse.

ANALIZA CADA CLÁUSULA DEL CONTRATO CON RIGOR. Para el puntaje: 0 = contrato impecable que \
cumple y supera la LFT; 100 = contrato extremadamente abusivo con múltiples violaciones graves."""


# ---------------------------------------------------------------------------
# Modelos de datos (salida estructurada)
# ---------------------------------------------------------------------------

class ClausulaAbusiva(BaseModel):
    clausula: str       # Identificador (ej: "Cláusula Séptima B)")
    articulo_lft: str   # Artículo(s) violado(s) (ej: "Art. 5 y Art. 87 LFT")
    severidad: Literal["alta", "media", "baja"]
    explicacion: str


class AnalisisContrato(BaseModel):
    resumen: str                              # 2-3 oraciones sobre el contrato
    clausulas_abusivas: list[ClausulaAbusiva]
    puntaje_abusividad: int                   # 0 = perfecto, 100 = extremadamente abusivo
    veredicto: str                            # Conclusión legal con recomendación


# ---------------------------------------------------------------------------
# Recursos cacheados por Streamlit (se inicializan una sola vez por sesión)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_chroma_client():
    import chromadb
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("❌ ANTHROPIC_API_KEY no encontrada en .env")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Lógica de negocio
# ---------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
    return text.strip()


def retrieve_articles(contract_text: str) -> tuple[list[dict], bool]:
    """Retorna (artículos, rag_disponible)."""
    try:
        client = get_chroma_client()
        client.heartbeat()
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(contract_text[:2000])
        collection = client.get_collection(COLLECTION)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        articles = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return articles, True
    except Exception:
        return [], False


def analyze_contract(contract_text: str, articles: list[dict]) -> AnalisisContrato:
    # Construye el contexto RAG
    if articles:
        blocks = []
        for i, art in enumerate(articles, 1):
            fuente  = art["metadata"].get("fuente", "LFT")
            artculo = art["metadata"].get("articulo", "")
            blocks.append(f"[{fuente}] {artculo}\n{art['text'].strip()}")
        rag_context = "### ARTÍCULOS RELEVANTES RECUPERADOS POR RAG:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n---\n\n"
    else:
        rag_context = ""

    excerpt = contract_text[:MAX_CONTRACT]
    if len(contract_text) > MAX_CONTRACT:
        excerpt += "\n\n[... contrato truncado por longitud ...]"

    user_msg = (
        f"{rag_context}"
        f"### CONTRATO A ANALIZAR:\n\n{excerpt}\n\n"
        "Analiza CADA CLÁUSULA con rigor. "
        "puntaje_abusividad: 0 = contrato perfecto, 100 = extremadamente abusivo."
    )

    client = get_anthropic_client()

    # messages.parse() con salida estructurada Pydantic
    # cache_control en el system prompt para amortizar costo en múltiples análisis
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        output_format=AnalisisContrato,
    )

    # Acumula tokens en session_state para mostrar costo
    u = response.usage
    ss = st.session_state
    ss.tokens_input   = ss.get("tokens_input", 0)   + u.input_tokens
    ss.tokens_output  = ss.get("tokens_output", 0)  + u.output_tokens
    ss.tokens_cached  = ss.get("tokens_cached", 0)  + getattr(u, "cache_read_input_tokens", 0)

    return response.parsed_output


# ---------------------------------------------------------------------------
# Estilos — Legal Noir
# Playfair Display (serif elegante) + paleta oscura/dorada
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Cormorant+Garamond:wght@300;400;600&display=swap');

:root {
    --gold:    #D4AF37;
    --gold2:   #F0CE60;
    --bg:      #0D0D1A;
    --bg2:     #13131F;
    --bg3:     #1C1C2E;
    --border:  rgba(212,175,55,0.25);
    --text:    #E8E0CC;
    --muted:   #7A7090;
    --red:     #A93226;
    --orange:  #C0632A;
    --green:   #1E7A4A;
    --yellow:  #B8860B;
}

/* ── Ocultar chrome de Streamlit ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ── Fondo general ── */
.stApp { background: var(--bg) !important; }
.stApp > div { background: transparent !important; }
section[data-testid="stSidebar"] { background: var(--bg2) !important; }

/* ── Tipografía global ── */
html, body, [class*="css"] {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: var(--text) !important;
}

/* ── Contenedor principal con respiro ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 2px !important;
    transition: border-color .3s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--gold) !important;
}

/* ── Botón primario ── */
.stButton > button[kind="primary"] {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    font-size: .85rem !important;
    border-radius: 1px !important;
    padding: .65rem 1.5rem !important;
    transition: background .25s, color .25s !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--gold) !important;
    color: var(--bg) !important;
}

/* ── Expanders ── */
details {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    margin-bottom: .5rem !important;
}
summary {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    color: var(--text) !important;
    padding: .75rem 1rem !important;
}

/* ── Alertas ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    border-left-width: 3px !important;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: .75rem 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: var(--gold) !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--bg3) !important;
    border-radius: 1px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── Code blocks ── */
code {
    background: var(--bg3) !important;
    color: var(--gold2) !important;
    border: 1px solid var(--border) !important;
    font-size: .85rem !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* ── Input de texto ── */
input, textarea {
    background: var(--bg3) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Componentes de UI
# ---------------------------------------------------------------------------

SEV_COLOR = {"alta": "#A93226", "media": "#C0632A", "baja": "#1E7A4A"}
SEV_LABEL = {"alta": "ALTA",    "media": "MEDIA",   "baja": "BAJA"}


def _score_meta(score: int) -> tuple[str, str]:
    if score <= 20:  return "#1E7A4A", "Contrato Seguro"
    if score <= 50:  return "#B8860B", "Requiere Revisión"
    if score <= 75:  return "#C0632A", "Contrato Problemático"
    return "#A93226", "Contrato Muy Abusivo"


def render_score(score: int) -> None:
    color, label = _score_meta(score)
    pct = score / 100
    # Arco SVG que muestra el porcentaje como gauge semicircular
    dash = pct * 188          # circunferencia del arco = ~188
    st.markdown(f"""
    <div style="text-align:center;padding:32px 0 16px;">
      <svg width="200" height="110" viewBox="0 0 200 110">
        <path d="M20,100 A80,80 0 0,1 180,100"
              fill="none" stroke="#1C1C2E" stroke-width="16" stroke-linecap="round"/>
        <path d="M20,100 A80,80 0 0,1 180,100"
              fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
              stroke-dasharray="{dash:.1f} 188"
              style="transition:stroke-dasharray .8s ease"/>
        <text x="100" y="88" text-anchor="middle"
              font-family="Playfair Display,serif" font-size="42" font-weight="900"
              fill="{color}">{score}</text>
        <text x="100" y="108" text-anchor="middle"
              font-family="Cormorant Garamond,serif" font-size="11"
              fill="#7A7090" letter-spacing="2">ÍNDICE DE ABUSIVIDAD</text>
      </svg>
      <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                  font-style:italic;color:{color};margin-top:4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_clause_card(c) -> None:
    color  = SEV_COLOR[c.severidad]
    label  = SEV_LABEL[c.severidad]
    with st.expander(f"**{c.clausula}**  —  Severidad {label}"):
        st.markdown(
            f"<div style='display:flex;gap:12px;align-items:flex-start;'>"
            f"<span style='background:{color};color:#fff;font-size:.72rem;"
            f"font-family:\"Cormorant Garamond\",serif;letter-spacing:.1em;"
            f"padding:3px 10px;border-radius:1px;white-space:nowrap;flex-shrink:0;'>"
            f"{label}</span>"
            f"<code style='font-size:.9rem;'>{c.articulo_lft}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='margin-top:12px;font-family:\"Cormorant Garamond\",serif;"
            f"font-size:1.05rem;line-height:1.6;color:#E8E0CC;'>{c.explicacion}</p>",
            unsafe_allow_html=True,
        )


def render_veredicto(veredicto: str, score: int) -> None:
    color, _ = _score_meta(score)
    st.markdown(
        f"<div style='border-left:3px solid {color};padding:14px 20px;"
        f"background:#13131F;font-family:\"Cormorant Garamond\",serif;"
        f"font-size:1.1rem;line-height:1.7;color:#E8E0CC;border-radius:0 2px 2px 0;'>"
        f"{veredicto}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Saul · Análisis de Contratos",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Pre-calentar embeddings al abrir la app — evita crashes durante el análisis
    if "models_ready" not in st.session_state:
        with st.spinner("Iniciando sistema de análisis..."):
            get_embeddings()
        st.session_state.models_ready = True
        st.rerun()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:48px 0 32px;">
      <div style="font-family:'Playfair Display',serif;font-size:4rem;font-weight:900;
                  letter-spacing:.18em;color:#D4AF37;line-height:1;">SAUL</div>
      <div style="font-family:'Cormorant Garamond',serif;font-size:1rem;
                  letter-spacing:.35em;color:#7A7090;text-transform:uppercase;
                  margin-top:8px;">Análisis Legal · Contratos Laborales Mexicanos</div>
      <div style="width:60px;height:1px;background:#D4AF37;margin:20px auto 0;opacity:.4;"></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    # ── Columna izquierda ────────────────────────────────────────────────────
    with left:
        st.markdown(
            "<p style='font-family:\"Playfair Display\",serif;font-size:1.1rem;"
            "letter-spacing:.08em;color:#D4AF37;text-transform:uppercase;"
            "margin-bottom:16px;'>Documento</p>",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Arrastra tu contrato aquí — PDF, TXT o MD",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown(
                f"<p style='font-size:.9rem;color:#7A7090;margin:8px 0 16px;'>"
                f"📄 {uploaded.name}</p>",
                unsafe_allow_html=True,
            )
            analyze = st.button("Analizar Contrato", type="primary", use_container_width=True)

            if analyze:
                with st.spinner("Extrayendo texto..."):
                    text = extract_text(uploaded.getvalue(), uploaded.name)

                if not text:
                    st.error("No se pudo extraer texto del documento.")
                    st.stop()

                if len(text) > 50_000:
                    st.warning(
                        f"Documento inusualmente largo ({len(text):,} chars). "
                        f"Solo se analizarán los primeros {MAX_CONTRACT:,} caracteres."
                    )

                with st.spinner("Consultando base de leyes..."):
                    articles, rag_ok = retrieve_articles(text)

                if not rag_ok:
                    st.caption("Base de leyes no disponible — usando conocimiento interno.")

                with st.spinner("Analizando cláusulas..."):
                    try:
                        result = analyze_contract(text, articles)
                        st.session_state.result = result
                    except anthropic.AuthenticationError:
                        st.error("API Key inválida. Verifica ANTHROPIC_API_KEY en .env")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error al analizar: {e}")
                        st.stop()

                st.rerun()

        else:
            st.markdown("""
            <div style="border:1px solid rgba(212,175,55,0.15);padding:28px;
                        border-radius:2px;margin-top:8px;">
              <p style="font-family:'Cormorant Garamond',serif;font-size:1rem;
                        color:#7A7090;line-height:1.8;margin:0;">
                Saul analiza contratos laborales mexicanos e identifica:
                <br><br>
                <span style="color:#D4AF37;">◆</span> Puntaje de abusividad (0–100)<br>
                <span style="color:#D4AF37;">◆</span> Cláusulas que violan la LFT<br>
                <span style="color:#D4AF37;">◆</span> Artículo exacto vulnerado<br>
                <span style="color:#D4AF37;">◆</span> Severidad y explicación legal<br>
                <span style="color:#D4AF37;">◆</span> Veredicto y recomendación
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Costo de sesión (discreto, al fondo)
        ti = st.session_state.get("tokens_input", 0)
        if ti > 0:
            to = st.session_state.get("tokens_output", 0)
            tc = st.session_state.get("tokens_cached", 0)
            costo = (ti - tc) * COST_INPUT + tc * COST_CACHED + to * COST_OUTPUT
            st.markdown(
                f"<p style='font-size:.78rem;color:#3A3050;margin-top:32px;'>"
                f"Sesión · {ti:,} tokens · ${costo:.4f} USD</p>",
                unsafe_allow_html=True,
            )

    # ── Columna derecha: resultados ──────────────────────────────────────────
    with right:
        if "result" not in st.session_state:
            st.markdown("""
            <div style="height:420px;display:flex;align-items:center;
                        justify-content:center;flex-direction:column;gap:16px;">
              <div style="font-size:3rem;opacity:.15;">⚖</div>
              <div style="font-family:'Cormorant Garamond',serif;font-size:.9rem;
                          letter-spacing:.2em;color:#3A3050;text-transform:uppercase;">
                Sube un contrato para comenzar
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        r: AnalisisContrato = st.session_state.result

        render_score(r.puntaje_abusividad)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Resumen
        st.markdown(
            "<p style='font-family:\"Playfair Display\",serif;font-size:1rem;"
            "letter-spacing:.08em;color:#D4AF37;text-transform:uppercase;"
            "margin:16px 0 8px;'>Resumen</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-family:\"Cormorant Garamond\",serif;font-size:1.1rem;"
            f"line-height:1.7;color:#C8C0B0;'>{r.resumen}</p>",
            unsafe_allow_html=True,
        )

        # Veredicto
        st.markdown(
            "<p style='font-family:\"Playfair Display\",serif;font-size:1rem;"
            "letter-spacing:.08em;color:#D4AF37;text-transform:uppercase;"
            "margin:20px 0 8px;'>Veredicto</p>",
            unsafe_allow_html=True,
        )
        render_veredicto(r.veredicto, r.puntaje_abusividad)

        # Cláusulas
        n = len(r.clausulas_abusivas)
        st.markdown(
            f"<p style='font-family:\"Playfair Display\",serif;font-size:1rem;"
            f"letter-spacing:.08em;color:#D4AF37;text-transform:uppercase;"
            f"margin:24px 0 12px;'>Cláusulas Detectadas"
            f"<span style='font-size:.8rem;font-weight:400;margin-left:12px;"
            f"color:#7A7090;'>({n})</span></p>",
            unsafe_allow_html=True,
        )

        if not r.clausulas_abusivas:
            st.markdown(
                "<p style='color:#1E7A4A;font-family:\"Cormorant Garamond\",serif;"
                "font-size:1.05rem;'>✓ No se detectaron cláusulas abusivas.</p>",
                unsafe_allow_html=True,
            )
            return

        orden = {"alta": 0, "media": 1, "baja": 2}
        for c in sorted(r.clausulas_abusivas, key=lambda x: orden[x.severidad]):
            render_clause_card(c)

        # Totales por severidad
        alta  = sum(1 for c in r.clausulas_abusivas if c.severidad == "alta")
        media = sum(1 for c in r.clausulas_abusivas if c.severidad == "media")
        baja  = sum(1 for c in r.clausulas_abusivas if c.severidad == "baja")
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Severidad Alta",  alta)
        c2.metric("Severidad Media", media)
        c3.metric("Severidad Baja",  baja)


if __name__ == "__main__":
    main()
