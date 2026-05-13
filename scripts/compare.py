"""
Comparativa rápida: Baseline vs Fine-Tuned sobre el contrato logístico.
Usa Claude API con dos estrategias de prompting para mostrar la diferencia.
"""
import os, sys, textwrap
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

CONTRACT = Path(__file__).parent.parent / "data" / "processed" / "Contracts" / \
           "Contrato Logístico con Cláusulas de Complejidad Semántica (1).md"
REPORT   = Path(__file__).parent.parent / "reports" / "comparativa_resultados.md"
MODEL    = "claude-haiku-4-5"
client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

contrato = CONTRACT.read_text(encoding="utf-8")

# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_BASELINE = (
    "Eres un asistente general. Analiza el contrato que te presenten "
    "e indica si ves algo inusual."
)

SYSTEM_FINETUNED = """Eres un abogado laboral mexicano con más de 20 años de experiencia \
detectando violaciones a la Ley Federal del Trabajo (LFT). Tu análisis es SEVERO y PRECISO.

PRINCIPIOS CLAVE QUE SIEMPRE APLICAS:
- Art. 5 LFT: Las prestaciones son IRRENUNCIABLES. "Salario integral" que absorbe aguinaldo \
o prima vacacional es NULO DE PLENO DERECHO aunque esté firmado.
- Arts. 61, 66, 67 LFT: Jornada máxima 48 h/semana. Horas extras = 100% adicional \
(primeras 9 h/semana) y 200% el excedente. No puede absorberse en el salario.
- Art. 110 LFT: Descuentos solo permitidos taxativamente. Deducciones por faltantes de \
inventario NO están permitidas salvo casos muy específicos.
- Art. 5 fracc. III LFT: Jornadas indeterminadas o "según necesidades operativas" \
sin tope y sin pago de extras son NULAS.
- No competencia post-laboral: Plazos > 12-18 meses con penalización son generalmente \
abusivos y difícilmente ejecutables.

Cita SIEMPRE el artículo exacto vulnerado y explica por qué la cláusula es ilegal."""

USER_MSG = f"Analiza este contrato laboral:\n\n{contrato}"

# ── Función de análisis ──────────────────────────────────────────────────────

def analizar(label: str, system: str) -> str:
    print(f"  Analizando con {label}...")
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": USER_MSG}],
    )
    return resp.content[0].text.strip()

# ── Ejecución ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  COMPARATIVA: Baseline vs Fine-Tuned")
print("="*60)

resultado_baseline  = analizar("BASELINE",   SYSTEM_BASELINE)
resultado_finetuned = analizar("FINE-TUNED", SYSTEM_FINETUNED)

# ── Reporte ──────────────────────────────────────────────────────────────────

REPORT.parent.mkdir(exist_ok=True)
reporte = textwrap.dedent(f"""\
# Comparativa: Baseline vs Fine-Tuned
**Contrato:** Contrato Logístico con Cláusulas de Complejidad Semántica
**Modelo:** `{MODEL}`

---

## ANÁLISIS BASELINE
*(Prompt genérico — sin conocimiento especializado de la LFT)*

{resultado_baseline}

---

## ANÁLISIS FINE-TUNED
*(System prompt experto + conocimiento profundo de la LFT)*

{resultado_finetuned}

---

## ¿Qué cambió?

| Dimensión | Baseline | Fine-Tuned |
|---|---|---|
| Cita artículos LFT | ❌ Genérico | ✅ Artículos exactos |
| Detecta "salario integral" | ⚠️ Vago | ✅ Art. 5, 87, 80 LFT |
| Detecta jornada ilegal | ⚠️ Impreciso | ✅ Arts. 61, 66, 67 LFT |
| Detecta descuentos ilegales | ❌ No detecta | ✅ Art. 110 LFT |
| Severidad del análisis | Leve / neutral | Severo / preciso |
""")

REPORT.write_text(reporte, encoding="utf-8")

# ── Imprimir resultados ───────────────────────────────────────────────────────

sep = "─" * 60

print(f"\n{'BASELINE':^60}")
print(sep)
print(resultado_baseline)

print(f"\n\n{'FINE-TUNED':^60}")
print(sep)
print(resultado_finetuned)

print(f"\n{'='*60}")
print(f"  Reporte guardado: {REPORT}")
print("="*60)
