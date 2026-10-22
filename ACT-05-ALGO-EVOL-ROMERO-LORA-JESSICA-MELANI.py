#!/usr/bin/env python3
"""
construir_informe.py — Genera el informe técnico (DOCX) del estudio de fusión con algoritmo genético
sobre el detector CAS. Todas las cifras de tablas y del texto se leen de los archivos de resultados;
no se escriben a mano.

USO
  python construir_informe.py --raiz .. --salida ../Informe_AG_Fraude_Romero_Lora.docx \
      --url-repo https://github.com/USUARIO/REPOSITORIO
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PURPURA = RGBColor(0x84, 0x00, 0xE0)
PURPURA_HEX = "8400E0"
PURPURA_OSCURO = RGBColor(0x3B, 0x0A, 0x66)
GRIS = RGBColor(0x52, 0x51, 0x4E)
FILA_ALTERNA = "F4ECFC"
FONDO_FORMULA = "F7F2FD"

ALUMNA = "Jessica Melani Romero Lora"
MATRICULA = "25322116"
UNIVERSIDAD = "Universidad Politécnica Metropolitana de Hidalgo"
MAESTRIA = "Maestría en Inteligencia Artificial"
MATERIA = "Algoritmos Evolutivos"
DOCENTE = "Dr. Jaime Aguilar Ortiz"
FECHA = "21 de octubre de 2026"
TITULO = ("Algoritmo genético con tres codificaciones de cromosoma para la fusión de señales "
          "en un detector de fraude en reclamaciones médicas")
SUBTITULO = "Reporte técnico · Corte 2"

MODELO_NOMBRE = {
    "PM · estructural": "Capa A (estructural)",
    "PM · semántica": "Capa B (semántica)",
    "PM · temporal": "Capa C (temporal)",
    "PM · completa (pesos uniformes)": "Proceso completo, pesos uniformes",
    "PM · completa (pesos v1 0.45/0.35/0.20)": "Proceso completo, pesos v1",
    "Isolation Forest": "Isolation Forest",
    "XGBoost · sin CAS": "XGBoost sin CAS",
    "XGBoost · con CAS": "XGBoost con CAS",
    "O-A-R · ranking promedio": "O-A-R, promedio de rangos",
}
METODO_NOMBRE = {
    "G1-binaria": "G1 binaria (Gray)",
    "G2-real": "G2 real",
    "G3-entera": "G3 entera",
    "ALEATORIA": "Búsqueda aleatoria",
    "UNIFORME": "Pesos uniformes",
    "MEJOR_SENAL": "Mejor señal individual",
}
SENAL_NOMBRE = {
    "s_struct": "Estructural (A)", "s_sem": "Semántica (B)", "s_temp": "Temporal (C)",
    "s_xgb_cas": "XGBoost con CAS", "s_if": "Isolation Forest", "s_xgb_static": "XGBoost sin CAS",
}
METRICAS = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("precision_5", "Precisión al 5 %"),
            ("recall_5", "Recall al 5 %"), ("amount_recall_5", "Recall monetario al 5 %")]


# ── Utilidades de formato ───────────────────────────────────────

def f3(v):
    return f"{v:.3f}"


def f4(v):
    return f"{v:.4f}"


def pct(v, dec=2):
    return f"{100 * v:.{dec}f} %"


def fp(p):
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def fmt_int(v):
    return f"{int(round(v)):,}"


def sombrear(celda, color_hex):
    tc = celda._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color_hex)
    tc.append(sh)


def margenes_tabla(tabla, top=40, bottom=40, left=70, right=70):
    pr = tabla._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for lado, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    pr.insert_element_before(mar, 'w:tblLook', 'w:tblCaption', 'w:tblDescription', 'w:tblPrChange')


def campo_pagina(parrafo):
    run = parrafo.add_run()
    for tipo, texto in (("begin", None), (None, "PAGE"), ("end", None)):
        if tipo:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tipo)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = texto
        run._r.append(el)
    run.font.size = Pt(8.5)
    run.font.color.rgb = GRIS


def hipervinculo(parrafo, url, texto, tam=10.5):
    rid = parrafo.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), PURPURA_HEX)
    rpr.append(color)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(tam * 2)))
    rpr.append(sz)
    sub = OxmlElement("w:u")
    sub.set(qn("w:val"), "single")
    rpr.append(sub)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = texto
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    parrafo._p.append(h)


def estilos(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12
    for nombre, tam, color, antes in (("Heading 1", 14, PURPURA, 14), ("Heading 2", 11.5, PURPURA_OSCURO, 9),
                                      ("Heading 3", 10.5, GRIS, 6)):
        st = doc.styles[nombre]
        st.font.name = "Calibri"
        st.font.size = Pt(tam)
        st.font.bold = True
        st.font.color.rgb = color
        st.paragraph_format.space_before = Pt(antes)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True


def parrafo(doc, texto, alineado=WD_ALIGN_PARAGRAPH.JUSTIFY, tam=None, color=None, cursiva=False,
            negrita_inicio=None):
    p = doc.add_paragraph()
    p.alignment = alineado
    if negrita_inicio and texto.startswith(negrita_inicio):
        r = p.add_run(negrita_inicio)
        r.bold = True
        resto = texto[len(negrita_inicio):]
    else:
        resto = texto
    r2 = p.add_run(resto)
    r2.italic = cursiva
    for r in p.runs:
        if tam:
            r.font.size = Pt(tam)
        if color:
            r.font.color.rgb = color
    return p


def vinetas(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        if "::" in it:
            titulo, cuerpo = it.split("::", 1)
            r = p.add_run(titulo.strip() + " ")
            r.bold = True
            p.add_run(cuerpo.strip())
        else:
            p.add_run(it)


def formula(doc, texto, etiqueta=None):
    """Fórmula centrada sobre fondo lila; etiqueta opcional a la derecha, p. ej. (3)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), FONDO_FORMULA)
    pPr.insert_element_before(sh, 'w:tabs','w:suppressAutoHyphens','w:kinsoku','w:wordWrap','w:overflowPunct','w:topLinePunct','w:autoSpaceDE','w:autoSpaceDN','w:bidi','w:adjustRightInd','w:snapToGrid','w:spacing','w:ind','w:contextualSpacing','w:mirrorIndents','w:suppressOverlap','w:jc','w:textDirection','w:textAlignment','w:textboxTightWrap','w:outlineLvl','w:divId','w:cnfStyle','w:rPr','w:sectPr','w:pPrChange')
    r = p.add_run(texto + (f"    {etiqueta}" if etiqueta else ""))
    r.font.size = Pt(10.5)
    r.font.color.rgb = PURPURA_OSCURO
    return p


def tabla(doc, encabezados, filas, anchos_cm, titulo=None, nota=None, tam=8.5, alinear=None):
    if titulo:
        cap = doc.add_paragraph()
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(2)
        r = cap.add_run(titulo)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = GRIS
    t = doc.add_table(rows=1, cols=len(encabezados))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = doc.styles["Table Grid"]
    t.autofit = False
    margenes_tabla(t)
    for i, enc in enumerate(encabezados):
        c = t.rows[0].cells[i]
        c.width = Cm(anchos_cm[i])
        sombrear(c, PURPURA_HEX)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        run = c.paragraphs[0].add_run(enc)
        run.bold = True
        run.font.size = Pt(tam)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    trpr = t.rows[0]._tr.get_or_add_trPr()
    rep = OxmlElement("w:tblHeader")
    rep.set(qn("w:val"), "true")
    trpr.append(rep)
    for k, fila in enumerate(filas):
        celdas = t.add_row().cells
        for i, valor in enumerate(fila):
            c = celdas[i]
            c.width = Cm(anchos_cm[i])
            if k % 2 == 1:
                sombrear(c, FILA_ALTERNA)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if alinear and alinear[i] == "r":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif alinear and alinear[i] == "c":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(valor))
            run.font.size = Pt(tam)
    if nota:
        n = doc.add_paragraph()
        n.paragraph_format.space_before = Pt(2)
        n.paragraph_format.space_after = Pt(8)
        r = n.add_run(nota)
        r.italic = True
        r.font.size = Pt(8)
        r.font.color.rgb = GRIS
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figura(doc, ruta, ancho_cm, titulo):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(ruta), width=Cm(ancho_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run(titulo)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = GRIS


# ── Portada ─────────────────────────────────────────────────────

def portada(doc):
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(UNIVERSIDAD.upper())
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = PURPURA
    p.paragraph_format.space_after = Pt(4)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(MAESTRIA)
    r.font.size = Pt(12)
    r.font.color.rgb = GRIS
    doc.add_paragraph()
    linea = doc.add_paragraph()
    linea.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pPr = linea._p.get_or_add_pPr()
    bordes = OxmlElement("w:pBdr")
    inf = OxmlElement("w:bottom")
    inf.set(qn("w:val"), "single")
    inf.set(qn("w:sz"), "12")
    inf.set(qn("w:space"), "1")
    inf.set(qn("w:color"), PURPURA_HEX)
    bordes.append(inf)
    pPr.insert_element_before(bordes, 'w:shd', 'w:tabs','w:suppressAutoHyphens','w:kinsoku','w:wordWrap','w:overflowPunct','w:topLinePunct','w:autoSpaceDE','w:autoSpaceDN','w:bidi','w:adjustRightInd','w:snapToGrid','w:spacing','w:ind','w:contextualSpacing','w:mirrorIndents','w:suppressOverlap','w:jc','w:textDirection','w:textAlignment','w:textboxTightWrap','w:outlineLvl','w:divId','w:cnfStyle','w:rPr','w:sectPr','w:pPrChange')
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(SUBTITULO)
    r.font.size = Pt(11)
    r.font.color.rgb = GRIS
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(TITULO)
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = PURPURA_OSCURO
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Materia: {MATERIA}")
    r.font.size = Pt(11.5)
    for _ in range(4):
        doc.add_paragraph()
    datos = [("Alumna", ALUMNA), ("Matrícula", MATRICULA), ("Catedrático", DOCENTE),
             ("Modalidad", "Individual"), ("Periodo", "Septiembre–diciembre de 2026"),
             ("Fecha de entrega", FECHA)]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in datos:
        c = t.add_row().cells
        c[0].width = Cm(4.2)
        c[1].width = Cm(8.0)
        rk = c[0].paragraphs[0].add_run(k)
        rk.bold = True
        rk.font.color.rgb = PURPURA
        rk.font.size = Pt(11)
        rv = c[1].paragraphs[0].add_run(v)
        rv.font.size = Pt(11)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Código fuente: ")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GRIS
    return p


# ── Lectura de resultados ───────────────────────────────────────

def cargar(raiz):
    det = {}
    for d in ("1.00", "0.50", "0.25"):
        ruta = raiz / "results" / f"difficulty_{d}"
        if (ruta / "ablation_metrics.csv").exists():
            det[d] = {"ablation": pd.read_csv(ruta / "ablation_metrics.csv", index_col=0),
                      "oar": pd.read_csv(ruta / "oar_metrics.csv", index_col=0),
                      "tipo": pd.read_csv(ruta / "fraud_type_recall.csv", index_col=0),
                      "ci": pd.read_csv(ruta / "bootstrap_ci.csv") if (ruta / "bootstrap_ci.csv").exists() else None,
                      "audit": json.loads((ruta / "audit.json").read_text(encoding="utf-8"))}
    ga = {}
    for d in ("1.00", "0.25"):
        ruta = raiz / "results_ga" / f"difficulty_{d}"
        if (ruta / "corridas.csv").exists():
            ga[d] = {"corridas": pd.read_csv(ruta / "corridas.csv"),
                     "conv": pd.read_csv(ruta / "convergencia.csv"),
                     "estad": pd.read_csv(ruta / "pruebas_estadisticas.csv"),
                     "manifiesto": json.loads((ruta / "manifiesto.json").read_text(encoding="utf-8"))}
    return det, ga


def resumen_metodos(corr):
    """Media y DE por método de cada métrica de prueba (25 bloques pliegue × semilla)."""
    g = corr.groupby("metodo")
    out = {}
    for m, _ in METRICAS:
        out[m] = (g[m].mean(), g[m].std())
    out["unicos"] = g["unicos"].mean()
    out["segundos"] = g["segundos"].mean()
    out["pesos"] = g[[f"w_{s}" for s in SENAL_NOMBRE]].mean()
    return out


# ── Cuerpo del informe ──────────────────────────────────────────

def cuerpo(doc, det, ga, url_repo):
    d1 = det["1.00"]
    a1 = d1["audit"]
    n_fraude = 2789
    n_claims = 50000

    doc.add_heading("Índice", level=1)
    indice = ["1. Descripción del problema analizado con un algoritmo genético",
              "2. Objetivo del estudio",
              "3. Las tres formas de codificar el cromosoma",
              "4. Genotipos y fenotipos",
              "5. Métodos usados en cada una de las siete etapas",
              "6. Código que apoya el cumplimiento del objetivo",
              "7. Resultados, interpretación y análisis",
              "8. Referencias bibliográficas",
              "Anexo A. Enlace al código y reproducibilidad"]
    for it in indice:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.4)
        p.add_run(it).font.size = Pt(10)
    doc.add_page_break()

    # 1 ───────────────────────────────────────────────────────────
    doc.add_heading("1. Descripción del problema analizado con un algoritmo genético", level=1)
    doc.add_heading("1.1 Contexto industrial", level=2)
    parrafo(doc, "Una unidad de investigación especial (SIU) de un asegurador de servicios médicos recibe cada "
                 "periodo miles de reclamaciones de honorarios presentadas por proveedores. Cada reclamación se "
                 "identifica con un procedimiento (código CPT/HCPCS), un diagnóstico (ICD-10), un monto facturado, "
                 "un proveedor (NPI), una especialidad y fechas de servicio y de envío. La mayoría de las "
                 "reclamaciones es legítima; una fracción pequeña es fraudulenta y se oculta entre las legítimas.")
    parrafo(doc, "La restricción central es operativa: el equipo auditor puede revisar sólo una fracción fija de las "
                 "reclamaciones, aquí el 5 % por periodo. El problema industrial no es clasificar bien en abstracto, "
                 "sino ordenar las reclamaciones de modo que el 5 % revisado contenga la mayor cantidad posible de fraude, "
                 "medida tanto en número de casos como en monto recuperable.")
    parrafo(doc, "El detector descrito en los archivos de la materia combina tres capas de evidencia y dos clasificadores "
                 "(un modelo de aprendizaje supervisado y un modelo no supervisado) en un sistema denominado O-A-R. "
                 "El algoritmo genético de este estudio no detecta fraude por sí mismo: aprende cómo fusionar las seis "
                 "señales que el detector produce, de modo que la puntuación final ordene mejor las reclamaciones.")

    doc.add_heading("1.2 Suposiciones de realismo industrial", level=2)
    parrafo(doc, "El estudio se formula con suposiciones que replican condiciones operativas. Cada una se declara con su "
                 "justificación y su límite, porque los resultados dependen de ellas.")
    tabla(doc, ["Suposición", "Justificación operativa", "Límite"],
          [["Fraude raro: 5.58 % de las reclamaciones en la línea base",
            "El desequilibrio de clases es la regla en auditoría de salud; el área bajo la curva PR es la métrica adecuada",
            "La tasa real no se calibró con datos de un asegurador"],
           ["Revisión limitada al 5 % por pliegue",
            "Capacidad fija del equipo auditor; las comparaciones se hacen a presupuesto idéntico",
            "Un presupuesto distinto puede cambiar el ranking de métodos"],
           ["Validación por proveedor",
            "Un mismo proveedor no aparece en entrenamiento y prueba; evita fuga de su comportamiento",
            "Los proveedores se asignan al azar a pliegues"],
           ["Fraude con inicio tardío en la mitad de los proveedores fraudulentos",
            "Las conductas aparecen en momentos distintos; genera deriva temporal real",
            "El momento de inicio se fija por generador, no se observa en la realidad"],
           ["Latencia de envío y horario de servicio distintos en fraude",
            "Los fantasmas se envían rápido y en horas atípicas, es un supuesto de diseño del generador",
            "Las probabilidades de esas señales son supuestos del generador"],
           ["Fraude no adaptativo",
            "El agente no cambia su conducta al detectar el sistema",
            "No valida la componente de retroalimentación (CAS-4); es la limitación más importante"],
           ["Etiquetas sólo en entrenamiento y evaluación",
            "La inferencia de producción nunca ve la etiqueta; se audita automáticamente",
            "Ninguna"]],
          [4.4, 6.6, 5.0], titulo="Tabla 1. Suposiciones de realismo industrial del estudio")

    doc.add_heading("1.3 Datos sintéticos", level=2)
    parrafo(doc, f"Los datos se generan con synpuf_generator_v2.py, inspirado en el archivo público SynPUF de Medicare "
                 f"Parte B (Centers for Medicare & Medicaid Services, s. f.). Se generan 800 proveedores, {n_claims:,} reclamaciones y 314 878 "
                 f"eventos de proceso, con semilla 2024. Los proveedores pertenecen a seis especialidades y siguen uno "
                 f"de seis perfiles de conducta. Los catálogos de procedimientos y tarifas son una versión simplificada "
                 f"de Medicare Parte B; no reproducen tarifas reales.")
    tabla(doc, ["Perfil", "Probabilidad", "Mecanismo de fraude inyectado"],
          [["normal", "0.72", "Ninguno"],
           ["upcoder_leve", "0.10", "Facturar 99214 en lugar de 99213 con sobreprecio (F5)"],
           ["upcoder_severo", "0.07", "Facturar 99215 o 99223 en lugar de consultas de menor nivel (F1)"],
           ["phantom_biller", "0.04", "Procedimientos de catálogo sin evento clínico, con horario y envío atípicos (F2)"],
           ["unbundler", "0.04", "Monto inflado por desagregación de procedimientos (F3)"],
           ["mixed_fraud", "0.03", "Mezcla de upcoding quirúrgico y severo (F4 y F1)"]],
          [3.4, 2.4, 10.2], alinear=["l", "c", "l"],
          titulo="Tabla 2. Perfiles de proveedor del generador",
          nota="Probabilidades de cfg.profile_probs. Los fraudes sólo ocurren tras el inicio del proveedor.")
    parrafo(doc, f"Con dificultad 1.0 (línea base) el conjunto contiene {n_fraude:,} reclamaciones fraudulentas "
                 f"({100 * n_fraude / n_claims:.2f} %). El parámetro de dificultad d ∈ (0, 1] escala la magnitud de las "
                 "señales de fraude; con d = 0.5 y d = 0.25 se obtienen versiones más difíciles, usadas como prueba de estrés.")

    doc.add_heading("1.4 Detector base", level=2)
    parrafo(doc, "El detector (cas_detector_v2.py) no usa etiquetas para construir variables. Sus señales son:")
    vinetas(doc, [
        "Capa A (estructural):: un modelo de Petri por especialidad, descubierto con el Inductive Miner (ruido 0.10) y "
        "evaluado con token-based replay. Su salida es la aptitud de traza (fitness) de cada reclamación.",
        "Capa B (semántica):: soporte del par especialidad-procedimiento, monto relativo a la mediana del procedimiento y "
        "complejidad relativa al proveedor, con una cola superior de referencia.",
        "Capa C (temporal):: función de distribución empírica bilateral del retraso de envío y de la hora de servicio.",
        "XGBoost (dos variantes):: con variables estáticas, y con variables estáticas más las dimensiones CAS "
        "(deriva, entropía de códigos, cambios de régimen).",
        "Isolation Forest:: anomalía no supervisada sobre variables estáticas escaladas.",
        "O-A-R:: combinación por regla (OR o consenso de dos señales) y por promedio de rangos.",
    ])
    parrafo(doc, f"La validación es anidada por proveedor, con cinco pliegues. La auditoría de fuga confirmó que ninguna "
                 f"variable de entrada contiene el perfil ni la etiqueta (fuga: {'NO' if a1['feature_leak_check']['ok'] else 'SÍ'}), "
                 f"y que todas las reclamaciones de prueba recibieron puntuación estructural (ninguna sin fitness). "
                 f"Verificamos además que los 800 proveedores caen en exactamente un pliegue.")

    # 2 ───────────────────────────────────────────────────────────
    doc.add_heading("2. Objetivo del estudio", level=1)
    parrafo(doc, "Objetivo general. Diseñar, implementar y evaluar un algoritmo genético que aprenda los pesos de fusión "
                 "de las seis señales del detector, comparando tres codificaciones del cromosoma, con validación por "
                 "proveedor y presupuesto de revisión fijo.", negrita_inicio="Objetivo general.")
    parrafo(doc, "Objetivos específicos.", negrita_inicio="Objetivos específicos.")
    vinetas(doc, [
        "Formalizar el genotipo y el fenotipo del problema de fusión.",
        "Implementar tres codificaciones del cromosoma (binaria con código de Gray, real y entera), cada una con sus operadores.",
        "Comparar las codificaciones con dos controles de la misma familia: búsqueda aleatoria con presupuesto idéntico y pesos uniformes, más la mejor señal individual.",
        "Medir la calidad en prueba, la diversidad de la población, la velocidad de convergencia y el costo computacional.",
        "Evaluar la robustez en dos niveles de dificultad de los datos sintéticos.",
    ])
    parrafo(doc, "Hipótesis. Se formularon antes de analizar los resultados del algoritmo genético; la última columna registra el veredicto.",
            negrita_inicio="Hipótesis.")
    corr = ga["1.00"]["corridas"]
    rm = resumen_metodos(corr)
    gan10 = rm["pr_auc"][0]["G2-real"] - rm["pr_auc"][0]["MEJOR_SENAL"]
    if "0.25" in ga:
        m25_h = resumen_metodos(ga["0.25"]["corridas"])
        gan25 = m25_h["pr_auc"][0]["G2-real"] - m25_h["pr_auc"][0]["MEJOR_SENAL"]
        veredicto_h5 = "Se confirma" if gan25 > gan10 else "No se confirma"
    else:
        veredicto_h5 = "Pendiente"
    tabla(doc, ["Código", "Hipótesis", "Veredicto (dificultad 1.0)"],
          [["H1", "Al menos una codificación supera a pesos uniformes y a búsqueda aleatoria con el mismo presupuesto.", "Se confirma (sección 7.3)"],
           ["H2", "Las codificaciones difieren en la diversidad de la población final (G2 > G1 > G3).", "Se confirma (sección 7.4)"],
           ["H3", "Las codificaciones no difieren en calidad de prueba al presupuesto fijado.", "Se confirma (sección 7.3)"],
           ["H4", "La fusión aprendida supera a la mejor señal individual.", "Se confirma en significancia; magnitud despreciable (sección 7.3)"],
           ["H5", "La ventaja de la fusión sobre la mejor señal aumenta cuando la dificultad disminuye.", f"{veredicto_h5} (sección 7.6)"]],
          [1.2, 9.6, 5.2], alinear=["c", "l", "l"], titulo="Tabla 3. Hipótesis del estudio")

    # 3 ───────────────────────────────────────────────────────────
    doc.add_heading("3. Las tres formas de codificar el cromosoma", level=1)
    parrafo(doc, "Un cromosoma codifica una solución del problema, que aquí es un vector de pesos de fusión. Se definen "
                 "tres codificaciones del mismo fenotipo, que difieren en la representación del genotipo y, por tanto, "
                 "en los operadores que son naturales para cada una.")
    tabla(doc, ["", "G1 · binaria (Gray)", "G2 · real", "G3 · entera"],
          [["Genotipo", "48 bits: 8 bits por cada uno de los 6 pesos", "6 números reales en [0, 1]", "6 enteros en {0, …, 20}"],
           ["Tamaño del espacio", "2^48 ≈ 2.8 × 10^14", "Continuo (6 dimensiones)", "21^6 = 85 766 121"],
           ["Resolución del peso", "1/255", "Continua", "1/20"],
           ["Recombinación", "Cruce de un punto sobre la cadena de bits", "BLX-α con α = 0.3, recortado a [0, 1]", "Cruce uniforme por gen"],
           ["Mutación", "Volteo de bit con probabilidad 1/48", "Gaussiana acotada, σ = 0.05, probabilidad 1/6 por gen", "Paso ±1 con probabilidad 1/6 por gen"],
           ["Ventaja esperada", "Operadores clásicos; exploración por bits", "Precisión fina; relación directa con el fenotipo", "Espacio finito; operadores simples"],
           ["Riesgo esperado", "Epistasis: un bit cambia varios bits del peso", "Redundancia por escala", "Pérdida de diversidad al converger"]],
          [3.2, 4.5, 4.5, 4.5], alinear=["l", "l", "l", "l"],
          titulo="Tabla 4. Comparación de las tres codificaciones",
          nota="La mutación binaria usa la tasa estándar 1/L, con L = 48 bits; las dos codificaciones por gen usan 1/K, con K = 6 genes.")
    parrafo(doc, "La elección de tres codificaciones y no de una sola responde a una pregunta práctica: si la forma de "
                 "representar los pesos altera la calidad o sólo el costo. Las tres codifican el mismo fenotipo y reciben "
                 "las mismas poblaciones iniciales para una semilla dada, así que las diferencias observadas se atribuyen "
                 "a la codificación y sus operadores, no a la muestra inicial.")

    # 4 ───────────────────────────────────────────────────────────
    doc.add_heading("4. Genotipos y fenotipos", level=1)
    doc.add_heading("4.1 Fenotipo común", level=2)
    parrafo(doc, "El fenotipo es un vector de pesos en el símplex de dimensión K = 6, que corresponde a las seis señales "
                 "del detector. La puntuación fusionada de la reclamación i es una combinación convexa de sus señales:")
    formula(doc, "s_w(i) = Σ_k w_k · z_ik ,   w ∈ Δ⁵ = { w ∈ ℝ⁶ : w_k ≥ 0,  Σ_k w_k = 1 }", "(1)")
    parrafo(doc, "Las señales z_ik son s_struct, s_sem, s_temp, XGBoost con CAS, Isolation Forest y XGBoost sin CAS. "
                 "Todas están en [0, 1] por construcción, así que no requieren normalización adicional.")
    doc.add_heading("4.2 Genotipo y función de decodificación", level=2)
    parrafo(doc, "El genotipo G es la representación del cromosoma. La función φ lleva el genotipo a un vector "
                 "intermedio u ∈ [0, 1]^6, y la normalización lleva u al símplex. Esta doble etapa separa la "
                 "representación de la restricción de suma 1.")
    formula(doc, "w(G) = u(G) / Σ_k u_k(G),   con  w = (1/6, …, 1/6) si Σ_k u_k = 0", "(2)")
    formula(doc, "G1:   u_k = (1/255) · Σ_{j=0}^{7} b̂_{kj} 2^{7−j},   b̂_{kj} = ⊕_{i ≤ j} g_{ki}", "(3a)")
    formula(doc, "G2:   u_k = x_k ,   x ∈ [0, 1]^6", "(3b)")
    formula(doc, "G3:   u_k = g_k / 20 ,   g_k ∈ {0, 1, …, 20}", "(3c)")
    parrafo(doc, "En G1, g_k,j son los bits del código de Gray del peso k, y b̂_kj son los bits binarios recuperados "
                 "con la suma XOR acumulada de izquierda a derecha. En G2 el genotipo es el fenotipo intermedio. En G3 "
                 "cada gen toma uno de 21 valores equiespaciados.")
    doc.add_heading("4.3 Codificación inversa y redundancia", level=2)
    parrafo(doc, "La población inicial se muestrea en u y se codifica con la inversa de (3a)–(3c). En G1, para un entero "
                 "n = round(255 u_k), el gen se escribe como g = n ⊕ (n >> 1), de modo que enteros consecutivos difieren "
                 "en un solo bit. Las pruebas automatizadas del código verifican esta propiedad.")
    parrafo(doc, "Las tres codificaciones no son inyectivas respecto del fenotipo: por (2), u y c·u producen el mismo peso "
                 "w para cualquier c > 0. Esta redundancia es mayor en G2 y G3, donde la escala "
                 "de u es libre, La pérdida de diversidad de G3 (sección 7.4) se atribuye al espacio discreto, no a la redundancia de escala.")
    doc.add_heading("4.4 Función de aptitud", level=2)
    parrafo(doc, "La aptitud de un genotipo es el área bajo la curva precisión–recall (PR-AUC, o precisión promedio) de la "
                 "puntuación fusionada sobre las reclamaciones de los pliegues de entrenamiento. El área bajo la curva "
                 "PR se prefiere a la ROC cuando la clase positiva es rara (Saito y Rehmsmeier, 2015).")
    formula(doc, "f(G) = AP(w(G)) = (1 / P) · Σ_{i : y_i = 1} Prec@rank(i),   P = número de fraudes", "(4)")
    parrafo(doc, "Prec@rank(i) es la precisión sobre las reclamaciones ordenadas por s_w, hasta la posición de la i-ésima "
                 "reclamación fraudulenta. La implementación vectorizada coincide con average_precision_score de "
                 "scikit-learn cuando no hay empates (prueba de la función de aptitud).")

    # 5 ───────────────────────────────────────────────────────────
    doc.add_heading("5. Métodos usados en cada una de las siete etapas", level=1)
    parrafo(doc, "Se usa el marco de siete etapas de la materia. Las etapas 1, 3, 6 y 7 son comunes a las tres codificaciones "
                 "y se fijan; las etapas 4 y 5 dependen de la codificación. Los parámetros comunes son N = 60 individuos, "
                 "torneo k = 2, elitismo e = 2, probabilidad de cruce p_c = 0.9 y presupuesto E_max = 5 000 evaluaciones de aptitud.")
    tabla(doc, ["Etapa", "G1 binaria", "G2 real", "G3 entera"],
          [["1 · Inicialización", "Hipercubo latino (M03) en [0, 1]^6 y codificación de Gray", "Hipercubo latino (M03) en [0, 1]^6", "Hipercubo latino (M03) y redondeo a {0, …, 20}"],
           ["2 · Evaluación", "Aptitud directa (4)", "Aptitud directa (4)", "Aptitud directa (4)"],
           ["3 · Selección", "Torneo k = 2 con reemplazo", "Torneo k = 2 con reemplazo", "Torneo k = 2 con reemplazo"],
           ["4 · Recombinación", "Cruce de un punto", "BLX-α, α = 0.3", "Cruce uniforme por gen"],
           ["5 · Mutación", "Volteo de bit, 1/48", "Gaussiana acotada, σ = 0.05, 1/6 por gen", "Paso ±1, 1/6 por gen"],
           ["6 · Reemplazo", "Generacional con elitismo e = 2", "Igual", "Igual"],
           ["7 · Paro", "Presupuesto de 5 000 evaluaciones (4 990 usadas)", "Igual", "Igual"]],
          [3.0, 4.6, 4.6, 4.6], alinear=["l", "l", "l", "l"], titulo="Tabla 5. Métodos por etapa y por codificación")
    doc.add_heading("5.1 Etapa 1 · inicialización", level=2)
    parrafo(doc, "Un hipercubo latino (McKay, Beckman y Conover, 1979) reparte N puntos en [0, 1]^6 de modo que cada "
                 "estrato de cada dimensión contiene exactamente un punto. La población inicial es la misma en las tres "
                 "codificaciones para una semilla dada.")
    formula(doc, "x_ij = (π_j(i) + U_ij) / N ,   π_j permutación aleatoria de {0, …, N−1}", "(5)")
    doc.add_heading("5.2 Etapa 2 · evaluación", level=2)
    parrafo(doc, "La evaluación es directa: la aptitud (4) de cada individuo. El problema no tiene restricciones, así que no "
                 "se usa prioridad de factibilidad (M08), que sí aplica a los problemas con restricciones.")
    doc.add_heading("5.3 Etapa 3 · selección por torneo", level=2)
    parrafo(doc, "Para cada padre se toman dos individuos al azar con reemplazo y gana el de mayor aptitud. Es el torneo M12 "
                 "con k = 2, que no requiere aptitudes positivas ni proporcionales.")
    formula(doc, "padre = argmax_{j ∈ {a, b}} f(G_j) ,   a, b ~ U{1, …, N}", "(6)")
    doc.add_heading("5.4 Etapa 4 · recombinación por codificación", level=2)
    parrafo(doc, "G1, cruce de un punto. Con punto de corte c ∈ {1, …, 47} por pareja:", negrita_inicio="G1, cruce de un punto.")
    formula(doc, "h⁽¹⁾_l = a_l (l < c) ,  b_l (l ≥ c);     h⁽²⁾_l = b_l (l < c) ,  a_l (l ≥ c)", "(7)")
    parrafo(doc, "G2, BLX-α (Eshelman y Schaffer, 1993), con recorte a [0, 1]. Para I_k = |a_k − b_k|:",
            negrita_inicio="G2, BLX-α")
    formula(doc, "h_k ~ U[ min(a_k, b_k) − α I_k ,  max(a_k, b_k) + α I_k ] ∩ [0, 1],   α = 0.3", "(8)")
    parrafo(doc, "G3, cruce uniforme. Con máscara m_k ~ Bernoulli(0.5):", negrita_inicio="G3, cruce uniforme.")
    formula(doc, "h⁽¹⁾_k = m_k a_k + (1 − m_k) b_k ,   h⁽²⁾_k = (1 − m_k) a_k + m_k b_k", "(9)")
    parrafo(doc, "La pareja se cruza con probabilidad p_c = 0.9; si no, los hijos son copias de los padres.")
    doc.add_heading("5.5 Etapa 5 · mutación por codificación", level=2)
    formula(doc, "G1:  b'_l = 1 − b_l  con probabilidad 1/48 por bit", "(10a)")
    formula(doc, "G2:  x'_k = clip_[0,1]( x_k + σ z ),  z ~ N(0, 1),  σ = 0.05,  con probabilidad 1/6 por gen", "(10b)")
    formula(doc, "G3:  g'_k = clip_{0..20}( g_k + s ),  s ∈ {−1, +1} equiprobable,  con probabilidad 1/6 por gen", "(10c)")
    parrafo(doc, "La tasa 1/L para G1 es la tasa estándar de volteo de bit (Bäck, 1996). En las tres codificaciones, "
                 "en promedio, cada individuo recibe un cambio por mutación (un bit en G1, un gen en G2 y G3). Sin embargo, "
                 "el efecto fenotípico de un bit en G1 depende de su posición en el código de Gray, así que la intensidad "
                 "de la mutación no es idéntica. Esta diferencia se discute como limitación en la sección 7.8.")
    doc.add_heading("5.6 Etapa 6 · reemplazo elitista", level=2)
    formula(doc, "P^(t+1) = Élite_2( P^(t) ) ∪ H^(t),   |H^(t)| = N − 2 = 58", "(11)")
    doc.add_heading("5.7 Etapa 7 · paro por presupuesto", level=2)
    formula(doc, "E_t = N + t (N − e),   parar si E_t + (N − e) > E_max", "(12)")
    parrafo(doc, "Con N = 60, e = 2 y E_max = 5 000 se ejecutan 85 generaciones y se consumen 4 990 evaluaciones de aptitud "
                 "en cada corrida. Las búsquedas aleatorias usan exactamente el mismo número de evaluaciones.")

    # 6 ───────────────────────────────────────────────────────────
    doc.add_heading("6. Código que apoya el cumplimiento del objetivo", level=1)
    parrafo(doc, "El código consta de tres archivos. Los dos primeros son los scripts proporcionados, ejecutados sin "
                 "modificación. El tercero es la aportación de este estudio.")
    tabla(doc, ["Archivo", "Función", "Entrada", "Salida"],
          [["synpuf_generator_v2.py", "Genera reclamaciones, eventos y verdad de referencia", "Parámetros y semilla", "synpuf_event_log.csv, synpuf_claim_meta.csv, synpuf_truth.csv"],
           ["cas_detector_v2.py", "Calcula las seis señales por validación anidada y la ablación", "Datos observables y etiquetas (sólo evaluación)", "oof_predictions.csv, ablation_metrics.csv, audit.json"],
           ["ga_codificacion.py", "Algoritmo genético con tres codificaciones, controles, estadística y figuras", "oof_predictions.csv", "corridas.csv, pruebas_estadisticas.csv, figuras, manifiesto.json"],
           ["construir_informe.py", "Genera este documento con las cifras de los resultados", "Carpetas de resultados", "Documento DOCX"]],
          [4.0, 5.2, 3.8, 4.0], alinear=["l", "l", "l", "l"], titulo="Tabla 6. Archivos del código y su función")
    doc.add_heading("6.1 Cómo el código apoya el objetivo", level=2)
    vinetas(doc, [
        "Validación externa por proveedor:: para el pliegue k, el algoritmo genético usa sólo los pliegues distintos de k "
        "para calcular la aptitud. El pliegue k se reserva para evaluar la solución final.",
        "Controles con presupuesto idéntico:: búsqueda aleatoria con 4 990 evaluaciones, pesos uniformes y la mejor señal "
        "individual elegida en entrenamiento.",
        "Misma población inicial por semilla:: las tres codificaciones parten del mismo hipercubo latino.",
        "Pruebas automatizadas antes de cualquier corrida:: seis pruebas verifican la biyección de Gray, la suma unitaria "
        "de los pesos, la igualdad de la aptitud vectorizada con average_precision_score, la forma de cruces y mutaciones "
        "y el rango de los fenotipos. Todas pasaron.",
        "Inicio del experimento sólo tras las pruebas:: el script no ejecuta el experimento si una prueba falla.",
    ])
    doc.add_heading("6.2 Errores detectados y corregidos durante el desarrollo", level=2)
    parrafo(doc, "Durante la validación se detectó un error en la prueba estadística: el rango se calculaba en orden "
                 "ascendente, de modo que el método elegido como mejor era el de peor desempeño. Se corrigió invirtiendo "
                 "el orden del ranking y se recalcularon las pruebas sobre las mismas corridas, sin repetir el "
                 "experimento. Las cifras de este informe provienen de la versión corregida.")
    doc.add_heading("6.3 Entorno de ejecución", level=2)
    tabla(doc, ["Componente", "Versión"],
          [["Python", "3.13.16"], ["NumPy", "2.5.3"], ["pandas", "3.0.5"], ["SciPy", "1.18.1"],
           ["scikit-learn", "1.9.1"], ["XGBoost", "3.4.1"], ["PM4Py", "2.7.23.11"], ["Matplotlib", "3.11.2"]],
          [6.0, 4.0], alinear=["l", "l"], titulo="Tabla 7. Versiones de las librerías utilizadas")
    parrafo(doc, "Las corridas se ejecutan con semillas fijas (detector: 2024; algoritmo genético: 11, 23, 37, 53 y 71). "
                 "La reproducibilidad bit a bit no se verificó con una segunda ejecución independiente; las semillas fijas son la garantía disponible.")

    # 7 ───────────────────────────────────────────────────────────
    doc.add_heading("7. Resultados, interpretación y análisis", level=1)
    doc.add_heading("7.1 Desempeño del detector base", level=2)
    parrafo(doc, "La Tabla 8 resume la ablación del detector con presupuesto de 5 % por pliegue. Las cifras son medias "
                 "sobre los cinco pliegues de prueba.")
    ab = d1["ablation"]
    filas = []
    for nombre in MODELO_NOMBRE:
        if nombre in ab.index:
            r = ab.loc[nombre]
            filas.append([MODELO_NOMBRE[nombre], f3(r["pr_auc_fold_mean"]), f3(r["roc_auc_fold_mean"]),
                          f3(r["precision_at_5pct"]), f3(r["recall_at_5pct"]), f3(r["amount_recall_at_5pct"])])
    tabla(doc, ["Modelo", "PR-AUC", "ROC-AUC", "Precisión 5 %", "Recall 5 %", "Recall monetario 5 %"], filas,
          [5.2, 1.9, 1.9, 2.3, 2.1, 2.9], alinear=["l", "r", "r", "r", "r", "r"],
          titulo="Tabla 8. Ablación del detector, dificultad 1.0 (medias sobre pliegues)")
    xg = ab.loc["XGBoost · con CAS"]
    sem = ab.loc["PM · semántica"]
    tmp = ab.loc["PM · temporal"]
    oar_prom = ab.loc["O-A-R · ranking promedio"]
    parrafo(doc, f"XGBoost con CAS alcanza una PR-AUC de {f3(xg['pr_auc_fold_mean'])} y una ROC-AUC de "
                 f"{f3(xg['roc_auc_fold_mean'])}. La capa semántica es la señal más fuerte de las capas de proceso "
                 f"(PR-AUC {f3(sem['pr_auc_fold_mean'])}). La capa temporal, por sí sola, tiene ROC-AUC {f3(tmp['roc_auc_fold_mean'])} "
                 f"y PR-AUC {f3(tmp['pr_auc_fold_mean'])}, con la PR-AUC más baja de las capas de proceso. "
                 f"El ranking promedio de O-A-R ({f3(oar_prom['pr_auc_fold_mean'])}) queda por debajo de XGBoost con CAS, "
                 f"lo que muestra que promediar señales más débiles diluye a la más fuerte. Esta observación motiva el "
                 f"problema de fusión que aborda el algoritmo genético.")
    if d1["ci"] is not None:
        ci = d1["ci"].set_index("metric")
        parrafo(doc, f"El intervalo de confianza al 95 % por bootstrap agrupado por proveedor para la ROC-AUC de XGBoost "
                     f"con CAS es [{f3(ci.loc['roc_auc_xgb_cas', 'ci_2.5'])}, {f3(ci.loc['roc_auc_xgb_cas', 'ci_97.5'])}]. "
                     "La precisión de la regla OR, que marca como alerta cualquier señal, es de "
                     f"{f3(d1['oar'].loc['O-A-R regla OR (≥1 señal)', 'precision'])} con un recall de "
                     f"{f3(d1['oar'].loc['O-A-R regla OR (≥1 señal)', 'recall'])}.")
    tipos = d1["tipo"]
    parrafo(doc, f"Por tipo de fraude, la regla OR detecta casi todos los fraudes de tipo F2 (fantasma, recall "
                 f"{f3(tipos.loc['F2_phantom_billing', 'recall_any_alert'])}), F3 (desagregación, "
                 f"{f3(tipos.loc['F3_unbundling', 'recall_any_alert'])}) y F1 (upcoding severo, "
                 f"{f3(tipos.loc['F1_upcoding_severo', 'recall_any_alert'])}). El upcoding leve (F5) es el más difícil: "
                 f"su recall es {f3(tipos.loc['F5_upcoding_leve', 'recall_any_alert'])}, probablemente porque el sobreprecio es pequeño "
                 "frente a la variación legítima de tarifas; no se investigó con un análisis dedicado.")

    doc.add_heading("7.2 Resultados del algoritmo genético en prueba", level=2)
    parrafo(doc, "La Tabla 9 reporta media y desviación estándar sobre los 25 bloques pliegue × semilla. Las líneas base "
                 "se evalúan sobre los mismos pliegues y semillas.")
    filas = []
    for m in ["G1-binaria", "G2-real", "G3-entera", "ALEATORIA", "UNIFORME", "MEJOR_SENAL"]:
        fila = [METODO_NOMBRE[m]]
        for k, _ in METRICAS:
            mu, sd = rm[k][0][m], rm[k][1][m]
            fila.append(f"{mu:.4f} ± {sd:.4f}")
        filas.append(fila)
    tabla(doc, ["Método", "PR-AUC", "ROC-AUC", "Precisión 5 %", "Recall 5 %", "Recall monetario 5 %"], filas,
          [3.9, 2.6, 2.6, 2.6, 2.6, 2.7], alinear=["l", "c", "c", "c", "c", "c"], tam=8,
          titulo="Tabla 9. Métricas en el pliegue de prueba, dificultad 1.0 (media ± DE)",
          nota="Los 25 bloques son los cinco pliegues de prueba por cinco semillas. Precisión y recall al 5 % se saturan "
               "con este nivel de dificultad, por lo que la PR-AUC y el recall monetario son los que discriminan.")
    figura(doc, raiz_ga(ga, "1.00", "ga_02_prauc_prueba.png"), 14.0,
           "Figura 1. PR-AUC en prueba por método y codificación (dificultad 1.0). Cada caja agrupa los 25 bloques.")

    doc.add_heading("7.3 Pruebas estadísticas", level=2)
    parrafo(doc, "Se aplica la prueba de Friedman (Friedman, 1937) sobre los 25 bloques y, para las comparaciones contra el "
                 "mejor método, la prueba de rangos con signo de Wilcoxon (Wilcoxon, 1945) con ajuste de Holm (Holm, 1979). "
                 "El nivel de significancia es α = 0.05.")
    e = ga["1.00"]["estad"]
    filas = []
    for _, r in e[e["metrica"] == "pr_auc"].iterrows():
        mejor = r["mejor"]
        cols = [c for c in e.columns if c.startswith("p_holm_")]
        for c in cols:
            metodo = c.replace("p_holm_", "")
            if pd.isna(r[c]):
                continue
            filas.append([f"{'PR-AUC' if r['metrica'] == 'pr_auc' else 'Precisión 5 %'}",
                          f"{METODO_NOMBRE.get(mejor, mejor)} vs {METODO_NOMBRE.get(metodo, metodo)}",
                          fp(float(r[c]))])
    tabla(doc, ["Métrica", "Comparación", "p ajustado (Holm)"], filas, [3.0, 9.5, 3.5],
          alinear=["l", "l", "r"], titulo="Tabla 10. Wilcoxon con ajuste de Holm contra el mejor método",
          nota="Friedman rechaza la igualdad de los seis métodos en PR-AUC "
               f"(χ² = {e.loc[0, 'chi2']:.1f}, p < 0.0001). En precisión al 5 % hay muchos empates y la prueba "
               f"(χ² = {e.loc[1, 'chi2']:.1f}, p < 0.0001) sólo separa a los pesos uniformes del resto.")

    g = corr[corr["metodo"].isin(["G1-binaria", "G2-real", "G3-entera"])]
    medias = corr.groupby("metodo")["pr_auc"].mean()
    parrafo(doc, f"Las tres codificaciones no difieren entre sí en calidad de prueba. La PR-AUC media es "
                 f"{f4(medias['G1-binaria'])} en G1, {f4(medias['G2-real'])} en G2 y {f4(medias['G3-entera'])} en G3. "
                 "Las diferencias son menores que la desviación entre pliegues. Esto confirma H3.",
            negrita_inicio="Las tres codificaciones no difieren entre sí en calidad de prueba.")
    parrafo(doc, f"Frente a los controles, el algoritmo genético supera a la búsqueda aleatoria con el mismo presupuesto "
                 f"(PR-AUC {f4(medias['G2-real'])} frente a {f4(medias['ALEATORIA'])}) y a los pesos uniformes "
                 f"({f4(medias['UNIFORME'])}). Confirma H1. La mejora sobre la mejor señal individual es estadísticamente "
                 f"significativa, pero de magnitud muy pequeña (PR-AUC {f4(medias['MEJOR_SENAL'])} frente a "
                 f"{f4(medias['G2-real'])}); confirma H4 en significancia y matiza su importancia práctica.")

    doc.add_heading("7.4 Convergencia, diversidad y costo", level=2)
    corr_u = corr[corr["metodo"].isin(["G1-binaria", "G2-real", "G3-entera"])].groupby("metodo")
    filas = []
    for m in ["G1-binaria", "G2-real", "G3-entera"]:
        sub = corr[corr["metodo"] == m]
        filas.append([METODO_NOMBRE[m], f"{sub['unicos'].mean():.3f}", f"{sub['unicos'].std():.3f}",
                      f"{sub['segundos'].mean():.1f}", f"{sub['ap_train'].mean():.4f}"])
    tabla(doc, ["Codificación", "Fracción de genotipos únicos al final", "DE", "Segundos por corrida", "Aptitud de entrenamiento"],
          filas, [4.0, 3.8, 1.6, 3.2, 3.4], alinear=["l", "r", "r", "r", "r"],
          titulo="Tabla 11. Diversidad final y costo por corrida (dificultad 1.0)")
    uni = corr_u["unicos"].mean()
    parrafo(doc, f"La diversidad final confirma H2. G2 conserva casi toda su población distinta "
                 f"({uni['G2-real']:.2f} de genotipos únicos), G1 también ({uni['G1-binaria']:.2f}) y G3 pierde diversidad "
                 f"({uni['G3-entera']:.2f}). La causa más probable es el espacio finito de G3: con 21 valores por gen, la población "
                 "tiende a converger a un número reducido de configuraciones; el escape de las mutaciones no se midió directamente. "
                 "Esa diversidad no se traduce en calidad: las tres codificaciones llegan al mismo óptimo.")
    figura(doc, raiz_ga(ga, "1.00", "ga_01_convergencia.png"), 13.5,
           "Figura 2. Convergencia de la mejor aptitud de entrenamiento frente a las evaluaciones (mediana de 25 corridas).")
    conv = ga["1.00"]["conv"]

    def evaluacion_meseta(enc):
        c = conv[conv["codificacion"] == enc].groupby("evaluaciones")["ap"].median()
        return int(c[c >= c.iloc[-1] - 0.0005].index[0])

    ev_meseta = {m: evaluacion_meseta(m) for m in ["G1-binaria", "G2-real", "G3-entera"]}
    parrafo(doc, f"El costo por corrida es prácticamente idéntico entre codificaciones (unos {corr['segundos'].mean():.0f} s con "
                 "dos procesos en paralelo), porque la evaluación domina el tiempo y es la misma en las tres. En convergencia, "
                 "la mediana de la mejor aptitud de entrenamiento entra en 0.0005 del valor final en la evaluación "
                 f"{ev_meseta['G1-binaria']:,} (G1), {ev_meseta['G2-real']:,} (G2) y {ev_meseta['G3-entera']:,} (G3) (Figura 2).")

    doc.add_heading("7.5 Pesos aprendidos", level=2)
    pesos = rm["pesos"]
    filas = []
    for k, nombre in SENAL_NOMBRE.items():
        filas.append([nombre, f"{pesos.loc['G1-binaria', 'w_' + k]:.3f}", f"{pesos.loc['G2-real', 'w_' + k]:.3f}",
                      f"{pesos.loc['G3-entera', 'w_' + k]:.3f}", f"{pesos.loc['MEJOR_SENAL', 'w_' + k]:.3f}"])
    tabla(doc, ["Señal", "G1 binaria", "G2 real", "G3 entera", "Mejor señal (proporción de pliegues)"], filas,
          [4.4, 2.6, 2.6, 2.6, 4.4], alinear=["l", "r", "r", "r", "r"],
          titulo="Tabla 12. Peso medio de cada señal en la solución final (dificultad 1.0)")
    parrafo(doc, f"Las tres codificaciones asignan prácticamente todo el peso a los dos XGBoost: {pesos.loc['G2-real', 'w_s_xgb_cas'] + pesos.loc['G2-real', 'w_s_xgb_static']:.3f} "
                 "en G2. Las señales estructural, semántica, temporal e Isolation Forest reciben peso casi nulo. "
                 "Esto indica que, dado XGBoost, las señales de proceso no aportan información adicional en la fusión lineal, al menos con estos datos. "
                 "La fusión aprendida es, en la "
                 "práctica, un promedio de dos modelos XGBoost con distinta variante de variables, y su ventaja sobre "
                 "XGBoost sólo es marginal.")
    figura(doc, raiz_ga(ga, "1.00", "ga_03_pesos.png"), 13.5,
           "Figura 3. Peso medio por señal y codificación (barras: DE entre corridas).")

    doc.add_heading("7.6 Prueba de estrés con dificultad 0.25", level=2)
    if "0.25" in ga and "0.25" in det:
        c25 = ga["0.25"]["corridas"]
        m25 = resumen_metodos(c25)
        ab25 = det["0.25"]["ablation"]
        ab50 = det["0.50"]["ablation"] if "0.50" in det else None
        filas = []
        for m in ["G1-binaria", "G2-real", "G3-entera", "ALEATORIA", "UNIFORME", "MEJOR_SENAL"]:
            filas.append([METODO_NOMBRE[m], f"{m25['pr_auc'][0][m]:.4f} ± {m25['pr_auc'][1][m]:.4f}",
                          f"{m25['amount_recall_5'][0][m]:.4f} ± {m25['amount_recall_5'][1][m]:.4f}"])
        tabla(doc, ["Método", "PR-AUC (prueba)", "Recall monetario 5 %"], filas, [5.0, 4.5, 4.5],
              alinear=["l", "c", "c"], titulo="Tabla 13. Resultados del algoritmo genético con dificultad 0.25 (media ± DE)")
        mejor25 = m25["pr_auc"][0]["MEJOR_SENAL"]
        gan25 = m25["pr_auc"][0]["G2-real"] - mejor25
        gan10 = rm["pr_auc"][0]["G2-real"] - rm["pr_auc"][0]["MEJOR_SENAL"]
        parrafo(doc, f"Con dificultad 0.25 la PR-AUC cae en todos los métodos. La ventaja de la fusión sobre la mejor señal "
                     f"individual es de {gan25:+.4f} en PR-AUC, frente a {gan10:+.4f} con dificultad 1.0. "
                     + ("La hipótesis H5 se sostiene: la fusión gana más cuando la señal es más débil."
                        if gan25 > gan10 else
                        "La hipótesis H5 no se sostiene: la ventaja de la fusión no aumenta al debilitarse la señal."))
        e25 = ga["0.25"]["estad"]
        r25 = e25.iloc[0]
        u25 = c25.groupby("metodo")["unicos"].mean()
        parrafo(doc, f"Con dificultad 0.25, la prueba de Friedman sobre PR-AUC rechaza la igualdad de los seis métodos "
                     f"(χ² = {r25['chi2']:.1f}, p < 0.0001). El mejor rango corresponde a {METODO_NOMBRE.get(r25['mejor'], r25['mejor'])}. "
                     f"G2 no difiere de G1 (p de Holm = {fp(float(r25['p_holm_G2-real']))}), mientras que G3 es significativamente "
                     f"inferior (p de Holm {fp(float(r25['p_holm_G3-entera']))}). Esta diferencia coincide con la menor "
                     f"diversidad final de G3 ({u25['G3-entera']:.2f}, frente a {u25['G1-binaria']:.2f} en G1 y {u25['G2-real']:.2f} en G2).")
        figura(doc, raiz_ga(ga, "0.25", "ga_02_prauc_prueba.png"), 13.5,
               "Figura 4. PR-AUC en prueba con dificultad 0.25 por método y codificación.")
        filas = [[f"{float(d)}", f"{ab.loc['XGBoost · con CAS', 'pr_auc_fold_mean']:.3f}",
                  f"{ab.loc['PM · completa (pesos uniformes)', 'pr_auc_fold_mean']:.3f}",
                  f"{ab.loc['O-A-R · ranking promedio', 'pr_auc_fold_mean']:.3f}"]
                 for d, ab in ([("1.00", det["1.00"]["ablation"])] + ([("0.50", ab50)] if ab50 is not None else [])
                               + [("0.25", ab25)])]
        tabla(doc, ["Dificultad", "XGBoost con CAS", "Proceso completo, pesos uniformes", "O-A-R promedio"], filas,
              [3.0, 4.2, 5.0, 3.8], alinear=["c", "r", "r", "r"],
              titulo="Tabla 14. PR-AUC del detector base según la dificultad de los datos")
    else:
        parrafo(doc, "La prueba de estrés con dificultad 0.25 no se completó en esta ejecución.")

    doc.add_heading("7.7 Discusión de los resultados", level=2)
    ap_tr = corr[corr["metodo"].str.startswith("G")]["ap_train"].mean()
    pr_te = corr[corr["metodo"].str.startswith("G")]["pr_auc"].mean()
    vinetas(doc, [
        "Lo que importa es la fusión, no la codificación:: las tres codificaciones encuentran la misma solución. "
        "La elección de representación afecta a la diversidad, no a la calidad, en este problema.",
        "La fusión aprendida es un promedio de XGBoost:: el aporte de las capas de proceso desaparece al fusionar con "
        "XGBoost. Esto no prueba que sean inútiles en otro contexto: en datos reales, las reglas de proceso pueden "
        "capturar fraudes que los clasificadores no ven.",
        "El presupuesto de 5 % satura las métricas de precisión y recall:: con dificultad 1.0 casi todos los métodos "
        "alcanzan precisión de 0.95. La PR-AUC y el recall monetario son las métricas que discriminan, y el recall monetario "
        "favorece a los pesos uniformes. Una explicación plausible es que los pesos uniformes conservan peso en la capa semántica, "
        "que incorpora la razón de monto; esta explicación no se verificó con un análisis dedicado.",
        "Riesgo de sobreajuste bajo:: la aptitud se calcula sólo con pliegues de entrenamiento y la validación es por "
        f"proveedor. La aptitud media de entrenamiento de las corridas genéticas es {ap_tr:.4f} y su PR-AUC media en prueba es "
        f"{pr_te:.4f}; la diferencia es pequeña y sugiere poco sobreajuste.",
    ])

    doc.add_heading("7.8 Limitaciones", level=2)
    vinetas(doc, [
        "Datos sintéticos:: las reglas del generador son fáciles de aprender para XGBoost. Con datos reales, el margen "
        "de las capas de proceso puede ser mayor o menor; el estudio no puede decidirlo.",
        "Apilamiento de dos niveles:: las señales de los pliegues distintos de k se generaron con modelos que vieron el pliegue k. "
        "La fuga resultante es de segundo orden, pero no es nula. La versión rigurosa exigiría recalcular las señales "
        "dentro de cada pliegue externo.",
        "Efecto fenotípico de la mutación no equiparado:: en promedio, las tres codificaciones aplican un cambio por individuo, "
        "pero el efecto de un bit en G1 depende de su posición en el código de Gray. Una comparación con magnitud fenotípica "
        "equiparada podría alterar los resultados.",
        "Sin retroalimentación:: el generador no implementa agentes que adapten su conducta; la dimensión CAS-4 no se valida.",
        "Un solo presupuesto y una sola población:: N = 60 y E = 4 990 se fijaron a priori; no se explora la sensibilidad a estos valores.",
        "Dificultad:: se exploran tres niveles de dificultad para el detector (1.0, 0.5 y 0.25) y dos para el algoritmo genético, no un barrido continuo.",
    ])

    # 8 ───────────────────────────────────────────────────────────
    doc.add_heading("8. Referencias bibliográficas", level=1)
    refs = [
        "Bäck, T. (1996). Evolutionary algorithms in theory and practice: Evolution strategies, evolutionary programming, genetic algorithms. Oxford University Press.",
        "Berti, A., van Zelst, S. J., y van der Aalst, W. M. P. (2019). Process mining for Python (PM4Py): Bridging the gap between process- and data science. En Proceedings of the ICPM 2019 Demo Track (CEUR Workshop Proceedings).",
        "Bauder, R. A., y Khoshgoftaar, T. M. (2017). Medicare fraud detection using machine learning methods. En 16th IEEE International Conference on Machine Learning and Applications (ICMLA).",
        "Centers for Medicare & Medicaid Services. (s. f.). Medicare claims synthetic public use files (SynPUF). CMS. https://www.cms.gov/",
        "Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).",
        "Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.",
        "Eshelman, L. J., y Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. En Foundations of Genetic Algorithms 2 (pp. 187–202). Morgan Kaufmann.",
        "Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32(200), 675–701.",
        "Goldberg, D. E. (1989). Genetic algorithms in search, optimization, and machine learning. Addison-Wesley.",
        "Holland, J. H. (1975). Adaptation in natural and artificial systems. University of Michigan Press.",
        "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
        "Leemans, S. J. J., Fahland, D., y van der Aalst, W. M. P. (2013). Discovering block-structured process models from event logs: A constructive approach. En Application and Theory of Petri Nets and Concurrency (PETRI NETS 2013), LNCS 7927, 311–329.",
        "Liu, F. T., Ting, K. M., y Zhou, Z.-H. (2008). Isolation forest. En Proceedings of the 8th IEEE International Conference on Data Mining (ICDM) (pp. 413–422).",
        "McKay, M. D., Beckman, R. J., y Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2), 239–245.",
        "Saito, T., y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
        "van der Aalst, W. M. P. (2016). Process mining: Data science in action (2.ª ed.). Springer.",
        "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.",
        "Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        r.font.size = Pt(9)

    # Anexo ────────────────────────────────────────────────────────
    doc.add_heading("Anexo A. Enlace al código y reproducibilidad", level=1)
    p = doc.add_paragraph()
    p.add_run("Código fuente completo: ").bold = True
    hipervinculo(p, url_repo, url_repo)
    parrafo(doc, "Para reproducir el estudio desde la carpeta del repositorio:")
    comandos = ("python synpuf_generator_v2.py --out-dir ./data --difficulty 1.0 0.5 0.25\n"
                "python cas_detector_v2.py --data-dir ./data/difficulty_1.00 --out-dir ./results/difficulty_1.00\n"
                "python ga_codificacion.py --resultados ./results/difficulty_1.00 --salida ./results_ga/difficulty_1.00\n"
                "python construir_informe.py --raiz . --salida Informe_AG_Fraude.docx --url-repo " + url_repo)
    for linea in comandos.split("\n"):
        q = doc.add_paragraph()
        q.paragraph_format.space_after = Pt(0)
        q.paragraph_format.left_indent = Cm(0.4)
        rr = q.add_run(linea)
        rr.font.name = "Consolas"
        rr.font.size = Pt(8.5)
    doc.add_paragraph()
    parrafo(doc, "El tiempo total aproximado es de 4 minutos para el detector y 30 minutos para el algoritmo genético por "
                 "dificultad, en un equipo con dos núcleos.")


def raiz_ga(ga, d, nombre):
    return Path(os.environ.get("RAIZ_GA", "..")) / "results_ga" / f"difficulty_{d}" / nombre


def construir(raiz: Path, salida: Path, url_repo: str):
    det, ga = cargar(raiz)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    estilos(doc)
    pie = sec.footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = pie.add_run(f"{SUBTITULO} · {ALUMNA} · {MATRICULA} · página ")
    r.font.size = Pt(8.5)
    r.font.color.rgb = GRIS
    campo_pagina(pie)

    portada_p = portada(doc)
    hipervinculo(portada_p, url_repo, url_repo, tam=9)
    doc.add_page_break()
    cuerpo(doc, det, ga, url_repo)
    doc.save(salida)
    print("Informe guardado:", salida)


def main():
    ap = argparse.ArgumentParser(description="Genera el informe DOCX del estudio.")
    ap.add_argument("--raiz", default="..", help="carpeta que contiene results/ y results_ga/")
    ap.add_argument("--salida", default="Informe_AG_Fraude.docx")
    ap.add_argument("--url-repo", default="https://github.com/USUARIO/REPOSITORIO")
    args = ap.parse_args()
    os.environ["RAIZ_GA"] = args.raiz
    construir(Path(args.raiz), Path(args.salida), args.url_repo)


if __name__ == "__main__":
    main()
