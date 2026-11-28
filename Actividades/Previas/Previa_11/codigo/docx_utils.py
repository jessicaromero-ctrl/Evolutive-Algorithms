#!/usr/bin/env python3
"""docx_utils.py — Utilidades de formato DOCX (portada, tablas, figuras, fórmulas) para los reportes del proyecto."""

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


