#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth

def draw_kv(c, x, y, key, val, key_w=7.0*cm, line_h=0.6*cm, font_size=10):
    c.setFont("Helvetica-Bold", font_size)
    c.drawString(x, y, f"{key}:")
    c.setFont("Helvetica", font_size)
    c.drawString(x + key_w, y, str(val))
    return y - line_h

def draw_wrapped(c, text, x, y, max_width, line_h=0.5*cm, font_name="Helvetica", font_size=10):
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_h
            line = w
    if line:
        c.drawString(x, y, line)
        y -= line_h
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics.json")
    ap.add_argument("--out", required=True, help="Output PDF path")
    args = ap.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        M = json.load(f)

    c = canvas.Canvas(args.out, pagesize=A4)
    width, height = A4
    margin = 2*cm
    max_w = width - 2*margin
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "POS Tagging Evaluation Report")
    y -= 1.0*cm

    # Dataset summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Dataset Summary")
    y -= 0.6*cm

    c.setFont("Helvetica", 10)
    y = draw_kv(c, margin, y, "Format", M.get("format"))
    y = draw_kv(c, margin, y, "Tagset family", M.get("tagset_family"))
    y = draw_kv(c, margin, y, "Sentences", M.get("num_sentences"))
    y = draw_kv(c, margin, y, "Tokens", M.get("num_tokens"))
    y = draw_kv(c, margin, y, "Mismatched sentence lengths", M.get("mismatched_sentence_lengths"))

    # Global Metrics + annotations
    y -= 0.4*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Global Metrics")
    y -= 0.6*cm
    G = M.get("global", {})
    label_map = {
        "accuracy": "Accuracy (token-level)",
        "precision_micro": "Precision (micro, ≈ accuracy)",
        "recall_micro": "Recall (micro, ≈ accuracy)",
        "f1_micro": "F1 (micro, ≈ accuracy)",
        "precision_macro": "Precision (macro, unweighted)",
        "recall_macro": "Recall (macro, unweighted)",
        "f1_macro": "F1 (macro, unweighted)",
        "precision_weighted": "Precision (weighted by support)",
        "recall_weighted": "Recall (weighted by support)",
        "f1_weighted": "F1 (weighted by support)",
    }
    for k in [
        "accuracy",
        "precision_micro","recall_micro","f1_micro",
        "precision_macro","recall_macro","f1_macro",
        "precision_weighted","recall_weighted","f1_weighted"
    ]:
        v = G.get(k, 0.0)
        if isinstance(v, float):
            v = f"{v:.4f}"
        y = draw_kv(c, margin, y, label_map.get(k, k), v)

    # Metric annotations (legend)
    y -= 0.2*cm
    c.setFont("Helvetica-Oblique", 9)
    notes = M.get("metric_notes", {})
    for title_key in ["accuracy","micro","macro","weighted"]:
        t = {
            "accuracy":"Accuracy",
            "micro":"Micro",
            "macro":"Macro",
            "weighted":"Weighted"
        }[title_key]
        line = f"{t}: {notes.get(title_key, '')}"
        y = draw_wrapped(c, line, margin, y, max_w, line_h=0.45*cm, font_name="Helvetica-Oblique", font_size=9)

    # Per-sentence exact match
    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Per-Sentence Exact Match")
    y -= 0.6*cm
    S = M.get("per_sentence", {})
    y = draw_kv(c, margin, y, "Exact-match accuracy", f"{S.get('exact_match_accuracy', 0.0):.4f}")
    y = draw_kv(c, margin, y, "Num exact", S.get("num_exact"))
    y = draw_kv(c, margin, y, "Num sentences", S.get("num_sentences"))

    # Pairs
    y -= 0.4*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Keyphrase-like Pair Metrics")
    y -= 0.6*cm
    P = M.get("pairs", {})
    c.setFont("Helvetica", 10)
    for cat, stats in P.items():
        line = f"{cat}: P={stats.get('precision',0.0):.3f} " \
               f"R={stats.get('recall',0.0):.3f} F1={stats.get('f1',0.0):.3f} " \
               f"(TP={stats.get('tp',0)} FP={stats.get('fp',0)} FN={stats.get('fn',0)})"
        y = draw_wrapped(c, line, margin, y, max_w)

        if y < margin + 6*cm:
            c.showPage(); y = height - margin

    # Examples for ADJ+NOUN (FP/FN)
    ex_all = M.get("pair_examples", {})
    ex = ex_all.get("ADJ+NOUN", {})
    if ex:
        if y < margin + 10*cm:
            c.showPage(); y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Examples — ADJ+NOUN")
        y -= 0.6*cm

        # False Positives
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "False Positives (pred = ADJ+NOUN, gold ≠ ADJ+NOUN)")
        y -= 0.5*cm
        c.setFont("Helvetica", 10)
        for exi in ex.get("false_positives", [])[:5]:
            line1 = f"• Pair '{exi.get('pair_words','')}' at pos {exi.get('position')}  |  gold={exi.get('gold_pair')}  pred={exi.get('pred_pair')}"
            y = draw_wrapped(c, line1, margin, y, max_w)
            y = draw_wrapped(c, "  Sentence: " + exi.get("sentence_text",""), margin, y, max_w)
            if y < margin + 4*cm:
                c.showPage(); y = height - margin

        # False Negatives
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "False Negatives (gold = ADJ+NOUN, pred ≠ ADJ+NOUN)")
        y -= 0.5*cm
        c.setFont("Helvetica", 10)
        for exi in ex.get("false_negatives", [])[:5]:
            line1 = f"• Pair '{exi.get('pair_words','')}' at pos {exi.get('position')}  |  gold={exi.get('gold_pair')}  pred={exi.get('pred_pair')}"
            y = draw_wrapped(c, line1, margin, y, max_w)
            y = draw_wrapped(c, "  Sentence: " + exi.get("sentence_text",""), margin, y, max_w)
            if y < margin + 4*cm:
                c.showPage(); y = height - margin

    # Confusion matrix
    cm_png = M.get("confusion_matrix_png")
    if cm_png and os.path.exists(cm_png):
        if y < margin + 10*cm:
            c.showPage(); y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Confusion Matrix")
        y -= 0.6*cm
        img = ImageReader(cm_png)
        img_w, img_h = img.getSize()
        scale = min(1.0, (width - 2*margin) / img_w)
        draw_w = img_w * scale
        draw_h = img_h * scale
        c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True)
        y -= draw_h + 0.5*cm

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawRightString(width - margin, margin/2, "Generated by report_pdf.py")
    c.save()
    print(f"[OK] Report written to {args.out}")

if __name__ == "__main__":
    main()
