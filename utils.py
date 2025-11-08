# utils.py — final version for SQL Server + Flask chatbot
import io
import os
import base64
import matplotlib

# Use headless backend for servers (important!)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union, Iterable
from html import escape

# Optional OpenAI embeddings
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
except Exception:
    _openai_client = None


# -------------------------
# 🔹 Text Embedding (optional)
# -------------------------
def embed_text(texts: Union[str, List[str]], model: str = "text-embedding-3-small"):
    """
    Returns vector embeddings for a single text or list of texts using OpenAI API.
    """
    if _openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY in .env or remove embed_text usage.")
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True
    try:
        resp = _openai_client.embeddings.create(model=model, input=texts)
        embs = [r.embedding for r in resp.data]
        return embs[0] if single else embs
    except Exception as e:
        raise RuntimeError(f"Embedding request failed: {e}")


# -------------------------
# 🔹 Plot utilities
# -------------------------
def _safe_float_list(values):
    """Converts mixed numeric-like list into clean float list."""
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            try:
                out.append(float(str(v).replace(",", "")))
            except Exception:
                continue
    return out


def plot_to_base64(x: Iterable, y: Union[Iterable[float], Iterable[Iterable[float]]], kind='line', title=''):
    """
    Generates a chart (line, bar, pie, etc.) and returns base64 image URI for embedding in HTML.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if kind == 'line':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.plot(labels, vals, marker='o', linewidth=2, color="#2c7be5")
            for xi, yi in zip(labels, vals):
                try:
                    ax.annotate(f"{yi:,.0f}", (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8)
                except Exception:
                    pass
            ax.set_ylabel("Value")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

        elif kind == 'bar':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.bar(labels, vals, color="#5bc0de")
            for rect in ax.patches:
                h = rect.get_height()
                try:
                    ax.annotate(f"{h:,.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                except Exception:
                    pass
            plt.xticks(rotation=30, ha='right')
            ax.set_ylabel("Value")
            plt.tight_layout()

        elif kind == 'pie':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            if not vals:
                vals = [1] * len(labels)
            n = min(len(labels), len(vals))
            labels = labels[:n]
            vals = vals[:n]
            ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            plt.tight_layout()

        elif kind == 'scatter':
            vals_x = _safe_float_list(x)
            vals_y = _safe_float_list(y)
            ax.scatter(vals_x, vals_y, alpha=0.7, color="#6610f2")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.tight_layout()

        elif kind == 'stacked_bar':
            labels = list(map(str, x))
            if not hasattr(y[0], "__iter__") or isinstance(y[0], (str, bytes)):
                vals = _safe_float_list(y)
                ax.bar(labels, vals)
            else:
                series = []
                for series_vals in y:
                    series.append(_safe_float_list(series_vals))
                n = len(labels)
                for i in range(len(series)):
                    if len(series[i]) < n:
                        series[i] += [0.0] * (n - len(series[i]))
                    else:
                        series[i] = series[i][:n]
                bottoms = [0.0] * n
                for s in series:
                    ax.bar(labels, s, bottom=bottoms)
                    bottoms = [a + b for a, b in zip(bottoms, s)]
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()

        elif kind == 'hist':
            vals = _safe_float_list(y)
            ax.hist(vals, bins=20, color="#20c997")
            ax.set_ylabel("Count")
            ax.set_xlabel(title or "Value")
            plt.tight_layout()

        else:
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.plot(labels, vals, marker='o')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

        ax.set_title(title or "")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        try:
            plt.close(fig)
        except Exception:
            pass
        raise RuntimeError(f"Plot failed: {e}")


# -------------------------
# 🔹 HTML table formatter
# -------------------------
def rows_to_html_table(records):
    """
    Converts a list of dictionaries (records) into a clean, responsive HTML table.
    """
    if not records:
        return "<p><i>No rows</i></p>"

    headers = list(records[0].keys())

    def fmt(val):
        if val is None:
            return ""
        if isinstance(val, (int, float)) and (abs(val) >= 1 or val == 0):
            try:
                if float(val).is_integer():
                    return f"{int(val):,}"
                else:
                    return f"{float(val):,.2f}"
            except Exception:
                pass
        return escape(str(val))

    html = [
        "<div class='table-container'>",
        "<table style='width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden; font-family:Inter, sans-serif;'>",
        "<thead style='background:#0d6efd; color:white;'>",
        "<tr>"
    ]
    for h in headers:
        html.append(f"<th style='padding:10px; text-align:left;'>{escape(h)}</th>")
    html.append("</tr></thead><tbody>")
    for r in records:
        html.append("<tr style='background:#f8f9fa;'>")
        for h in headers:
            html.append(f"<td style='padding:8px; border-bottom:1px solid #dee2e6;'>{fmt(r.get(h))}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)
