#!/usr/bin/env python3
"""Render README.md to PDF with Mermaid support using headless Chrome."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path


CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google\\ Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    in_mermaid = False
    mermaid_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(s.strip() for s in paragraph if s.strip())
            parts.append(f"<p>{inline_markup(text)}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            items = "".join(f"<li>{inline_markup(item)}</li>" for item in list_items)
            parts.append(f"<ul>{items}</ul>")
            list_items = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if in_mermaid:
            if line.strip() == "```":
                mermaid_html = html.escape("\n".join(mermaid_lines))
                parts.append(f'<pre class="mermaid">{mermaid_html}</pre>')
                mermaid_lines = []
                in_mermaid = False
            else:
                mermaid_lines.append(line)
            continue

        if in_code:
            if line.strip() == "```":
                code_html = html.escape("\n".join(code_lines))
                class_attr = f' class="language-{code_lang}"' if code_lang else ""
                parts.append(f"<pre><code{class_attr}>{code_html}</code></pre>")
                code_lines = []
                code_lang = ""
                in_code = False
            else:
                code_lines.append(line)
            continue

        if line.startswith("```mermaid"):
            flush_paragraph()
            flush_list()
            in_mermaid = True
            mermaid_lines = []
            continue

        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            in_code = True
            code_lang = line[3:].strip()
            code_lines = []
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{inline_markup(heading_match.group(2).strip())}</h{level}>")
            continue

        bullet_match = re.match(r"^-\s+(.*)$", line)
        if bullet_match:
            flush_paragraph()
            list_items.append(bullet_match.group(1).strip())
            continue

        list_match = re.match(r"^\d+\.\s+(.*)$", line)
        if list_match:
            flush_paragraph()
            flush_list()
            parts.append(f"<p>{inline_markup(line)}</p>")
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_list()

    return "\n".join(parts)


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def build_html_document(title: str, body_html: str, landscape: bool, diagram_scale: float) -> str:
    page_size = "A4 landscape" if landscape else "A4"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #1f2937;
      --muted: #4b5563;
      --border: #d1d5db;
      --code-bg: #f3f4f6;
      --page-bg: #ffffff;
    }}
    body {{
      margin: 0;
      background: var(--page-bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 48px 64px;
    }}
    h1, h2, h3, h4, h5, h6 {{
      line-height: 1.2;
      margin: 1.2em 0 0.5em;
    }}
    p, ul, pre {{
      margin: 0.7em 0;
    }}
    ul {{
      padding-left: 1.4em;
    }}
    code {{
      background: var(--code-bg);
      padding: 0.12em 0.3em;
      border-radius: 4px;
      font-size: 0.94em;
    }}
    pre {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px 16px;
      overflow: auto;
      white-space: pre-wrap;
    }}
    pre code {{
      background: transparent;
      padding: 0;
    }}
    .mermaid {{
      background: #fff;
      padding: 12px 8px;
      text-align: center;
      break-inside: avoid-page;
      page-break-inside: avoid;
    }}
    .mermaid svg {{
      max-width: 100% !important;
      height: auto !important;
      transform-origin: top center;
      transform: scale({diagram_scale});
    }}
    @media print {{
      main {{
        max-width: none;
        padding: 0;
      }}
      .mermaid {{
        overflow: visible;
      }}
      .mermaid svg {{
        max-width: 100% !important;
      }}
    }}
    @page {{
      size: {page_size};
      margin: 14mm;
    }}
  </style>
</head>
<body>
  <main>
{body_html}
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{
      startOnLoad: false,
      securityLevel: "loose",
      theme: "default",
      flowchart: {{
        useMaxWidth: true,
        htmlLabels: true,
        nodeSpacing: 24,
        rankSpacing: 34,
        padding: 10,
      }},
    }});
    await mermaid.run({{ querySelector: ".mermaid" }});
    for (const block of document.querySelectorAll(".mermaid")) {{
      const svg = block.querySelector("svg");
      if (!svg) continue;
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.style.maxWidth = "100%";
      svg.style.height = "auto";
    }}
    document.body.setAttribute("data-mermaid-rendered", "true");
  </script>
</body>
</html>
"""


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return CHROME_CANDIDATES[0]


def render_pdf(
    readme_path: Path,
    output_pdf: Path,
    chrome_binary: str,
    html_out: Path | None,
    landscape: bool,
    diagram_scale: float,
) -> None:
    markdown_text = readme_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(markdown_text)
    html_text = build_html_document(readme_path.stem, body_html, landscape, diagram_scale)

    if html_out is None:
        html_file = output_pdf.with_suffix(".html")
    else:
        html_file = html_out
    html_file.write_text(html_text, encoding="utf-8")

    subprocess.run(
        [
            chrome_binary,
            "--headless=new",
            "--disable-gpu",
            "--disable-component-update",
            "--disable-background-networking",
            "--disable-sync",
            "--no-first-run",
            "--allow-file-access-from-files",
            f"--print-to-pdf={output_pdf}",
            "--virtual-time-budget=8000",
            html_file.resolve().as_uri(),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--output", type=Path, default=Path("README.pdf"))
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--chrome-binary", type=str, default=None)
    parser.add_argument("--landscape", action="store_true", help="Render the PDF in landscape mode.")
    parser.add_argument(
        "--diagram-scale",
        type=float,
        default=1.0,
        help="CSS scale multiplier applied to Mermaid SVGs after width fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chrome_binary = args.chrome_binary or find_chrome()
    render_pdf(args.readme, args.output, chrome_binary, args.html_out, args.landscape, args.diagram_scale)
    print(f"Wrote PDF to {args.output}")


if __name__ == "__main__":
    main()
