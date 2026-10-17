"""Convert TECHNICAL_REPORT.md to styled HTML and PDF using Google Chrome headless."""
import subprocess
from pathlib import Path

def build_report_pdf():
    md_path = Path("docs/TECHNICAL_REPORT.md")
    html_path = Path("docs/TECHNICAL_REPORT.html")
    pdf_path = Path("docs/TECHNICAL_REPORT.pdf")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Simple, elegant markdown to HTML converter for publication styling
    import markdown2
    html_body = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "code-friendly", "strike"])
    
    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Technical Report: Round 2 Final Build · Team Zero</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 20mm 18mm;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 10.5pt;
      line-height: 1.6;
      color: #1e293b;
      background: #ffffff;
      max-width: 850px;
      margin: 0 auto;
      padding: 30px 40px;
    }}
    h1 {{
      font-family: 'Outfit', sans-serif;
      font-size: 24pt;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.2;
      margin-bottom: 4px;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-family: 'Outfit', sans-serif;
      font-size: 15pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 24pt;
      margin-bottom: 8pt;
      border-bottom: 1.5px solid #e2e8f0;
      padding-bottom: 4px;
    }}
    h3 {{
      font-family: 'Outfit', sans-serif;
      font-size: 12pt;
      font-weight: 600;
      color: #334155;
      margin-top: 14pt;
      margin-bottom: 6pt;
    }}
    p {{
      margin-bottom: 10pt;
      text-align: justify;
    }}
    strong {{
      color: #0f172a;
    }}
    hr {{
      border: none;
      border-top: 1px solid #e2e8f0;
      margin: 18pt 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14pt 0;
      font-size: 9pt;
      line-height: 1.4;
    }}
    th {{
      background: #f8fafc;
      font-weight: 700;
      color: #0f172a;
      text-align: left;
      padding: 8px 10px;
      border-top: 1.5px solid #cbd5e1;
      border-bottom: 1.5px solid #cbd5e1;
      font-family: 'Inter', sans-serif;
    }}
    td {{
      padding: 7px 10px;
      border-bottom: 1px solid #e2e8f0;
      color: #334155;
    }}
    tr:nth-child(even) td {{
      background: #fafbfc;
    }}
    pre, code {{
      font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
      font-size: 8.8pt;
    }}
    code {{
      background: #f1f5f9;
      color: #0f172a;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 12pt 0;
      line-height: 1.45;
    }}
    pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
    }}
    ul, ol {{
      margin-left: 20px;
      margin-bottom: 10pt;
    }}
    li {{
      margin-bottom: 4pt;
    }}
    .math {{
      overflow-x: auto;
      padding: 4px 0;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path.write_text(styled_html, encoding='utf-8')
    print(f"HTML written to {html_path}")

    # Compile with headless Chrome
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path.resolve()}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_path.exists():
        print(f"PDF generated successfully at {pdf_path} ({pdf_path.stat().st_size} bytes)")
    else:
        print("Failed to generate PDF:", res.stderr)

if __name__ == '__main__':
    build_report_pdf()
