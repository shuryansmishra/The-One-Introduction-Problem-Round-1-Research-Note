import markdown2
from weasyprint import HTML

with open('docs/RESEARCH_NOTE.md', 'r') as f:
    text = f.read()

# Convert Markdown to HTML
html_content = markdown2.markdown(text, extras=["tables"])

# Generate PDF from HTML
HTML(string=html_content).write_pdf('docs/RESEARCH_NOTE.pdf')
