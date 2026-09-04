import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

out_dir = Path("docs")
out_dir.mkdir(exist_ok=True)
pdf_path = out_dir / "DRDO_EW_Smart_Scan_Whitepaper.pdf"

doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=letter,
    rightMargin=0.75 * inch,
    leftMargin=0.75 * inch,
    topMargin=0.75 * inch,
    bottomMargin=0.75 * inch
)

styles = getSampleStyleSheet()

# Custom military / academic aesthetic
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    textColor=HexColor('#1a252f'),
    alignment=1, # Centered
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=HexColor('#00838f'),
    alignment=1,
    spaceAfter=15
)

heading_style = ParagraphStyle(
    'SectionHead',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=12,
    leading=16,
    textColor=HexColor('#2c3e50'),
    spaceBefore=10,
    spaceAfter=4
)

body_style = ParagraphStyle(
    'DocBody',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    textColor=HexColor('#2b2b2b'),
    spaceAfter=6
)

code_style = ParagraphStyle(
    'EquationBox',
    parent=styles['Normal'],
    fontName='Courier-Bold',
    fontSize=9,
    leading=12,
    textColor=HexColor('#0d47a1'),
    alignment=1,
    spaceBefore=4,
    spaceAfter=6
)

story = []

# Header
story.append(Paragraph("Adaptive Bayesian Restless Multi-Armed Bandit Scheduling for Electronic Warfare", title_style))
story.append(Paragraph("DRDO Problem Statement ID: 26055 // Technical Architecture Whitepaper", subtitle_style))

# Abstract
story.append(Paragraph("<b>Abstract</b>—Legacy Electronic Support (ES) receivers rely on deterministic round-robin stepping across wideband RF spectra, resulting in low intercept probabilities (P_int ≈ 4%) against modern agile radar emitters. This paper presents an online Bayesian Restless Multi-Armed Bandit (RMAB) with conjugate Beta-Bernoulli updating, hardware PLL slew penalties, and Sequential Difference Histogramming (SDIF) de-interleaving. Validated against the Alan Turing Synthetic Radar Benchmark, the proposed framework demonstrates a 70.53% mean P_int across 1.76 million pulses, outperforming conventional architectures by 16.4x while enforcing deterministic starvation bounds (τ ≤ 60 ms).", body_style))
story.append(Spacer(1, 4))

# Section 1
story.append(Paragraph("1. Mathematical Formulation (POMDP & Whittle Index)", heading_style))
story.append(Paragraph("The monitoring of K = 16 sub-bands is modeled as a Partially Observable Markov Decision Process. Occupancy is updated online via conjugate Beta priors Beta(α_k, β_k) without pre-training:", body_style))
story.append(Paragraph("E[θ_k] = α_k / (α_k + β_k) | Updates: α_k ← α_k + min(h, 10) [hits], β_k ← β_k + 2.0 [misses]", code_style))
story.append(Paragraph("Synthesizer tuning incurs a non-linear slewing cost proportional to log-frequency hop distance. The scheduling index balances exploitation, anti-starvation exploration, and hardware settling constraints:", body_style))
story.append(Paragraph("I_k(t) = [E[θ_k] + T_hop] · Φ_k(t) · w_k + λ · (t - τ_k)/τ_max - κ · log2(1 + |k - a(t-1)|)", code_style))

# Section 2
story.append(Paragraph("2. Multi-Split Empirical Validation (Alan Turing Dataset)", heading_style))

data = [
    ["Split", "Pulses", "Emitters", "Round-Robin P_int", "Adaptive RMAB P_int", "Gain"],
    ["test_0.h5", "29,748", "78", "4.54%", "41.84%", "9.2x"],
    ["test_1.h5", "4,222", "7", "3.96%", "64.00%", "16.2x"],
    ["test_2.h5", "792,838", "19", "4.37%", "85.66%", "19.6x"],
    ["test_3.h5", "16,413", "32", "4.37%", "56.31%", "12.9x"],
    ["test_4.h5", "104,679", "44", "4.39%", "65.18%", "14.8x"],
    ["test_5.h5", "368,100", "58", "4.38%", "87.41%", "20.0x"],
    ["test_6.h5", "453,528", "28", "4.37%", "93.31%", "21.3x"],
    ["Aggregate", "1,769,528", "266", "4.34%", "70.53%", "16.4x"]
]

t = Table(data, colWidths=[1.1*inch, 1.1*inch, 0.9*inch, 1.4*inch, 1.5*inch, 0.8*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bdc3c7')),
    ('BACKGROUND', (0, -1), (-1, -1), HexColor('#ecf0f1')),
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ('TEXTCOLOR', (4, 1), (4, -1), HexColor('#27ae60')),
]))
story.append(t)
story.append(Spacer(1, 6))

# Section 3
story.append(Paragraph("3. Real-Time Radar De-interleaving & STANAG Interoperability", heading_style))
story.append(Paragraph("Intercepted pulse descriptor words (PDWs) feed directly into a real-time Sequential Difference Histogrammer (SDIF) that extracts fundamental Pulse Repetition Intervals (PRI), filters out harmonic ghost peaks, and correlates detected signatures against tactical emitter templates (e.g., Target Acquisition vs Fire Control radars). The resulting Electronic Order of Battle (EOB) is exportable in standardized JSON/CSV formats suitable for tactical datalinks.", body_style))

doc.build(story)
print(f"Whitepaper successfully generated: {pdf_path}")