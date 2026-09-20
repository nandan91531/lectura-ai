# =====================================================
# PDF & TXT EXPORTER UTILITY (FPDF2)
# =====================================================

import os
from fpdf import FPDF

def generate_pdf_notes(video_title, summary_text, chat_history, output_path="downloads/Lecture_Notes.pdf"):
    """
    Generates a beautifully formatted PDF document containing video summary and Q&A notes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 35, 60)
    pdf.cell(0, 10, "RAG AI Lecture Study Notes", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Source: {video_title}", ln=True, align="C")
    pdf.ln(5)
    
    # Divider line
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Summary Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 150)
    pdf.cell(0, 8, "1. Executive Summary & Key Takeaways", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    # Sanitize utf-8 for standard latin font or use multi_cell
    clean_summary = summary_text.encode('latin-1', 'replace').decode('latin-1') if summary_text else "No summary generated."
    pdf.multi_cell(0, 6, clean_summary)
    pdf.ln(8)
    
    # Q&A History Section
    if chat_history:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 80, 150)
        pdf.cell(0, 8, "2. Q&A Study Assistant Session", ln=True)
        pdf.ln(2)
        
        for q_idx, item in enumerate(chat_history, 1):
            if item["role"] == "user":
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(20, 100, 50)
                q_text = f"Q{q_idx}: {item['content']}".encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, q_text)
            elif item["role"] == "assistant":
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                a_text = f"A: {item['content']}".encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, a_text)
                pdf.ln(3)

    pdf.output(output_path)
    return output_path

def generate_txt_export(summary_text, transcription_data, chat_history):
    """
    Generates plain text export content for easy download.
    """
    lines = []
    lines.append("==================================================")
    lines.append("          RAG AI LECTURE STUDY NOTES              ")
    lines.append("==================================================\n")
    
    lines.append("--- 1. EXECUTIVE SUMMARY ---")
    lines.append(summary_text if summary_text else "No summary generated.")
    lines.append("\n" + "-"*50 + "\n")
    
    if chat_history:
        lines.append("--- 2. Q&A CHAT HISTORY ---")
        for item in chat_history:
            role = "USER" if item["role"] == "user" else "AI ASSISTANT"
            lines.append(f"[{role}]: {item['content']}\n")
        lines.append("-" * 50 + "\n")
        
    if transcription_data:
        lines.append("--- 3. FULL TRANSCRIPT WITH TIMESTAMPS ---")
        for seg in transcription_data:
            s_min = int(seg['start'] // 60)
            s_sec = int(seg['start'] % 60)
            e_min = int(seg['end'] // 60)
            e_sec = int(seg['end'] % 60)
            lines.append(f"[{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}] {seg['text']}")
            
    return "\n".join(lines)
