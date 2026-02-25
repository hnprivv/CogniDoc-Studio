from fpdf import FPDF
from utils import AppUtils

class PDFGenerator:
    @staticmethod
    def convert_text_to_pdf(text, output_path, font_family="Arial", font_size=12):
        try:
            pdf = FPDF()
            pdf.add_page()
            
            font_map = {
                "Sans-Serif (Arial)": "Arial",
                "Serif (Times)": "Times",
                "Monospace (Courier)": "Courier"
            }
            selected_font = font_map.get(font_family, "Arial")
            
            lines = text.split('\n')

            for line in lines:
                if not line.strip():
                    pdf.ln(5)
                    continue

                line_type, content = AppUtils.get_line_type(line)
                
                # 1. HEADINGS:
                if line_type in ['H1', 'H2', 'H3']:
                    size_offset = 8 if line_type == 'H1' else 4 if line_type == 'H2' else 2
                    pdf.set_font(selected_font, 'B', size=font_size + size_offset)
                    pdf.cell(0, 12, txt=content, ln=True, align='L')
                    pdf.ln(2)

                # 2. QUOTES
                elif line_type == 'QUOTE':
                    pdf.set_text_color(100, 100, 100)
                    pdf.set_x(20)
                    pdf.set_font(selected_font, 'I', size=font_size)
                    pdf.multi_cell(0, 8, txt=content, align='J')
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)

                # 3. BULLETS
                elif line_type == 'BULLET':
                    pdf.set_font(selected_font, size=font_size)
                    pdf.set_x(15)
                    pdf.cell(5, 10, txt=chr(149), ln=False)
                    pdf.multi_cell(0, 10, txt=content, align='J', markdown=True)

                # 4. NORMAL TEXT
                else:
                    pdf.set_font(selected_font, size=font_size)
                    pdf.multi_cell(0, 10, txt=content, align='J', markdown=True)
                    pdf.ln(2)
            
            pdf.output(output_path)
            return True, "PDF Generated Successfully"
        except Exception as e:
            return False, str(e)
