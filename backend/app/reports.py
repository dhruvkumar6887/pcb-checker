from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

def pdf_report(report, image_path):
    output = BytesIO()
    styles = getSampleStyleSheet()
    def paragraph(text, style='BodyText'):
        return Paragraph(escape(str(text)),styles[style])
    document = SimpleDocTemplate(output, title='PCB inspection report')
    story = [paragraph('PCB inspection report','Title'),paragraph(report['board_name'],'Heading2'),
             paragraph(report['summary']),paragraph(report['confidence_note']),Spacer(1,12)]
    width,height = report['original_size']
    scale = min(6.2*inch/width,5*inch/height)
    story.append(Image(str(image_path),width=width*scale,height=height*scale,kind='proportional'))
    for part in report['components']:
        story += [Spacer(1,10),paragraph(f"{part['designator']} - {part['value']} - {part['status']} ({part['confidence']:.0%})",'Heading3')]
        reading = part['ideal_reading']
        story += [paragraph(reading['measurement']),paragraph(reading['expected']),
                  paragraph(f"Source: {reading['source']}; confidence {reading['confidence']:.0%}"),
                  paragraph(part['downstream_effect'])]
        for cause in part['probable_defects']:
            story.append(paragraph(f"Possible cause: {cause['cause']} ({cause['likelihood']:.0%}; source: {cause['source']})"))
        for warning in part['warnings']: story.append(paragraph(warning))
    document.build(story)
    return output.getvalue()
