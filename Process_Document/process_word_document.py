import pandas as pd
from spire.doc import *
from spire.doc.common import *


def convert_table_into_text(table):
  table_data = []
  for j in range(0, table.Rows.Count):
    row_data = []
    for k in range(0, table.Rows.get_Item(j).Cells.Count):
      cell = table.Rows.get_Item(j).Cells.get_Item(k)
      cellText = ''
      for para in range(cell.Paragraphs.Count):
        paragraphText = cell.Paragraphs.get_Item(para).Text
        cellText += (paragraphText + ' ')
      row_data.append(cellText)
    table_data.append(row_data)
  df = pd.DataFrame(table_data, index = None)
  df.columns = df.iloc[0]
  table_text = df[1:].to_string(index = True)
      
  return table_text

def extract_word_content(document_path):
  doc = Document()
  doc.LoadFromFile(document_path)
  section = doc.Sections[0]
  list_section_content = []
  for s in range(doc.Sections.Count):
    elements = section.Body.ChildObjects
    section_content = ''
    section = doc.Sections.get_Item(s)
    for i in range(elements.Count):
      element = elements.get_Item(i)
      if element.DocumentObjectType == DocumentObjectType.Paragraph:
          paragraph = element
          section_content += paragraph.Text + '\n'

      elif element.DocumentObjectType == DocumentObjectType.Table:
          table = element
          table_text = convert_table_into_text(table)
          section_content += table_text + '\n'
    list_section_content.append(section_content)
  return list_section_content



