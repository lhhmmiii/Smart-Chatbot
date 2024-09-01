from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import BytesIO
from bs4 import BeautifulSoup
import re
import tabula
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader


def convert_pdf_to_html(pdf_path, output_html_path):
    with open(pdf_path, 'rb') as pdf_file, open(output_html_path, 'wb') as html_file:
        output_bytes = BytesIO()
        extract_text_to_fp(pdf_file, output_bytes, laparams=LAParams(), output_type='html')
        html_file.write(output_bytes.getvalue())


def indentify_separate_line(body_content):
  list_separate_page = []
  str_body = str(body_content)
  regex = r'position:absolute; top:(\d+)px;'
  _match = re.findall(regex, str_body)
  for i in _match:
    attribute = f'position:absolute; top:{i}px;'
    separate_page = body_content.find_all('div', attrs={'style': attribute})
    list_separate_page += separate_page
  return list_separate_page


def split_page_info(body_content, list_separate_page):
  list_page_content_with_html = []
  for i in range(len(list_separate_page) - 1):
    start_div = str(list_separate_page[i]) 
    end_div = str(list_separate_page[i+1]) 
    pattern = re.compile(
            re.escape(start_div) + r'(.*?)' + re.escape(end_div),
            re.DOTALL
        )
    match_ = re.findall(pattern, str(body_content))[0]
    list_page_content_with_html.append(BeautifulSoup(match_, 'html.parser'))

  return list_page_content_with_html

def preprocess_table(dfs):
    new_dfs = []
    for table in dfs:
        table = table.applymap(lambda x: x.replace('\r', '\n') if isinstance(x, str) else x)
        new_dfs.append(table)
    return new_dfs


def extract_table(page_content, table):
  # Xác định vị trí
  list_line_in_content = page_content.split('\n')
  start_pos = list_line_in_content.index(table.columns[0])
  end_pos = start_pos
  table_text = table.values.flatten()
  for i in range(len(list_line_in_content)-1,0,-1):
    if list_line_in_content[i] in table_text:
      end_pos = i
      break
  # Thay thế
  del list_line_in_content[start_pos:end_pos + 1]
  new_content = '\n'.join(list_line_in_content)
  table_text = table.to_string(index=False)
  combined_content = f"{new_content}{table_text}"
  return combined_content

def extract_info(pdf_path, html_file_path = "output.html"):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    convert_pdf_to_html(pdf_path, html_file_path)
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')
    body = soup.body
    list_page_content = []
    list_separate_page = indentify_separate_line(body)
    list_page_content_with_html = split_page_info(body, list_separate_page)
    for i, html_page in enumerate(list_page_content_with_html):
      content = ''
      for child in html_page.children:
        if child.name:
          text = child.get_text(strip=False)
          if text != '':
            content += text
      if i < len(tables):
        table = tables[0]
        new_content = extract_table(content, table)
        list_page_content.append(new_content)
      else:
        list_page_content.append(content)

    return list_page_content
