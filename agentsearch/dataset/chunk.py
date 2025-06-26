from langchain_text_splitters import HTMLSectionSplitter
from io import StringIO
from langchain_core.documents import Document
from typing import List

headers_to_split_on = [
    ("h1", "title"),
    ("h2", "section"),
    ("h3", "subsection"),
    ("h4", "subsubsection"),
    ("h6", "abstract"),
    ("p", "paragraph"),
]

html_splitter = HTMLSectionSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=True)

def create_chunks(path_to_html_doc: str) -> List[Document]:
    with open(path_to_html_doc, 'r') as f:
        file_content = f.read()
    file = StringIO(file_content)
    chunks = html_splitter.split_text_from_file(file)
    chunks = [chunk for chunk in chunks if chunk.metadata.get('paragraph') and len(chunk.page_content) > 100]
    return chunks

if __name__ == "__main__":
    path_to_html_doc = "papers/html/arXiv-2302.13837v2.html"
    chunks = create_chunks(path_to_html_doc)
    for i, chunk in enumerate(chunks):
        print(f"########### CHUNK {i} ###########\n{chunk.page_content}...\n\n")
