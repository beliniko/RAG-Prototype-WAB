from unstructured.partition.pdf import partition_pdf
import os

def load_documents_from_folder(folder_path):
    """Load all PDFs from a folder."""
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        elements = partition_pdf(full_path)
        text = "\n".join([el.text for el in elements if hasattr(el, "text")])
        docs.append(text)

    return docs, pdf_files  # Rückgabe der Texte + Dateinamen zur Nachverfolgung

if __name__ == "__main__":
    folder_path = "../data"  # Ordner mit PDFs
    documents, filenames = load_documents_from_folder(folder_path)

    for i, doc in enumerate(documents):
        print(f"Document {i+1} ({filenames[i]}):\n{doc[:500]}...\n")
    print(f"Loaded {len(documents)} documents from folder '{folder_path}'.")
