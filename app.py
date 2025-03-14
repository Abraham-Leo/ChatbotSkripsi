import gradio as gr
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from llama_cpp import Llama
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from huggingface_hub import hf_hub_download
from llama_index.core.llms import ChatMessage
from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.core.schema import Document

# ===================================
# 1️⃣ Fungsi Membaca Data Google Spreadsheet
# ===================================
def read_google_sheets():
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        
        SPREADSHEET_ID = "1e_cNMhwF-QYpyYUpqQh-XCw-OdhWS6EuYsoBUsVtdNg"
        sheet_names = ["datatarget", "datacuti", "dataabsen", "datalembur"]

        all_data = []
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        
        for sheet_name in sheet_names:
            try:
                sheet = spreadsheet.worksheet(sheet_name)
                data = sheet.get_all_values()
                all_data.append(f"=== Data dari {sheet_name.upper()} ===")
                all_data.extend([" | ".join(row) for row in data])
                all_data.append("\n")
            except gspread.exceptions.WorksheetNotFound:
                all_data.append(f"❌ ERROR: Worksheet {sheet_name} tidak ditemukan.")

        return "\n".join(all_data).strip()
    
    except gspread.exceptions.SpreadsheetNotFound:
        return "❌ ERROR: Spreadsheet tidak ditemukan!"
    
    except Exception as e:
        return f"❌ ERROR: {str(e)}"

# ===================================
# 2️⃣ Inisialisasi Model Llama
# ===================================
def initialize_llama_model():
    model_path = hf_hub_download(
        repo_id="TheBLoke/zephyr-7b-beta-GGUF",
        filename="zephyr-7b-beta.Q4_K_M.gguf",
        cache_dir="./models"
    )
    return model_path

# ===================================
# 3️⃣ Inisialisasi Pengaturan Model
# ===================================
def initialize_settings(model_path):
    Settings.llm = LlamaCPP(model_path=model_path, temperature=0.7)

# ===================================
# 4️⃣ Inisialisasi Index & Chat Engine
# ===================================
def initialize_index():
    text_data = read_google_sheets()
    document = Document(text=text_data)
    parser = SentenceSplitter(chunk_size=100, chunk_overlap=30)
    nodes = parser.get_nodes_from_documents([document])
    
    embedding = HuggingFaceEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Settings.embed_model = embedding
    
    index = VectorStoreIndex(nodes)
    return index

def initialize_chat_engine(index):
    retriever = index.as_retriever(similarity_top_k=3)
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        verbose=False  # ❌ Hapus verbose agar tidak ada referensi dokumen
    )
    return chat_engine

# ===================================
# 5️⃣ Fungsi untuk Merapikan Jawaban Chatbot
# ===================================
def clean_response(response):
    text = "".join(response.response_gen)  # Gabungkan teks yang dihasilkan
    text = text.replace("\n\n", "\n").strip()  # Hilangkan newline berlebihan
    text = text.replace("user:", "").replace("jawaban:", "").replace("assistant:", "").strip()
    return text

# ===================================
# 6️⃣ Fungsi untuk Menghasilkan Respons Chatbot
# ===================================
def generate_response(message, history, chat_engine):
    if history is None:
        history = []

    chat_messages = [
        ChatMessage(
            role="system",
            content=(
                "Anda adalah chatbot HRD yang membantu karyawan memahami administrasi perusahaan. "
                "Jangan menjawab menggunakan Bahasa Inggris. "
                "Gunakan Bahasa Indonesia dengan gaya profesional dan ramah. "
                "Jika informasi tidak tersedia dalam dokumen, katakan dengan sopan bahwa Anda tidak tahu. "
                "Jawaban harus singkat, jelas, dan sesuai konteks."
                "Jangan memberikan jawaban untuk pertanyaan yang tidak diajukan oleh pengguna. "
                "Jangan menyertakan rekomendasi pertanyaan lain."
            ),
        ),
    ]

    response = chat_engine.stream_chat(message)
    cleaned_text = clean_response(response)  # 🔹 Gunakan fungsi clean_response()

    history.append((message, cleaned_text))  # 🔹 Pastikan hanya teks yang masuk ke history
    return cleaned_text

# ===================================
# 7️⃣ Fungsi Utama untuk Menjalankan Aplikasi
# ===================================
def main():
    model_path = initialize_llama_model()
    initialize_settings(model_path)

    index = initialize_index()
    chat_engine = initialize_chat_engine(index)

    def chatbot_response(message, history=None):
        return generate_response(message, history, chat_engine)

    gr.Interface(
        fn=chatbot_response,
        inputs=["text"],
        outputs=["text"],
    ).launch()

if __name__ == "__main__":
    main()
