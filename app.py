# Import Library yang Diperlukan
import gradio as gr
import shutil
import os
import subprocess
from llama_cpp import Llama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import hf_hub_download
from llama_index.core.node_parser import SentenceSplitter

# Fungsi untuk mengunduh model Llama
def initialize_llama_model():
    # Unduh model jika belum ada di direktori kerja
    model_path = hf_hub_download(
        repo_id="TheBLoke/zephyr-7b-beta-GGUF",  # Nama repo model
        filename="zephyr-7b-beta.Q4_K_M.gguf",  # Nama file model
        cache_dir="./models"  # Lokasi direktori untuk menyimpan model
    )
    return model_path

# Fungsi untuk mengatur konfigurasi Settings
def initialize_settings(model_path):  
    Settings.llm = LlamaCPP(
        model_path=model_path,
        temperature=0.7,
    )

# Fungsi untuk Menginisialisasi Index
def initialize_index():
    # Tentukan dokumen input untuk pembacaan data
    documents = SimpleDirectoryReader(input_files=["bahandokumen/K3.txt",
                                                   "bahandokumen/bonus.txt",
                                                   "bahandokumen/cuti.txt",
                                                   "bahandokumen/disiplinkerja.txt",
                                                   "bahandokumen/fasilitas&bantuan.txt",
                                                   "bahandokumen/fasilitaskerja.txt",
                                                   "bahandokumen/hak.txt",
                                                   "bahandokumen/hubunganpengusaha&serikat.txt",
                                                   "bahandokumen/istilah.txt",
                                                   "bahandokumen/jaminanserikat.txt",
                                                   "bahandokumen/jamkes.txt",
                                                   "bahandokumen/jamsos.txt",
                                                   "bahandokumen/keluhkesah.txt",
                                                   "bahandokumen/kenaikanupah.txt",
                                                   "bahandokumen/kewajiban.txt",
                                                   "bahandokumen/kompensasi.txt",
                                                   "bahandokumen/larangan.txt",
                                                   "bahandokumen/lembur.txt",
                                                   "bahandokumen/luaskesepakatan.txt",
                                                   "bahandokumen/mogok.txt",
                                                   "bahandokumen/pelanggaran&sanksi.txt",
                                                   "bahandokumen/pendidikan.txt",
                                                   "bahandokumen/pengangkatan.txt",
                                                   "bahandokumen/penilaian&promosi.txt",
                                                   "bahandokumen/pensiun.txt",
                                                   "bahandokumen/perjadin.txt",
                                                   "bahandokumen/pesangon.txt",
                                                   "bahandokumen/phk.txt",
                                                   "bahandokumen/pihak.txt",
                                                   "bahandokumen/pkb.txt",
                                                   "bahandokumen/resign.txt",
                                                   "bahandokumen/sanksi.txt",
                                                   "bahandokumen/shift.txt",
                                                   "bahandokumen/syaratkerja.txt",
                                                   "bahandokumen/tatacara.txt",
                                                   "bahandokumen/tka.txt",
                                                   "bahandokumen/tunjangan.txt",
                                                   "bahandokumen/uangpisah.txt",
                                                   "bahandokumen/upah.txt",
                                                   "bahandokumen/upahlembur.txt",
                                                   "bahandokumen/waktukerja.txt"]).load_data()

    parser = SentenceSplitter(chunk_size=150, chunk_overlap=10)
    nodes = parser.get_nodes_from_documents(documents)
    embedding = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
    Settings.embed_model = embedding
    index = VectorStoreIndex(nodes)
    return index

# Inisialisasi Mesin Chat
def initialize_chat_engine(index):
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
    retriever = index.as_retriever(similarity_top_k=3)
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        verbose=True,
    )
    return chat_engine

# Fungsi untuk menghasilkan respons chatbot
def generate_response(message, history, chat_engine):
    chat_messages = [
        ChatMessage(
            role="system",
            content="Anda adalah chatbot yang selalu menjawab pertanyaan secara singkat, ramah, dan jelas dalam bahasa Indonesia."
        ),
    ]
    response = chat_engine.stream_chat(message)
    text = "".join(response.response_gen)  # Gabungkan semua token menjadi string
    history.append((message, text))  # Tambahkan ke riwayat
    return history

def clear_history(chat_engine):
    chat_engine.clear()
    
# Inisialisasi Komponen Gradio untuk UI
def launch_gradio(chat_engine):
    with gr.Blocks() as demo:
        # Mengatur tombol untuk menghapus riwayat chat
        clear_btn = gr.Button("Clear")
        clear_btn.click(lambda: clear_history(chat_engine))

        # Membuat antarmuka chat
        chat_interface = gr.ChatInterface(
            lambda message, history: generate_response(message, history, chat_engine)
        )
    demo.launch()

# Fungsi Utama untuk Menjalankan Aplikasi
def main():
    # Unduh model dan inisialisasi pengaturan
    model_path = initialize_llama_model()
    initialize_settings(model_path)  # Mengirimkan model_path ke fungsi initialize_settings
    # Inisialisasi index dan engine
    index = initialize_index()
    chat_engine = initialize_chat_engine(index)
    # Luncurkan antarmuka
    launch_gradio(chat_engine)
    
if __name__ == "__main__":
    main()
