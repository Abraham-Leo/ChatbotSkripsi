import gradio as gr
import gspread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from oauth2client.service_account import ServiceAccountCredentials
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document

# =============== 1. Cache dan Inisialisasi Index Google Sheets ===============
cached_index = None
cached_data = {}

def read_google_sheets_separated():
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        SPREADSHEET_ID = "1ZLmz1onvPEX4TbgPJbR4LxVZjIluf6BpISTiGS5_5Rg"
        sheet_names = ["datatarget", "datacuti", "dataabsen", "datalembur", "pkb"]
        spreadsheet = client.open_by_key(SPREADSHEET_ID)

        data_map = {}

        for sheet_name in sheet_names:
            try:
                sheet = spreadsheet.worksheet(sheet_name)
                data = sheet.get_all_values()
                headers = data[0]
                rows = data[1:]
                entries = []

                if sheet_name == "datatarget":
                    for row in rows:
                        if len(row) >= 4:
                            try:
                                jumlah = int(row[3])
                                status = "KURANG" if jumlah < 0 else "LEBIH"
                                entries.append(
                                    f"[SELISIH] Mesin: {row[0]} | Kategori: {row[1]} | Bulan: {row[2]} | Selisih: {abs(jumlah)} pcs ({status})"
                                )
                            except ValueError:
                                # Tangani jika data tidak valid
                                entries.append(
                                    f"[WARNING] Data tidak valid: {' | '.join(row)}"
                                )

                elif sheet_name == "datacuti":
                    for row in rows:
                        if len(row) >= 3:
                            entries.append(f"{row[0]} memiliki sisa cuti {row[1]} hari pada tahun {row[2]}")

                elif sheet_name == "dataabsen":
                    for row in rows:
                        if len(row) >= 3:
                            entries.append(f"Kehadiran {row[0]} adalah {row[1]} hari pada bulan {row[2]}")

                elif sheet_name == "datalembur":
                    for row in rows:
                        if len(row) >= 3:
                            entries.append(f"{row[0]} telah lembur sebanyak {row[1]} jam pada bulan {row[2]}")

                elif sheet_name == "pkb":
                    for row in rows:
                        if len(row) >= 4:
                            bab, poin, kategori, isi = row[0], row[1], row[2], row[3]
                            entries.append(f"Bab {bab}, Poin {poin} - Kategori: {kategori}\nIsi: {isi}")

                data_map[sheet_name] = entries
            except gspread.exceptions.WorksheetNotFound:
                data_map[sheet_name] = [f"❌ ERROR: Worksheet {sheet_name} tidak ditemukan."]

        return data_map
    except Exception as e:
        return {"error": str(e)}

def detect_intent(message):
    msg = message.lower()

    intent_keywords = {
        "pkb": ["ketentuan", "aturan", "kompensasi", "hak", "berlaku", "diperbolehkan", "pkb", "perusahaan", "pekerja", 
                "tenaga kerja asing", "jam kerja", "kerja lembur", "perjalanan dinas", "pengupahan", 
                "pemutusan hubungan kerja", "jaminan sosial", "kesejahteraan", "fasilitas kerja", 
                "alih tugas", "kewajiban", "disiplin kerja", "larangan", "sanksi", "mogok", 
                "pesangon", "penghargaan masa kerja", "uang pisah"],

        "cuti": ["cuti", "sisa cuti", "jumlah cuti", "berapa hari cuti", "libur"],

        "target": ["target", "aktual", "selisih", "produksi", "mesin", "pcs"],

        "lembur": ["lembur", "jam lembur", "berapa jam", "jam kerja tambahan"],

        "absensi": ["absensi", "hadir", "tidak hadir", "izin", "masuk", "alpha", "berapa hari masuk", "kehadiran"]
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        scores[intent] = sum(1 for k in keywords if k in msg)

    best_intent = max(scores, key=scores.get)
    
    # Jika tidak ada keyword yang cocok, fallback ke "all"
    return best_intent if scores[best_intent] > 0 else "all"

def initialize_index():
    global cached_index, cached_data
    cached_data = read_google_sheets_separated()
    all_text = sum(cached_data.values(), [])
    document = Document(text="\n".join(all_text))
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents([document])

    embedding = HuggingFaceEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Settings.embed_model = embedding
    cached_index = VectorStoreIndex(nodes)

def search_google_sheets_vector(query):
    if cached_data == {}:
        initialize_index()

    intent = detect_intent(query)

    if intent == "pkb":
        selected_data = cached_data.get("pkb", [])
    elif intent == "cuti":
        selected_data = cached_data.get("datacuti", [])
    elif intent == "target":
        selected_data = cached_data.get("datatarget", [])
    elif intent == "absensi":
        selected_data =cached_data.get("dataabsen", [])
    elif intent == "lembur":
        selected_data =cached_data.get("datalembur", [])
    else:
        selected_data = sum(cached_data.values(), [])

    document = Document(text="\n".join(selected_data))
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)
    nodes = parser.get_nodes_from_documents([document])

    embedding = HuggingFaceEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Settings.embed_model = embedding
    temp_index = VectorStoreIndex(nodes)

    retriever = temp_index.as_retriever(similarity_top_k=3)
    retriever.similarity_cutoff = 1.0
    retrieved_nodes = retriever.retrieve(query)

    results = [node.text for node in retrieved_nodes]
    return "\n".join(results) if results else "Maaf, saya tidak menemukan informasi yang relevan."

# =============== 2. Load Model Transformers ===============
def load_model():
    model_id = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
    )
    return pipe

# =============== 3. Prompt Generator ===============
def generate_prompt(user_message, context_data):
    prompt = f"""
### SISTEM:
Anda adalah chatbot HRD yang membantu karyawan memahami administrasi perusahaan. 
Jangan menjawab menggunakan Bahasa Inggris. 
Gunakan Bahasa Indonesia dengan gaya profesional dan ramah. 
Jika informasi tidak tersedia dalam dokumen, katakan dengan sopan bahwa Anda tidak tahu.
Jawaban harus singkat, jelas, dan sesuai konteks. 
Jangan memberikan jawaban untuk pertanyaan yang tidak diajukan oleh pengguna. 
Jangan menyertakan rekomendasi pertanyaan lain.
### DATA:
{context_data}
### PERTANYAAN:
{user_message}
### JAWABAN:
"""
    return prompt.strip()

# =============== 4. Generate Response ===============
def should_use_history(message):
    keywords = ["jika", "tadi", "sebelumnya","kalau begitu", "gimana kalau", "lanjutkan", "terus", "bagaimana dengan", "berarti", "jadi", "oke lalu"]
    return any(kata in message.lower() for kata in keywords)

def generate_response(message, history, pipe):
    if should_use_history(message) and history:
        previous_message = history[-1][0]
        combined_message = previous_message + " " + message
    else:
        combined_message = message

    context = search_google_sheets_vector(combined_message)

    if "❌ ERROR" in context or context.strip() == "" or "tidak ditemukan" in context.lower():
        return "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan tersebut."

    full_prompt = generate_prompt(message, context)
    response = pipe(full_prompt)[0]["generated_text"]

    cleaned = response.split("### JAWABAN:")[-1].split("###")[0].strip()
    return cleaned  # hanya return string!

# =============== 5. Jalankan Gradio ===============
def main():
    pipe = load_model()
    initialize_index()

    def chatbot_fn(message, history):
        return generate_response(message, history, pipe)

    gr.ChatInterface(
        fn=chatbot_fn,
        title="Chatbot HRD - Transformers",
        theme="compact"
    ).launch(share=True)

if __name__ == "__main__":
    main()
