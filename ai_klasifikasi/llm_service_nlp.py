import os
import requests
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_clean(teks: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Kamu adalah normalisasi teks untuk aplikasi pencatatan keuangan Indonesia.
Tugasmu: ubah deskripsi transaksi menjadi frasa Bahasa Indonesia yang sederhana dan umum.
Aturan:
- Hapus nama orang dan nama tempat spesifik
- Ganti nama brand/aplikasi asing dengan kategori umumnya dalam Bahasa Indonesia
- Pertahankan makna aslinya, jangan ubah konteks transaksi
- Hasil harus 2-5 kata saja, Bahasa Indonesia
- Jangan tambahkan penjelasan, langsung tulis hasilnya saja
- Jangan menambahkan kata "tagihan" untuk sewa/kosan/kontrakan

Contoh:
- "makan ayam bakar di warung pak balil" → "makan ayam bakar"
- "beli netflix bulanan" → "langganan hiburan streaming"
- "top up gopay" → "isi saldo dompet digital"
- "bensin di SPBU Pertamina Batam" → "beli bahan bakar"
- "nyicil kosan cibaduyut" -> "cicilan kosan"
- "beli obat di klinik
- "bayar kost bu Endah" → "bayar tempat tinggal" """
                },
                {
                    "role": "user",
                    "content": teks
                }
            ],
            temperature=0.1,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM] Error: {e}, pakai teks asli")
        return teks
 
      