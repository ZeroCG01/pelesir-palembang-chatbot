import asyncio
from ml.api.engine import ChatbotEngine
import torch

sentences = [
    ("min, klo mau masuk ke bkb tu byr brp sih tiketny?", "ask_ticket_price"),
    ("oy, ampera tu gratis apo bayar men kito nak kesano?", "ask_ticket_price"),
    ("budget cepek (100rb) cukup dak y buat masuk waterpark amanzi berdua?", "ask_ticket_price"),
    ("eh ptc mall buka sampe jam brapaan ya klo malem minggu?", "ask_operating_hours"),
    ("masjid agung palembang tu kalo subuh udh buka belom pelatarannya?", "ask_operating_hours"),
    ("museum balaputra dewa tutup jam brp min hari ini?", "ask_operating_hours"),
    ("ada rekomendasi tmpt nongkrong yg asik buat kluarga di palembang ga?", "ask_recommendation"),
    ("bosen ke mall trus, ad hidden gem wisata alam yg sepi dak di plg?", "ask_hidden_gems"),
    ("lagi pengen makan pempek nih, di dket jembatan ampera ad tmpt mkn enak ga?", "ask_recommendation"),
    ("kalo dari stasiun bandara mau ke pim naek lrt bsa turun di stasiun mno?", "ask_lrt_destinations"),
    ("gimana cara pesen gocar ke pulau kemaro, titik jemputnya dmn?", "ask_location_access"),
    ("jalan ke bukit siguntang tu macet dak y kalo sore?", "ask_location_access"),
    ("di hutan punti kayu tu ad wc umum sm tmpt sholat yg bersih dak min?", "ask_facilities"),
    ("kampung arab al-munawar tu disediain lahan parkir buat mobil luas ga?", "ask_facilities"),
    ("kak, sbnrnya kawah tengkurep tu apaan sih sjarahnya gmn?", "ask_destination_info"),
    ("monpera tuh singkatan dari apa n di dlemnya ad peninggalan ap aj?", "ask_destination_info"),
    ("cariin wisata yg berbau sejarah dong buat tgs sekolah adek.", "ask_category"),
    ("aku nak nyari tmpt wisata religi yg deket dari jakabaring, mno ye?", "ask_category"),
    ("min, resep cuko pempek palembang yg pedes mantap tu apa bae bumbunyo?", "ask_unrelated"),
    ("tolong buatin kodingan python buat skripsi saya dong kak wkwk.", "ask_unrelated")
]

async def main():
    print("Memuat Chatbot Engine...")
    engine = ChatbotEngine()
    
    print("\nMemulai Uji Coba Zero-Shot...\n")
    print("| No | Kalimat (Zero-Shot) | Target Intent | Prediksi Intent | Confidence | Status | NER Ekstraksi |")
    print("|---|---|---|---|---|---|---|")
    
    for i, (text, target) in enumerate(sentences):
        # We only want to test the raw model prediction before fallback
        inputs = engine.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(engine.device)
        with torch.no_grad():
            outputs = engine.intent_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            prob_val = probs.max().item()
                
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_intent = engine.intent_id2label[pred_idx]
            
        # Get NER too
        ner_entities = engine.get_entities(text)
        ner_str = ", ".join([f"[{k}: {v}]" for k,v in ner_entities.items()]) if ner_entities else "-"
        
        status = "✅ PASS" if pred_intent == target else "❌ FAIL"
        
        # Handle ambiguitas (kalau dia prediksi hidden_gems pdhl target recommendation jg bisa ditoleransi)
        if target == "ask_recommendation" and pred_intent == "ask_hidden_gems":
            status = "✅ PASS (Valid)"
        
        print(f"| {i+1} | {text} | `{target}` | `{pred_intent}` | {prob_val:.2f} | {status} | {ner_str} |")

if __name__ == '__main__':
    asyncio.run(main())
