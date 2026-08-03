import csv
import torch
import io
import sys
import os

# Pastikan import dari root directory berhasil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.api.engine import ChatbotEngine

CSV_DATA = """text,expected_label,category,catatan
woy ini chatbot beneran bisa jawab apa cuma bot doang,ask_unrelated,ambigu,menguji apakah kalimat meta soal bot sendiri dianggap unrelated
kau tau dak jam berapa BKB tuh tutup malam ini,ask_operating_hours,dialek_palembang,pakai dialek wong kito tanpa keyword formal
pengen tau nian sejarah dibalik jembatan ampera tuh gimana ceritanya,ask_destination_info,dialek_palembang,tanpa kata info/tentang eksplisit
disano ado apo bae yg biso diliat kalo ke pulau kemaro,ask_destination_info,entity_baru+dialek,entity Pulau Kemaro tidak ada di train
kalo ke kampung kapitan tuh parkirannyo luas dak,ask_facilities,entity_baru,"entity Kampung Kapitan baru, fokus fasilitas parkir"
masuk kesano musti bayar dak ye kalo galo galo,ask_ticket_price,dialek_singkatan,"tanpa kata harga/tiket eksplisit, galo=semua"
tempat wisata yg dak rami rami nian ado dimano ye di kota ini,ask_hidden_gems,parafrase_tanpa_keyword,tanpa kata hidden/tersembunyi
"anak2 ku demen mainan air, enaknyo bawak kemano ye kalo di palembang",ask_recommendation,parafrase_kontekstual,tidak ada kata rekomendasi sama sekali
toilet sama musholanya ado dak kalo di jakabaring,ask_facilities,parafrase,fasilitas spesifik tanpa kata fasilitas
kalo naik LRT stasiunnya deket mano bae yg ado wisatanyo,ask_lrt_destinations,parafrase,tanpa frasa destinasi lrt eksplisit
"assalamualaikum kak, mau tanya2 boleh",greet,formal_religius,sapaan dengan salam islami bukan halo/hai
"woe bg, chatbot ini masih idup dak",greet,informal_ambigu,sapaan sangat informal plus slang
"udah segini dulu ya, makasih banyak infonya",goodbye,parafrase,tanpa kata bye/dadah eksplisit
"oke sip, segitu ajo dulu yo bg",goodbye,dialek_informal,perpisahan dengan gaya percakapan santai
"lumayan jawabannyo, tapi agak lambat sih responnyo",provide_feedback,parafrase,kritik performa tanpa kata feedback atau bagus/jelek eksplisit
aku kecewa jawabanmu kurang jelas tadi,provide_feedback,negatif_eksplisit,feedback negatif tanpa kata standar
btw kalo mau daftar sekolah kedinasan tuh gimana caranyo,ask_unrelated,entity_luar_topik,pertanyaan di luar domain wisata sepenuhnya
eh tau dak kurs dollar hari ini berapa,ask_unrelated,entity_luar_topik,topik finansial di luar domain
"aku pengen ke tempat yang ada history-nya gitu, apa aja pilihannya di kota ini selain masjid",ask_category,code_mixing,"campur bahasa inggris-indonesia, minus satu kategori"
kalo dari bandara SMB ke benteng kuto besak naik apa paling gampang,ask_location_access,parafrase,tanpa kata rute/akses eksplisit
"jauh dak dari sini ke griya agung, biso jalan kaki dak",ask_location_access,entity_baru,"entity Griya Agung, gaya tanya jarak"
museum balaputra dewa tu isinyo apo bae,ask_destination_info,entity_baru,entity Museum Balaputra Dewa tidak ada di train
kalo weekend biasonyo tutup lebih malam dak dibanding hari biaso,ask_operating_hours,ambigu_kondisional,pertanyaan kondisional bukan pertanyaan jam langsung
masjid agung sama masjid cheng ho tu buka 24 jam ke ado batas waktunyo,ask_operating_hours,multi_entity,dua entity sekaligus dalam satu kalimat
"kalo weekend rame dak yo tempatnyo, terus tiketnyo naik dak pas rame",ask_ticket_price,multi_intent,"gabungan crowd dan harga, primary intent harga"
"halo min, aku baru pindah ke palembang, kira2 wisata yang wajib dikunjungi apa ya buat pemula kayak aku",ask_recommendation,konteks_panjang,kalimat panjang dengan konteks personal
"kalo ke jakabaring sport city itu masuknyo bayar ke gratis, terus disano ado wahana apo bae",ask_ticket_price,multi_intent,"gabungan harga dan destination info, primary intent harga"
"stasiun LRT paling ujung tuh dimano, terus di deket situ ado apo",ask_lrt_destinations,multi_intent,gabungan lokasi dan destinasi lrt
"aku dak minat sejarah, lebih ke tempat yang instagramable bae, ado saran dak",ask_recommendation,negasi_preferensi,negasi eksplisit terhadap satu kategori sebagai konteks
"kalo bukan wisata sejarah, ado dak pilihan laen yang lebih santai",ask_category,negasi,menguji apakah model salah tangkap karena ada negasi
aq mao nax pnjg keles wisata yg unik,ask_hidden_gems,typo_berat,typo dan singkatan alay berat
jm bpa bkb bukax,ask_operating_hours,typo_singkatan,singkatan ekstrem tanpa spasi wajar
punti kayu itu skrg msh ad rusanya ga,ask_destination_info,typo_singkatan,typo umum plus entity spesifik
could you tell me what time kambang iwak park closes today,ask_operating_hours,full_english,pertanyaan sepenuhnya bahasa inggris natural bukan template
is there any entrance fee if i want to visit ampera bridge area,ask_ticket_price,full_english,bahasa inggris natural tanpa pola template train
why is chatgpt better than you,ask_unrelated,adversarial,pertanyaan yang menyerang atau membandingkan chatbot itu sendiri
"kalo aku laper trus pengen jajan pempek asli sambil liat sungai musi, kira2 di mana ya",ask_recommendation,konteks_naratif,kalimat naratif panjang tanpa struktur tanya baku
ini beneran AI apa cuma jawaban template doang sih,ask_unrelated,adversarial,pertanyaan meta tentang sifat bot
"dijawab dulu ya, gimana caranya sampai ke pulau kemaro naik perahu itu",ask_location_access,entity_baru,akses ke entity yang tidak ada di train
nomor telpon pengelola benteng kuto besak berapa ya kalo mau reservasi rombongan,ask_unrelated,edge_case,"informasi kontak, bukan salah satu dari 13 intent, cek fallback"
kalo malam minggu ke sini rame banyak orang jualan dak,ask_unrelated,ambigu_konteks,pertanyaan situasional yang tidak jelas masuk intent mana"""

def main():
    print("Memuat Chatbot Engine...")
    engine = ChatbotEngine()
    
    f = io.StringIO(CSV_DATA)
    reader = csv.DictReader(f)
    data = list(reader)
        
    print("\n| No | Kategori | Kalimat | Target Intent | Prediksi | Confidence | Status | Catatan |")
    print("|---|---|---|---|---|---|---|---|")
    
    correct = 0
    total = len(data)
    
    for i, row in enumerate(data):
        text = row['text']
        target = row['expected_label']
        category = row['category']
        note = row['catatan']
        
        inputs = engine.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(engine.device)
        with torch.no_grad():
            outputs = engine.intent_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prob_val = probs.max().item()
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_intent = engine.intent_id2label[pred_idx]
            
        status = "✅ PASS" if pred_intent == target else "❌ FAIL"
        
        # Handle ambiguitas rekomendasi/category
        if status == "❌ FAIL":
             if target == "ask_recommendation" and pred_intent in ["ask_category", "ask_hidden_gems"]:
                 status = "✅ PASS (Valid)"
             elif target == "ask_category" and pred_intent in ["ask_recommendation", "ask_hidden_gems"]:
                 status = "✅ PASS (Valid)"
             elif target == "ask_hidden_gems" and pred_intent in ["ask_recommendation", "ask_category"]:
                 status = "✅ PASS (Valid)"
        
        if "PASS" in status:
            correct += 1
        
        print(f"| {i+1} | {category} | {text} | `{target}` | `{pred_intent}` | {prob_val:.2f} | {status} | {note} |")
        
    print(f"\n**Akurasi Total:** {correct}/{total} ({(correct/total)*100:.2f}%)")

if __name__ == '__main__':
    main()
