import os
import random
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_ANON_KEY", "")

# Inisialisasi Supabase client jika url dan key tersedia
supabase: Client = None
if url and key:
    supabase = create_client(url, key)

# Kamus singkatan umum wisata Palembang
ABBREVIATIONS = {
    "bkb": "Benteng Kuto Besak",
    "smb": "Sultan Mahmud Badaruddin",
    "smb ii": "Sultan Mahmud Badaruddin II",
    "smb 2": "Sultan Mahmud Badaruddin II",
    "ptc": "Palembang Trade Center",
    "pim": "Palembang Icon",
    "ps": "Palembang Square",
    "ki": "Kambang Iwak",
    "ampera": "Jembatan Ampera",
    "al munawar": "Kampung Arab Al-Munawar",
    "al-munawar": "Kampung Arab Al-Munawar",
    "monpera": "Monpera",
    "kemaro": "Pulau Kemaro",
    "kampung kapitan": "Kampung Kapitan",
    "punti kayu": "Hutan Wisata Punti Kayu",
    "kambang iwak": "Kambang Iwak",
    "benteng kuto besak": "Benteng Kuto Besak",
    "masjid agung": "Masjid Agung",
    "museum smb": "Sultan Mahmud Badaruddin",
    "jembatan ampera": "Jembatan Ampera",
    "pulau kemaro": "Pulau Kemaro",
}

def normalize_destination_name(name: str) -> str:
    """Mengubah singkatan menjadi nama lengkap berdasarkan kamus"""
    if not name:
        return name
    
    name_lower = name.lower().strip()
    
    # Cek kecocokan eksak
    if name_lower in ABBREVIATIONS:
        return ABBREVIATIONS[name_lower]
        
    # Cek parsial (replace)
    for short, full in ABBREVIATIONS.items():
        # Regex boundary manual dengan spasi (untuk menghindari salah replace)
        if f" {short} " in f" {name_lower} ":
            name_lower = f" {name_lower} ".replace(f" {short} ", f" {full.lower()} ").strip()
            
    return name_lower

def get_destination_from_supabase(destination_name: str):
    """Fungsi helper untuk mencari destinasi di Supabase berdasarkan nama"""
    if not supabase:
        return None
    
    normalized_name = normalize_destination_name(destination_name)
    
    try:
        # Mencari destinasi dengan case-insensitive atau ilike
        response = supabase.table("destinations").select("*").ilike("name", f"%{normalized_name}%").limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error querying supabase: {e}")
        return None

def build_response(intent: str, entities: dict) -> str:
    """
    Memetakan intent dan entities menjadi jawaban teks natural (bahasa Indonesia).
    """
    dest_name = entities.get("DESTINATION", "").strip()

    # Heuristik: Jika model mendeteksi nama destinasi, tetapi intent-nya meleset menjadi hal yang tidak relevan (misal kalimat terlalu pendek seperti "apa itu smb")
    # Paksa intent menjadi ask_destination_info
    info_seeking_intents = [
        "ask_ticket_price", "ask_operating_hours", "ask_lrt_destinations", 
        "ask_location_access", "ask_destination_info", "ask_facilities", "rule_hotel"
    ]
    if dest_name and intent not in info_seeking_intents:
        intent = "ask_destination_info"

    if intent == "greet":
        greetings = [
            "Halo! Saya Palbot 🤖. Selamat datang di Pelesir Palembang! Ada yang bisa saya bantu untuk rencana perjalanan Anda?",
            "Hai sobat! Palbot di sini. Mau keliling Palembang hari ini? Boleh tanya saya tentang rekomendasi, harga tiket, atau lokasi wisata ya!",
            "Halo! Senang bertemu denganmu. Ada tempat wisata tertentu di Palembang yang ingin kamu ketahui?"
        ]
        return random.choice(greetings)

    elif intent == "goodbye":
        goodbyes = [
            "Terima kasih sudah menggunakan Palbot! Selamat menikmati perjalanan Anda di Palembang!",
            "Sampai jumpa lagi! Jangan ragu untuk bertanya kalau butuh bantuan wisata ya.",
            "Semoga harimu menyenangkan! Selamat menjelajahi keindahan Palembang."
        ]
        return random.choice(goodbyes)

    elif intent == "rule_hotel":
        daerah = entities.get("DAERAH", "").strip()
        murah = entities.get("MURAH", False)
        mahal = entities.get("MAHAL", False)
        if not supabase:
            return "Maaf, saya sedang tidak terhubung ke database."
        try:
            query = supabase.table("destinations").select("name, address, price_min, price_max").eq("category", "akomodasi")
            
            if daerah:
                # Gunakan 2 kata pertama jika ada, agar lebih akurat (misal: "palembang square", "ptc")
                daerah_keyword = " ".join(daerah.split()[:2])
                daerah_keyword = normalize_destination_name(daerah_keyword) # Coba normalkan singkatan (PTC -> Palembang Trade Center)
                # Pakai syntax or_ untuk mencari di nama atau alamat
                query = query.or_(f"name.ilike.%{daerah_keyword}%,address.ilike.%{daerah_keyword}%")
                
            if murah:
                query = query.order("price_min", desc=False)
            elif mahal:
                query = query.order("price_max", desc=True)
            
            response = query.limit(10).execute()
            if response.data and len(response.data) > 0:
                results = response.data
                if not murah and not mahal:
                    random.shuffle(results)
                
                results = results[:3] # Ambil 3 terbaik / teratas / teracak
                
                hotels_list = []
                for h in results:
                    if h.get("price_min") and h.get("price_max") and h["price_min"] != h["price_max"]:
                        harga_teks = f"Rp {h['price_min']:,} - Rp {h['price_max']:,}".replace(",", ".")
                    elif h.get("price_min"):
                        harga_teks = f"Rp {h['price_min']:,}".replace(",", ".")
                    else:
                        harga_teks = "Harga tidak tersedia"
                    hotels_list.append(f"- {h['name']} ({harga_teks})")
                
                hotels_text = "\n".join(hotels_list)
                
                if murah and daerah:
                    return f"Tentu, ini rekomendasi penginapan termurah di sekitar {daerah}:\n{hotels_text}"
                elif mahal and daerah:
                    return f"Tentu, ini rekomendasi penginapan termewah/termahal di sekitar {daerah}:\n{hotels_text}"
                elif murah:
                    return f"Tentu, ini rekomendasi penginapan dengan harga termurah:\n{hotels_text}"
                elif mahal:
                    return f"Tentu, ini pilihan hotel eksklusif/termewah di Palembang:\n{hotels_text}"
                elif daerah:
                    return f"Berikut rekomendasi penginapan di sekitar {daerah}:\n{hotels_text}"
                return f"Berikut beberapa rekomendasi penginapan di Palembang:\n{hotels_text}"
            else:
                return f"Maaf, saya tidak menemukan penginapan di daerah '{daerah}' dalam database saya."
        except Exception as e:
            print(e)
            return "Terjadi kesalahan saat mencari data penginapan."

    elif intent == "rule_itinerary":
        days = entities.get("DAYS", 1)
        itinerary = []
        
        # Ambil beberapa wisata alam, sejarah, dan kuliner secara acak
        if supabase:
            try:
                res_sejarah = supabase.table("destinations").select("name").eq("category", "sejarah").execute()
                res_alam = supabase.table("destinations").select("name").eq("category", "alam").execute()
                res_kuliner = supabase.table("destinations").select("name").eq("category", "kuliner").execute()
                
                sejarah = [d["name"] for d in res_sejarah.data] if res_sejarah.data else ["Kawasan Benteng Kuto Besak (BKB)", "Museum Balaputra Dewa"]
                alam = [d["name"] for d in res_alam.data] if res_alam.data else ["Kambang Iwak Besak", "Pulau Kemaro"]
                kuliner = [d["name"] for d in res_kuliner.data] if res_kuliner.data else ["Pempek Candy", "RM Sri Melayu"]
                
                random.shuffle(sejarah)
                random.shuffle(alam)
                random.shuffle(kuliner)
                
                for i in range(days):
                    pagi = sejarah[i % len(sejarah)]
                    siang = kuliner[i % len(kuliner)]
                    sore = alam[i % len(alam)]
                    itinerary.append(f"Hari {i+1}:\n- Pagi: Mengunjungi dan menikmati {pagi}\n- Siang: Istirahat dan mencicipi kuliner di {siang}\n- Sore: Bersantai di {sore}")
            except Exception as e:
                print("Error db itinerary:", e)
                for i in range(days):
                    itinerary.append(f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
        else:
            for i in range(days):
                itinerary.append(f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
                
        return f"Tentu! Ini usulan rencana perjalanan {days} hari di Palembang yang disusun khusus untuk Anda:\n\n" + "\n\n".join(itinerary)

    elif intent == "ask_recommendation":
        category = entities.get("CATEGORY", "")
        if category:
            return f"Untuk kategori {category}, Palembang punya banyak tempat menarik! Coba kunjungi fitur 'Rekomendasi' di aplikasi untuk melihat daftar lengkapnya."
        return "Palembang punya banyak spot keren lho! Coba kunjungi Jembatan Ampera di malam hari, atau Kampung Arab Al-Munawar untuk wisata sejarah. Ingin rekomendasi yang lebih spesifik seperti kuliner atau museum?"

    elif intent == "ask_ticket_price":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                price_min = dest_data.get("price_min")
                price_max = dest_data.get("price_max")
                
                if price_min is None and price_max is None:
                    return f"Maaf, saya tidak menemukan informasi harga tiket untuk '{dest_data.get('name')}'."
                elif price_min == 0 and price_max == 0:
                    price_text = "Gratis"
                elif price_min == price_max:
                    price_text = f"Rp {price_min:,}".replace(",", ".")
                else:
                    price_text = f"Rp {price_min:,} - Rp {price_max:,}".replace(",", ".")
                
                return f"Harga tiket masuk untuk {dest_data.get('name')} adalah: {price_text}."
            return f"Maaf, saya tidak menemukan tempat wisata '{dest_name}'. Coba sebutkan nama lengkap tempatnya ya."
        return "Boleh beri tahu saya nama tempat wisatanya? Nanti saya carikan informasi harga tiketnya."

    elif intent == "ask_operating_hours":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                hours = dest_data.get("operating_hours", "Jam operasional tidak tersedia.")
                return f"Jam operasional {dest_data.get('name')} adalah: {hours}."
            return f"Maaf, saya tidak menemukan informasi jam operasional untuk '{dest_name}'. Coba sebutkan nama lengkapnya."
        return "Boleh beri tahu saya nama tempat wisatanya? Nanti saya carikan informasi jam bukanya."

    elif intent == "ask_destination_info":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                desc = dest_data.get("description_id", "")
                # Potong deskripsi agar tidak terlalu panjang
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                return f"Berikut info singkat tentang {dest_data.get('name')}: {desc} Jika butuh info lebih detail, silakan tanyakan lagi!"
            return f"Maaf, saya tidak menemukan tempat wisata '{dest_name}' di Palembang. Coba tuliskan nama yang lebih lengkap."
        return "Bisa sebutkan nama tempat wisata yang ingin kamu ketahui informasinya?"

    elif intent == "ask_hidden_gems":
        return "Suka yang tersembunyi ya? Coba kunjungi Pulau Kemaro di pagi hari, atau nikmati suasana syahdu di Kampung Kapitan. Pasti pengalaman yang beda banget dari biasanya!"

    elif intent == "ask_lrt_destinations":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                lrt = dest_data.get("lrt_accessible", False)
                if lrt:
                    return f"Ya! Anda bisa menggunakan fasilitas LRT untuk menuju ke {dest_data.get('name')}."
                return f"Sepertinya {dest_data.get('name')} tidak bisa dijangkau langsung dengan LRT. Anda mungkin perlu menyambung dengan angkot atau ojek online."
            return f"Tempat wisata '{dest_name}' tidak ditemukan, coba nama lengkapnya."
        return "Transportasi LRT sangat efisien lho di Palembang! Sebutkan tempat wisatanya, nanti saya bantu carikan stasiun LRT terdekatnya."

    elif intent == "ask_location_access":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                address = dest_data.get("address", "Alamat tidak tersedia.")
                return f"{dest_data.get('name')} beralamat di: {address}. Anda bisa melihat petanya langsung di aplikasi ini!"
            return f"Saya tidak dapat menemukan rute menuju '{dest_name}'."
        return "Mau ke mana hari ini? Sebutkan lokasinya, nanti saya beri tahu alamat lengkapnya."

    elif intent == "ask_category":
        return "Aplikasi Pelesir Palembang memiliki beberapa kategori wisata: Sejarah, Budaya, Kuliner, Religi, dan Taman/Alam. Anda sedang tertarik ke kategori yang mana?"

    elif intent == "ask_facilities":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                facilities = dest_data.get("facilities", [])
                if facilities and isinstance(facilities, list):
                    fac_text = ", ".join(facilities)
                    return f"Fasilitas yang ada di {dest_data.get('name')} antara lain: {fac_text}."
                elif facilities and isinstance(facilities, str):
                    return f"Fasilitas di {dest_data.get('name')}: {facilities}."
                return f"Maaf, saya tidak memiliki data fasilitas untuk {dest_data.get('name')} saat ini."
            return f"Tempat '{dest_name}' tidak ditemukan."
        return "Fasilitas apa yang Anda cari, dan di lokasi mana?"

    elif intent == "provide_feedback":
        return "Terima kasih atas ulasan atau laporannya! Kami sangat menghargai masukan Anda untuk kemajuan aplikasi Pelesir Palembang."

    # Default fallback
    return "Maaf, saya kurang paham maksud Anda. Saya adalah asisten wisata Palembang, Anda bisa bertanya seputar rekomendasi tempat, harga tiket, atau penginapan!"
