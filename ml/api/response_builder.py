import os
import random
import difflib
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
    
    import re
    # Bersihkan tanda baca (terutama di akhir, seperti "?", ".", ",") yang sering tertangkap NER
    name_clean = re.sub(r'[^\w\s]', '', name)
    name_lower = name_clean.lower().strip()
    
    # Cek kecocokan eksak
    if name_lower in ABBREVIATIONS:
        return ABBREVIATIONS[name_lower]
        
    # Cek parsial (replace)
    for short, full in ABBREVIATIONS.items():
        # Regex boundary manual dengan spasi (untuk menghindari salah replace)
        if f" {short} " in f" {name_lower} ":
            name_lower = f" {name_lower} ".replace(f" {short} ", f" {full.lower()} ").strip()
            
    return name_lower

from functools import lru_cache

@lru_cache(maxsize=128)
def get_destination_from_supabase(destination_name: str):
    """Fungsi helper untuk mencari destinasi di Supabase berdasarkan nama, dengan in-memory caching"""
    if not supabase:
        return None
    
    normalized_name = normalize_destination_name(destination_name)
    
    try:
        # Mencari destinasi dengan case-insensitive atau ilike
        response = supabase.table("destinations").select("*").ilike("name", f"%{normalized_name}%").limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
            
        # Jika tidak ketemu (mungkin typo), coba ambil semua nama dan lakukan fuzzy string matching
        all_dests = supabase.table("destinations").select("name").execute()
        if all_dests.data:
            db_names = [d["name"] for d in all_dests.data]
            # Cari nama yang paling mirip (toleransi kecocokan 60%)
            closest_matches = difflib.get_close_matches(normalized_name, db_names, n=1, cutoff=0.6)
            if closest_matches:
                best_match = closest_matches[0]
                print(f"Typo correction: '{normalized_name}' -> '{best_match}'")
                # Query ulang dengan nama yang benar
                response = supabase.table("destinations").select("*").eq("name", best_match).limit(1).execute()
                if response.data and len(response.data) > 0:
                    return response.data[0]
                    
        return None
    except Exception as e:
        print(f"Error querying supabase: {e}")
        return None

def build_response(intent: str, entities: dict, query: str = "") -> str:
    """
    Memetakan intent dan entities menjadi jawaban teks natural (bahasa Indonesia).
    """

    dest_name = entities.get("DESTINATION", "").strip()

    # Deteksi bahasa Inggris
    en_words = {"what", "where", "how", "when", "is", "are", "do", "can", "tell", "the", "to", "in", "of", "for", "at", "it", "i", "you", "this", "that", "not", "with", "have", "has", "an", "a", "ticket", "price", "open", "close", "time", "recommend", "recommendation", "much", "does", "any"}
    words_in_text = set(query.lower().split())
    en = len(words_in_text & en_words) >= 1

    # Heuristik: Jika model mendeteksi nama destinasi, tetapi intent-nya meleset menjadi hal yang tidak relevan
    info_seeking_intents = [
        "ask_ticket_price", "ask_operating_hours", "ask_lrt_destinations", 
        "ask_location_access", "ask_destination_info", "ask_facilities", "rule_hotel"
    ]
    if dest_name and intent not in info_seeking_intents:
        intent = "ask_destination_info"

    if intent == "greet":
        if en:
            greetings = [
                "Hello! I am Palbot 🤖. Welcome to Pelesir Palembang! How can I help you with your travel plans?",
                "Hi there! Palbot here. Ready to explore Palembang? You can ask me for recommendations, ticket prices, or locations!",
                "Hello! Nice to meet you. Is there a specific tourist spot in Palembang you want to know about?"
            ]
        else:
            greetings = [
                "Halo! Saya Palbot 🤖. Selamat datang di Pelesir Palembang! Ada yang bisa saya bantu untuk rencana perjalanan Anda?",
                "Hai sobat! Palbot di sini. Mau keliling Palembang hari ini? Boleh tanya saya tentang rekomendasi, harga tiket, atau lokasi wisata ya!",
                "Halo! Senang bertemu denganmu. Ada tempat wisata tertentu di Palembang yang ingin kamu ketahui?"
            ]
        return random.choice(greetings)

    elif intent == "goodbye":
        if en:
            goodbyes = [
                "Thank you for using Palbot! Enjoy your trip in Palembang!",
                "See you later! Don't hesitate to ask if you need more travel assistance.",
                "Have a great day! Happy exploring Palembang."
            ]
        else:
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
            return "Sorry, I am currently not connected to the database." if en else "Maaf, saya sedang tidak terhubung ke database."
        try:
            query_db = supabase.table("destinations").select("name, address, price_min, price_max").eq("category", "akomodasi")
            
            if daerah:
                daerah_keyword = " ".join(daerah.split()[:2])
                daerah_keyword = normalize_destination_name(daerah_keyword)
                query_db = query_db.or_(f"name.ilike.%{daerah_keyword}%,address.ilike.%{daerah_keyword}%")
                
            if murah:
                query_db = query_db.order("price_min", desc=False)
            elif mahal:
                query_db = query_db.order("price_max", desc=True)
            
            response = query_db.limit(10).execute()
            if response.data and len(response.data) > 0:
                results = response.data
                if not murah and not mahal:
                    random.shuffle(results)
                
                results = results[:3]
                
                hotels_list = []
                for h in results:
                    pmin = h.get("price_min") or 0
                    pmax = h.get("price_max") or 0
                    
                    if pmin != 0 and pmax != 0 and pmin != pmax:
                        harga_teks = f"Rp {pmin:,} - Rp {pmax:,}".replace(",", ".")
                    elif pmin != 0:
                        harga_teks = f"Rp {pmin:,}".replace(",", ".")
                    else:
                        harga_teks = "Price unavailable" if en else "Harga tidak tersedia"
                    hotels_list.append(f"- {h['name']} ({harga_teks})")
                
                hotels_text = "\n".join(hotels_list)
                
                if murah and daerah:
                    return f"Sure, here are the cheapest accommodations around {daerah}:\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan termurah di sekitar {daerah}:\n{hotels_text}"
                elif mahal and daerah:
                    return f"Sure, here are the most luxurious accommodations around {daerah}:\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan termewah di sekitar {daerah}:\n{hotels_text}"
                elif murah:
                    return f"Sure, here are the cheapest accommodations:\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan dengan harga termurah:\n{hotels_text}"
                elif mahal:
                    return f"Sure, here are exclusive hotels in Palembang:\n{hotels_text}" if en else f"Tentu, ini pilihan hotel eksklusif di Palembang:\n{hotels_text}"
                elif daerah:
                    return f"Here are accommodations around {daerah}:\n{hotels_text}" if en else f"Berikut rekomendasi penginapan di sekitar {daerah}:\n{hotels_text}"
                return f"Here are some accommodation recommendations in Palembang:\n{hotels_text}" if en else f"Berikut beberapa rekomendasi penginapan di Palembang:\n{hotels_text}"
            else:
                return f"Sorry, I couldn't find any accommodations around '{daerah}' in my database." if en else f"Maaf, saya tidak menemukan penginapan di daerah '{daerah}' dalam database saya."
        except Exception as e:
            print(e)
            return "An error occurred while fetching accommodation data." if en else "Terjadi kesalahan saat mencari data penginapan."

    elif intent == "rule_itinerary":
        days = entities.get("DAYS", 1)
        itinerary = []
        
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
                    if en:
                        itinerary.append(f"Day {i+1}:\n- Morning: Visit and enjoy {pagi}\n- Afternoon: Rest and taste culinary at {siang}\n- Evening: Relax at {sore}")
                    else:
                        itinerary.append(f"Hari {i+1}:\n- Pagi: Mengunjungi dan menikmati {pagi}\n- Siang: Istirahat dan mencicipi kuliner di {siang}\n- Sore: Bersantai di {sore}")
            except Exception as e:
                print("Error db itinerary:", e)
                for i in range(days):
                    itinerary.append(f"Day {i+1}: Explore Palembang from morning to evening!" if en else f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
        else:
            for i in range(days):
                itinerary.append(f"Day {i+1}: Explore Palembang from morning to evening!" if en else f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
                
        if en:
            return f"Sure! Here is a {days}-day itinerary in Palembang crafted for you:\n\n" + "\n\n".join(itinerary)
        return f"Tentu! Ini usulan rencana perjalanan {days} hari di Palembang yang disusun khusus untuk Anda:\n\n" + "\n\n".join(itinerary)

    elif intent == "ask_recommendation":
        category = entities.get("CATEGORY", "")
        if category:
            if en:
                return f"For {category}, Palembang has many interesting places! Check out the 'Recommendations' feature in the app for the full list."
            return f"Untuk kategori {category}, Palembang punya banyak tempat menarik! Coba kunjungi fitur 'Rekomendasi' di aplikasi untuk melihat daftar lengkapnya."
        
        if en:
            return "Palembang has many cool spots! Try visiting Ampera Bridge at night, or Al-Munawar Arab Village for historical tourism. Do you want more specific recommendations like culinary or museums?"
        return "Palembang punya banyak spot keren lho! Coba kunjungi Jembatan Ampera di malam hari, atau Kampung Arab Al-Munawar untuk wisata sejarah. Ingin rekomendasi yang lebih spesifik seperti kuliner atau museum?"

    elif intent == "ask_ticket_price":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                price_min = dest_data.get("price_min") or 0
                price_max = dest_data.get("price_max") or 0
                
                if price_min == 0 and price_max == 0:
                    price_text = "Free or price unavailable" if en else "Gratis atau harga tidak tersedia"
                elif price_min == price_max:
                    price_text = f"Rp {price_min:,}".replace(",", ".")
                else:
                    price_text = f"Rp {price_min:,} - Rp {price_max:,}".replace(",", ".")
                
                if en:
                    return f"The entrance ticket price for {dest_data.get('name')} is: {price_text}."
                return f"Harga tiket masuk untuk {dest_data.get('name')} adalah: {price_text}."
            
            if en:
                return f"Sorry, I couldn't find the tourist spot '{dest_name}'. Please state the full name."
            return f"Maaf, saya tidak menemukan tempat wisata '{dest_name}'. Coba sebutkan nama lengkap tempatnya ya."
        
        if en:
            return "Could you tell me the name of the tourist spot? I'll find the ticket price for you."
        return "Boleh beri tahu saya nama tempat wisatanya? Nanti saya carikan informasi harga tiketnya."

    elif intent == "ask_operating_hours":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                hours = dest_data.get("operating_hours")
                if en:
                    hours_text = hours if hours else "Operating hours unavailable."
                    return f"The operating hours of {dest_data.get('name')} are: {hours_text}."
                else:
                    hours_text = hours if hours else "Jam operasional tidak tersedia."
                    return f"Jam operasional {dest_data.get('name')} adalah: {hours_text}."
            
            if en:
                return f"Sorry, I couldn't find the operating hours for '{dest_name}'. Please state the full name."
            return f"Maaf, saya tidak menemukan informasi jam operasional untuk '{dest_name}'. Coba sebutkan nama lengkapnya."
        
        if en:
            return "Could you tell me the name of the tourist spot? I'll find the opening hours for you."
        return "Boleh beri tahu saya nama tempat wisatanya? Nanti saya carikan informasi jam bukanya."

    elif intent == "ask_destination_info":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                desc = dest_data.get("description_id", "")
                if en:
                    desc = dest_data.get("description_en", desc) # Pakai bahasa inggris jika ada di db (opsional)
                    
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                    
                if en:
                    return f"Here is brief info about {dest_data.get('name')}: {desc} If you need more details, just ask!"
                return f"Berikut info singkat tentang {dest_data.get('name')}: {desc} Jika butuh info lebih detail, silakan tanyakan lagi!"
            
            if en:
                return f"Sorry, I couldn't find the tourist spot '{dest_name}' in Palembang. Try writing the full name."
            return f"Maaf, saya tidak menemukan tempat wisata '{dest_name}' di Palembang. Coba tuliskan nama yang lebih lengkap."
        
        if en:
            return "Could you state the name of the tourist spot you want to know about?"
        return "Bisa sebutkan nama tempat wisata yang ingin kamu ketahui informasinya?"

    elif intent == "ask_hidden_gems":
        if en:
            return "Looking for hidden gems? Try visiting Kemaro Island in the morning, or enjoy the serene atmosphere at Kapitan Village. It's a completely different experience!"
        return "Suka yang tersembunyi ya? Coba kunjungi Pulau Kemaro di pagi hari, atau nikmati suasana syahdu di Kampung Kapitan. Pasti pengalaman yang beda banget dari biasanya!"

    elif intent == "ask_lrt_destinations":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                lrt = dest_data.get("lrt_accessible", False)
                if lrt:
                    return f"Yes! You can use the LRT to get to {dest_data.get('name')}." if en else f"Ya! Anda bisa menggunakan fasilitas LRT untuk menuju ke {dest_data.get('name')}."
                
                return f"It seems {dest_data.get('name')} cannot be reached directly by LRT. You might need to take a connecting angkot or online taxi." if en else f"Sepertinya {dest_data.get('name')} tidak bisa dijangkau langsung dengan LRT. Anda mungkin perlu menyambung dengan angkot atau ojek online."
            
            return f"Tourist spot '{dest_name}' not found, try the full name." if en else f"Tempat wisata '{dest_name}' tidak ditemukan, coba nama lengkapnya."
        
        return "The LRT is very efficient in Palembang! Mention the tourist spot, and I'll help find the nearest LRT station." if en else "Transportasi LRT sangat efisien lho di Palembang! Sebutkan tempat wisatanya, nanti saya bantu carikan stasiun LRT terdekatnya."

    elif intent == "ask_location_access":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                address = dest_data.get("address")
                if en:
                    addr_text = address if address else "Address unavailable."
                    return f"{dest_data.get('name')} is located at: {addr_text}. You can view the map directly in this app!"
                else:
                    addr_text = address if address else "Alamat tidak tersedia."
                    return f"{dest_data.get('name')} beralamat di: {addr_text}. Anda bisa melihat petanya langsung di aplikasi ini!"
            
            return f"I couldn't find the route to '{dest_name}'." if en else f"Saya tidak dapat menemukan rute menuju '{dest_name}'."
        
        return "Where do you want to go today? Mention the location, and I'll give you the full address." if en else "Mau ke mana hari ini? Sebutkan lokasinya, nanti saya beri tahu alamat lengkapnya."

    elif intent == "ask_category":
        if en:
            return "The Pelesir Palembang app has several tourism categories: History, Culture, Culinary, Religious, and Nature/Parks. Which category are you interested in?"
        return "Aplikasi Pelesir Palembang memiliki beberapa kategori wisata: Sejarah, Budaya, Kuliner, Religi, dan Taman/Alam. Anda sedang tertarik ke kategori yang mana?"

    elif intent == "ask_facilities":
        if dest_name:
            dest_data = get_destination_from_supabase(dest_name)
            if dest_data:
                facilities = dest_data.get("facilities", [])
                if facilities and isinstance(facilities, list):
                    fac_text = ", ".join(facilities)
                    return f"The facilities at {dest_data.get('name')} include: {fac_text}." if en else f"Fasilitas yang ada di {dest_data.get('name')} antara lain: {fac_text}."
                elif facilities and isinstance(facilities, str):
                    return f"Facilities at {dest_data.get('name')}: {facilities}." if en else f"Fasilitas di {dest_data.get('name')}: {facilities}."
                
                return f"Sorry, I don't have facility data for {dest_data.get('name')} right now." if en else f"Maaf, saya tidak memiliki data fasilitas untuk {dest_data.get('name')} saat ini."
            
            return f"Place '{dest_name}' not found." if en else f"Tempat '{dest_name}' tidak ditemukan."
        
        return "What facilities are you looking for, and at which location?" if en else "Fasilitas apa yang Anda cari, dan di lokasi mana?"

    elif intent == "provide_feedback":
        if en:
            return "Thank you for your review or report! We highly appreciate your feedback to improve the Pelesir Palembang app."
        return "Terima kasih atas ulasan atau laporannya! Kami sangat menghargai masukan Anda untuk kemajuan aplikasi Pelesir Palembang."

    # Default fallback
    if en:
        return "Sorry, I don't quite understand what you mean. I am a Palembang tourism assistant, you can ask about recommendations, ticket prices, or accommodations!"
    return "Maaf, saya kurang paham maksud Anda. Saya adalah asisten wisata Palembang, Anda bisa bertanya seputar rekomendasi tempat, harga tiket, atau penginapan!"

def build_rich_response(intent: str, entities: dict, text_reply: str) -> dict:
    """
    Membungkus text_reply dengan data terstruktur (cards, actions, quick_replies)
    untuk dirender secara interaktif di aplikasi mobile (Frontend).
    """
    result = {
        "reply": text_reply,
        "actions": None,
        "cards": None,
        "quick_replies": None
    }
    
    dest_name = entities.get("DESTINATION", "").strip()
    
    # 1. Intent seputar Destinasi Spesifik
    if intent in ["ask_destination_info", "ask_ticket_price", "ask_operating_hours", "ask_facilities", "ask_lrt_destinations", "ask_location_access"]:
        if dest_name:
            # Karena get_destination_from_supabase menggunakan @lru_cache, 
            # pemanggilan ulang di sini tidak akan membebani database (0 latency)
            dest_data = get_destination_from_supabase(dest_name)
            
            if dest_data:
                dest_id = str(dest_data["id"])
                
                # --- A. Destination Card ---
                img_url = None
                if dest_data.get("image_urls") and isinstance(dest_data["image_urls"], list) and len(dest_data["image_urls"]) > 0:
                    img_url = dest_data["image_urls"][0]
                    
                result["cards"] = [{
                    "id": dest_id,
                    "name": dest_data["name"],
                    "image_url": img_url,
                    "rating": 4.5, # Default/Mock rating
                    "category": dest_data.get("category", ""),
                    "price_text": None
                }]
                
                # --- B. Action Buttons ---
                actions = [
                    {"type": "navigate_detail", "label": "📋 Lihat Detail", "destination_id": dest_id}
                ]
                
                if dest_data.get("latitude") and dest_data.get("longitude"):
                    actions.append({
                        "type": "navigate_map", 
                        "label": "🗺️ Buka Peta", 
                        "destination_id": dest_id,
                        "lat": float(dest_data["latitude"]),
                        "lng": float(dest_data["longitude"])
                    })
                
                result["actions"] = actions
                
                # --- C. Quick Replies ---
                # Jangan tampilkan saran pertanyaan yang sama dengan intent saat ini
                all_suggestions = {
                    "ask_ticket_price": {"label": "💰 Harga Tiket", "message": f"berapa harga tiket {dest_data['name']}?"},
                    "ask_operating_hours": {"label": "🕐 Jam Buka", "message": f"jam buka {dest_data['name']}?"},
                    "ask_facilities": {"label": "🏗️ Fasilitas", "message": f"fasilitas apa saja di {dest_data['name']}?"},
                    "ask_lrt_destinations": {"label": "🚆 Akses LRT", "message": f"apakah {dest_data['name']} bisa naik LRT?"},
                    "ask_location_access": {"label": "📍 Alamat", "message": f"alamat {dest_data['name']} dimana?"}
                }
                
                # Hapus intent saat ini dari daftar saran
                if intent in all_suggestions:
                    del all_suggestions[intent]
                    
                # Ambil 3 saran acak
                import random
                suggest_keys = list(all_suggestions.keys())
                random.shuffle(suggest_keys)
                
                result["quick_replies"] = [all_suggestions[k] for k in suggest_keys[:3]]
                
    # 2. Intent Sapaan (Greeting)
    elif intent == "greet":
        result["quick_replies"] = [
            {"label": "🏛️ Wisata Sejarah", "message": "rekomendasi wisata sejarah di palembang"},
            {"label": "🍜 Kuliner Khas", "message": "rekomendasi wisata kuliner di palembang"},
            {"label": "📅 Buatkan Itinerary", "message": "buatkan jadwal perjalanan 1 hari di palembang"}
        ]
        
    # 3. Intent Rekomendasi Kategori Lokal (Fallbacks)
    elif intent == "ask_category":
        result["quick_replies"] = [
            {"label": "🏛️ Sejarah", "message": "wisata sejarah"},
            {"label": "🍜 Kuliner", "message": "wisata kuliner"},
            {"label": "🌳 Alam/Taman", "message": "wisata alam"},
            {"label": "🕌 Religi", "message": "wisata religi"}
        ]
        
    return result
