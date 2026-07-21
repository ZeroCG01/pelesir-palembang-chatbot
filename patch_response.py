import sys

def patch():
    file_path = "ml/api/response_builder.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update function signature
    content = content.replace(
        "def build_response(intent: str, entities: dict) -> str:",
        "def build_response(intent: str, entities: dict, query: str = \"\") -> str:"
    )
    
    # The actual implementation of build_response
    new_impl = """
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
                
                hotels_text = "\\n".join(hotels_list)
                
                if murah and daerah:
                    return f"Sure, here are the cheapest accommodations around {daerah}:\\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan termurah di sekitar {daerah}:\\n{hotels_text}"
                elif mahal and daerah:
                    return f"Sure, here are the most luxurious accommodations around {daerah}:\\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan termewah di sekitar {daerah}:\\n{hotels_text}"
                elif murah:
                    return f"Sure, here are the cheapest accommodations:\\n{hotels_text}" if en else f"Tentu, ini rekomendasi penginapan dengan harga termurah:\\n{hotels_text}"
                elif mahal:
                    return f"Sure, here are exclusive hotels in Palembang:\\n{hotels_text}" if en else f"Tentu, ini pilihan hotel eksklusif di Palembang:\\n{hotels_text}"
                elif daerah:
                    return f"Here are accommodations around {daerah}:\\n{hotels_text}" if en else f"Berikut rekomendasi penginapan di sekitar {daerah}:\\n{hotels_text}"
                return f"Here are some accommodation recommendations in Palembang:\\n{hotels_text}" if en else f"Berikut beberapa rekomendasi penginapan di Palembang:\\n{hotels_text}"
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
                        itinerary.append(f"Day {i+1}:\\n- Morning: Visit and enjoy {pagi}\\n- Afternoon: Rest and taste culinary at {siang}\\n- Evening: Relax at {sore}")
                    else:
                        itinerary.append(f"Hari {i+1}:\\n- Pagi: Mengunjungi dan menikmati {pagi}\\n- Siang: Istirahat dan mencicipi kuliner di {siang}\\n- Sore: Bersantai di {sore}")
            except Exception as e:
                print("Error db itinerary:", e)
                for i in range(days):
                    itinerary.append(f"Day {i+1}: Explore Palembang from morning to evening!" if en else f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
        else:
            for i in range(days):
                itinerary.append(f"Day {i+1}: Explore Palembang from morning to evening!" if en else f"Hari {i+1}: Eksplorasi kota Palembang dari pagi hingga sore!")
                
        if en:
            return f"Sure! Here is a {days}-day itinerary in Palembang crafted for you:\\n\\n" + "\\n\\n".join(itinerary)
        return f"Tentu! Ini usulan rencana perjalanan {days} hari di Palembang yang disusun khusus untuk Anda:\\n\\n" + "\\n\\n".join(itinerary)

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
"""
    
    # We will replace everything after `dest_name = entities.get("DESTINATION", "").strip()`
    # Find the split point
    split_point = '    dest_name = entities.get("DESTINATION", "").strip()'
    idx = content.find(split_point)
    
    if idx != -1:
        new_content = content[:idx] + new_impl
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Patched response_builder.py successfully.")
    else:
        print("Failed to patch response_builder.py")

if __name__ == "__main__":
    patch()
