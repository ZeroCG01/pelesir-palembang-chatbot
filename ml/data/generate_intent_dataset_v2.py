"""
Generator Dataset Intent v2 — Bilingual (ID + EN), 13 Intent, ~8.000 kalimat
Untuk retraining model Intent Classification XLM-RoBERTa
"""
import csv
import random
import os
import itertools
import re

random.seed(42)

# ============================================================
# DAFTAR DESTINASI WISATA PALEMBANG (untuk slot filling)
# ============================================================
DESTINATIONS_FULL = [
    "Benteng Kuto Besak", "Jembatan Ampera", "Pulau Kemaro", "Monpera",
    "Museum Sultan Mahmud Badaruddin II", "Kambang Iwak", "Hutan Wisata Punti Kayu",
    "Masjid Agung Palembang", "Kampung Kapitan", "Kampung Arab Al-Munawar",
    "Taman Kambang Iwak Besak", "Palembang Trade Center", "Palembang Icon",
    "Palembang Square", "Jakabaring Sport City", "Taman Kebon Rojo",
    "Museum Balaputra Dewa", "Al Quran Al Akbar", "Bukit Siguntang",
    "Masjid Cheng Ho", "Fantasy Island", "Amanzi Waterpark",
    "OPI Mall", "Pasar 16 Ilir", "Kawah Tengkurep",
]

DESTINATIONS_SHORT = [
    "BKB", "SMB", "SMB II", "PTC", "PIM", "Ampera", "Kemaro", "Monpera",
    "Punti Kayu", "Kambang Iwak", "KI", "Al-Munawar", "Kampung Kapitan",
    "Jakabaring", "Siguntang", "Amanzi",
]

DESTINATIONS_EN = [
    "Benteng Kuto Besak", "Ampera Bridge", "Kemaro Island", "Monpera",
    "Sultan Mahmud Badaruddin II Museum", "Kambang Iwak Park",
    "Punti Kayu Forest", "Grand Mosque of Palembang", "Kampung Kapitan",
    "Al-Munawar Arab Village", "Jakabaring Sport City",
]

ALL_DEST = DESTINATIONS_FULL + DESTINATIONS_SHORT

def get_dest_cycle():
    shuffled = ALL_DEST[:]
    random.shuffle(shuffled)
    return itertools.cycle(shuffled)

def get_dest_cycle_en():
    shuffled = DESTINATIONS_EN[:]
    random.shuffle(shuffled)
    return itertools.cycle(shuffled)

CATEGORIES_ID = ["wisata alam", "wisata sejarah", "wisata kuliner", "wisata religi", "wisata budaya", "taman", "museum", "masjid"]
CATEGORIES_EN = ["nature tourism", "historical sites", "culinary tourism", "religious tourism", "cultural tourism", "parks", "museums"]

PRICES_ID = [
    "Rp 10.000", "Rp 15.000", "Rp 20.000", "Rp 25.000", "Rp 5.000",
    "Rp 50.000", "Rp 30.000", "gratis", "10 ribu", "20 ribu", "lima ribu",
    "50 ribuan", "sepuluh ribu", "dua puluh ribu",
]

# ============================================================
# TEMPLATE-TEMPLATE PER INTENT
# ============================================================

def gen_greet():
    """Kalimat sapaan (ID + EN)"""
    sentences = []
    id_sentences = [
        "Halo", "Hai", "Selamat pagi", "Selamat siang", "Selamat sore",
        "Selamat malam", "Ping", "P", "Assalamualaikum", "Halo bot",
        "Hai chatbot", "Permisi", "Test", "Hi", "Pagi", "Siang", "Sore",
        "Malam", "Halo aplikasi", "Salam kenal", "Bisa bantu saya?",
        "Halo ada yang bisa dibantu?", "Test bot", "Cek", "Bot",
        "Ada orang?", "Halo min", "Hai kak", "Waalaikumsalam",
        "Halo bang", "Permisi mau tanya", "Halo kak mau nanya dong",
        "Hay", "Haloo", "Halloo", "Haii", "Heyy", "Hey",
        "Hai TanyaKito", "Halo TanyaKito", "Oi", "Woi",
        "Misi", "Misi mau tanya", "Kak mau tanya",
        "Bang mau nanya", "Min mau tanya", "Selamat datang",
        "Halo selamat pagi", "Hai selamat siang", "Pagi kak",
        "Siang kak", "Sore kak", "Malam kak", "Halo halo",
        "Hai hai", "Tes tes", "1 2 3 test", "Halo permisi",
        "Permisi kak", "Maaf mau tanya", "Kak", "Bang",
        "Numpang tanya dong", "Mau tanya nih", "Boleh tanya?",
        "Tanya dong", "Mau nanya dong kak", "Misi nanya dong",
        "Halo saya mau bertanya", "Permisi saya ingin bertanya",
        "Hai boleh tanya?", "Halo bisa tanya sesuatu?",
        "Halo apakah ada yang bisa bantu?", "Hai mau konsultasi dong",
        "Hello mau tanya", "Hay mau nanya", "Halo ada admin?",
        "Halo siapa di sini?", "Ada yang aktif?", "Bot aktif?",
        "Halo apakah botnya aktif?", "Hai kak masih aktif?",
        "Pagi min", "Siang min", "Sore min", "Malam min",
        "Assalamualaikum kak", "Assalamualaikum min",
        "Assalamualaikum mau tanya", "Bismillah mau tanya",
        "Halo kak saya turis", "Hai saya wisatawan",
        "Halo saya pengunjung baru", "Hai saya pertama kali ke Palembang",
        "Halo saya dari Jakarta mau tanya", "Hai saya baru ke Palembang",
        "Permisi saya mau jalan-jalan di Palembang",
        "Halo mau keliling Palembang nih",
    ]
    en_sentences = [
        "Hello", "Hi", "Hey", "Good morning", "Good afternoon",
        "Good evening", "Hi there", "Hey there", "Hello bot",
        "Hi chatbot", "Hello there", "Greetings", "Howdy",
        "What's up", "Yo", "Sup", "Hi, can you help me?",
        "Hello, I need help", "Hey, anyone there?",
        "Hi, I'm visiting Palembang", "Hello, I'm a tourist",
        "Hi, first time in Palembang", "Hello TanyaKito",
        "Good morning, I have a question", "Hi, I'd like to ask something",
        "Hello, is anyone available?", "Hey, can I ask a question?",
        "Hi there, I need some information", "Hello, I'm looking for help",
        "Good day", "Hi bot", "Hey bot", "Help me please",
        "I need assistance", "Can someone help me?",
        "Hello, I want to explore Palembang", "Hi, planning to visit Palembang",
    ]
    for s in id_sentences:
        sentences.append((s, "greet"))
    for s in en_sentences:
        sentences.append((s, "greet"))
    
    # Variasi tambahan dengan typo ringan
    typo_variants = [
        "Hlo", "Haloo", "Hallloo", "Haii", "Haai", "Selaamt pagi",
        "Slmat pagi", "Slmt siang", "Assalamu'alaikum", "Asalamualaikum",
        "Tess", "Tes", "Helo", "Helloo",
    ]
    for s in typo_variants:
        sentences.append((s, "greet"))
    
    return sentences


def gen_goodbye():
    """Kalimat perpisahan"""
    sentences = []
    id_list = [
        "Terima kasih", "Makasih", "Makasih ya", "Makasih banyak",
        "Terima kasih banyak", "Sampai jumpa", "Dadah", "Bye",
        "Bye bye", "Selamat tinggal", "Oke terima kasih",
        "Oke makasih", "Oke thanks", "Sudah cukup terima kasih",
        "Cukup sekian", "Oke segitu dulu", "Udah cukup",
        "Makasih infonya", "Terima kasih infonya",
        "Makasih ya infonya", "Thanks infonya",
        "Oke deh makasih", "Oke sip makasih", "Sip makasih",
        "Mantap makasih", "Oke noted makasih",
        "Baik terima kasih", "Baiklah terima kasih banyak",
        "Oke segitu aja dulu ya", "Cukup dulu ya",
        "Sampai ketemu lagi", "See you", "Bye kak",
        "Dadah kak", "Makasih kak", "Thanks kak",
        "Oke deh bye", "Sampai jumpa lagi", "Bye bye kak",
        "Terima kasih sudah membantu", "Makasih udah bantu",
        "Thanks udah bantu ya", "Oke terima kasih banyak ya",
        "Wassalamualaikum", "Waalaikumsalam", "Assalamualaikum bye",
        "Oke makasih kak sangat membantu", "Terima kasih sangat membantu",
        "Makasih ya kak sangat berguna infonya",
        "Oke noted terima kasih banyak kak",
        "Sip terima kasih kak", "Baik kak terima kasih",
        "Oke deh kak sampai jumpa", "Bye kak makasih ya",
        "Udah itu aja kak makasih", "Segitu dulu kak thanks",
        "Oke cukup kak makasih banyak", "Sampai nanti kak",
        "Nanti tanya lagi ya kak", "Makasih kak nanti saya tanya lagi",
        "Oke kak sampai jumpa", "Bye bye makasih",
        "Terima kasih atas informasinya", "Makasih atas bantuannya",
        "Thanks for the info", "Oke mantap kak",
    ]
    en_list = [
        "Thank you", "Thanks", "Thanks a lot", "Thank you very much",
        "Goodbye", "Bye", "Bye bye", "See you", "See you later",
        "That's all, thanks", "Okay thanks", "OK thank you",
        "Thanks for the help", "Thank you for the information",
        "That's enough, thank you", "Bye for now", "Take care",
        "Thanks so much", "Appreciate it", "Thanks for your help",
        "OK bye", "Alright thanks", "Got it, thanks",
        "Perfect, thank you", "Great, thanks", "Awesome thanks",
        "OK that's all I needed", "Thank you, goodbye",
        "Thanks, see you next time", "Bye, thanks for everything",
    ]
    for s in id_list:
        sentences.append((s, "goodbye"))
    for s in en_list:
        sentences.append((s, "goodbye"))
    return sentences


def gen_ask_ticket_price():
    """Pertanyaan harga tiket"""
    sentences = []
    
    # Template ID
    id_templates = [
        "Berapa harga tiket masuk {dest}?",
        "Harga tiket {dest} berapa ya?",
        "Tiket masuk {dest} berapa?",
        "Berapa biaya masuk ke {dest}?",
        "Biaya tiket {dest} berapa kak?",
        "Mau tanya harga tiket {dest}",
        "Harga masuk {dest} berapa ya kak?",
        "Berapa ya tiket {dest}?",
        "Tiket {dest} berapa sih?",
        "Kira-kira harga tiket {dest} berapa?",
        "Berapa duit masuk {dest}?",
        "Bayar berapa kalo masuk {dest}?",
        "Mau ke {dest} bayar berapa?",
        "Ke {dest} bayar berapa ya?",
        "Masuk {dest} gratis atau bayar?",
        "{dest} bayar berapa?",
        "{dest} gratis gak sih?",
        "{dest} berbayar atau gratis?",
        "Apakah {dest} gratis?",
        "Apa {dest} bayar?",
        "HTM {dest} berapa?",
        "Harga tiket masuk {dest} sekarang berapa?",
        "Update harga tiket {dest} dong",
        "Tiket masuk ke {dest} sekarang berapa ya?",
        "Mau tanya dong kak harga tiket {dest}",
        "Kak berapa harga tiket {dest}?",
        "Bang tiket {dest} berapa?",
        "Min harga masuk {dest} berapa?",
        "Permisi mau tanya harga tiket {dest}",
        "Info harga tiket {dest} dong kak",
    ]
    
    # Template EN
    en_templates = [
        "How much is the ticket to {dest}?",
        "What's the ticket price for {dest}?",
        "How much does it cost to enter {dest}?",
        "What's the entry fee for {dest}?",
        "Is {dest} free?",
        "Do I need to pay to enter {dest}?",
        "How much is the admission to {dest}?",
        "What's the entrance fee at {dest}?",
        "Ticket price for {dest}?",
        "How much to get into {dest}?",
        "Is there an entry fee for {dest}?",
        "What does it cost to visit {dest}?",
        "Price to enter {dest}?",
        "Is admission to {dest} free?",
        "How much for {dest} tickets?",
    ]
    
    # Tanpa destinasi (follow-up)
    no_dest_id = [
        "Berapa harga tiketnya?", "Harga tiket masuknya berapa?",
        "Tiketnya berapa?", "Biaya masuknya berapa?",
        "Bayar berapa?", "Gratis atau bayar?", "HTM nya berapa?",
        "Berapa ya tiketnya?", "Harga masuknya berapa kak?",
        "Kira-kira berapa ya tiketnya?", "Mahal gak tiketnya?",
        "Tiketnya mahal gak?", "Berapa duit tiketnya?",
    ]
    no_dest_en = [
        "How much is the ticket?", "What's the ticket price?",
        "How much is it?", "Is it free?", "What's the entry fee?",
        "How much does it cost?", "Price?", "Ticket price?",
    ]
    
    dest_cycle = get_dest_cycle()
    dest_cycle_en = get_dest_cycle_en()
    for t in id_templates:
        for _ in range(8):
            sentences.append((t.format(dest=next(dest_cycle)), "ask_ticket_price"))
    for t in en_templates:
        for _ in range(5):
            sentences.append((t.format(dest=next(dest_cycle_en)), "ask_ticket_price"))
    for s in no_dest_id:
        sentences.append((s, "ask_ticket_price"))
    for s in no_dest_en:
        sentences.append((s, "ask_ticket_price"))
    
    return sentences


def gen_ask_operating_hours():
    """Pertanyaan jam operasional"""
    sentences = []
    
    id_templates = [
        "Jam buka {dest} kapan?",
        "{dest} buka jam berapa?",
        "Jam operasional {dest}?",
        "{dest} buka dari jam berapa sampai jam berapa?",
        "Kapan {dest} buka?",
        "{dest} tutup jam berapa?",
        "Jam tutup {dest} kapan?",
        "{dest} buka setiap hari?",
        "Apakah {dest} buka hari Minggu?",
        "{dest} buka hari apa aja?",
        "Jadwal buka {dest} kapan ya?",
        "Mau tanya jam operasional {dest}",
        "Kak {dest} buka jam berapa ya?",
        "Info jam buka {dest} dong",
        "{dest} buka sampai malam gak?",
        "Apakah {dest} buka 24 jam?",
        "{dest} hari libur buka gak?",
        "Kalo hari Sabtu {dest} buka?",
        "Weekend {dest} buka?",
        "Weekday {dest} buka jam berapa?",
        "Pagi-pagi {dest} sudah buka belum?",
        "{dest} mulai buka jam berapa?",
        "Dari jam berapa {dest} buka?",
        "Kapan sih {dest} buka?",
        "Hari ini {dest} buka gak ya?",
    ]
    
    en_templates = [
        "What time does {dest} open?",
        "When does {dest} close?",
        "What are the opening hours of {dest}?",
        "Is {dest} open today?",
        "What are the operating hours of {dest}?",
        "Is {dest} open on weekends?",
        "What time does {dest} close?",
        "When is {dest} open?",
        "Is {dest} open on Sunday?",
        "Opening hours for {dest}?",
        "Does {dest} open every day?",
        "What days is {dest} open?",
        "Is {dest} open in the morning?",
    ]
    
    no_dest_id = [
        "Jam bukanya kapan?", "Buka jam berapa?", "Tutup jam berapa?",
        "Jam operasionalnya?", "Kapan bukanya?", "Buka setiap hari?",
        "Buka hari apa aja?", "Hari ini buka?", "Buka sampai malam?",
        "Masih buka gak ya?", "Sekarang masih buka?",
    ]
    no_dest_en = [
        "What time does it open?", "When does it close?",
        "Opening hours?", "Is it open today?", "Operating hours?",
    ]
    
    dest_cycle = get_dest_cycle()
    dest_cycle_en = get_dest_cycle_en()
    for t in id_templates:
        for _ in range(8):
            sentences.append((t.format(dest=next(dest_cycle)), "ask_operating_hours"))
    for t in en_templates:
        for _ in range(5):
            sentences.append((t.format(dest=next(dest_cycle_en)), "ask_operating_hours"))
    for s in no_dest_id:
        sentences.append((s, "ask_operating_hours"))
    for s in no_dest_en:
        sentences.append((s, "ask_operating_hours"))
    
    return sentences


def gen_ask_location_access():
    """Pertanyaan lokasi dan akses"""
    sentences = []
    
    id_templates = [
        "Dimana lokasi {dest}?",
        "{dest} di mana ya?",
        "Alamat {dest} dimana?",
        "Lokasi {dest} dimana kak?",
        "Bagaimana cara ke {dest}?",
        "Cara menuju {dest} gimana?",
        "Naik apa ke {dest}?",
        "Dari mana ke {dest}?",
        "{dest} lokasinya dimana?",
        "Alamat lengkap {dest}?",
        "{dest} ada di daerah mana?",
        "Arah ke {dest} gimana?",
        "Mau ke {dest} lewat mana?",
        "Rute ke {dest} gimana?",
        "Dari pusat kota ke {dest} berapa lama?",
        "{dest} jauh gak dari pusat kota?",
        "Bisa naik angkot ke {dest}?",
        "{dest} deket stasiun LRT mana?",
        "Akses ke {dest} mudah gak?",
        "{dest} bisa ditempuh pakai apa?",
        "Kak dimana ya {dest}?",
        "Min alamat {dest} apa ya?",
        "Info lokasi {dest} dong",
        "Tolong kasih tau alamat {dest}",
        "Posisi {dest} dimana ya?",
    ]
    
    en_templates = [
        "Where is {dest} located?",
        "What's the address of {dest}?",
        "How do I get to {dest}?",
        "Where is {dest}?",
        "How to reach {dest}?",
        "What's the location of {dest}?",
        "Is {dest} far from the city center?",
        "How to go to {dest}?",
        "Directions to {dest}?",
        "Can I take public transport to {dest}?",
        "Address of {dest}?",
        "Location of {dest}?",
    ]
    
    no_dest_id = [
        "Dimana lokasinya?", "Alamatnya dimana?", "Cara ke sana gimana?",
        "Lokasinya dimana ya?", "Alamat lengkapnya?", "Di daerah mana?",
        "Jauh gak dari sini?", "Bisa naik apa ke sana?",
        "Akses ke sana gimana?", "Lewat mana ya?",
    ]
    no_dest_en = [
        "Where is it?", "What's the address?", "How do I get there?",
        "Where is it located?", "How far is it?", "Directions?",
    ]
    
    dest_cycle = get_dest_cycle()
    dest_cycle_en = get_dest_cycle_en()
    for t in id_templates:
        for _ in range(8):
            sentences.append((t.format(dest=next(dest_cycle)), "ask_location_access"))
    for t in en_templates:
        for _ in range(5):
            sentences.append((t.format(dest=next(dest_cycle_en)), "ask_location_access"))
    for s in no_dest_id:
        sentences.append((s, "ask_location_access"))
    for s in no_dest_en:
        sentences.append((s, "ask_location_access"))
    
    return sentences


def gen_ask_facilities():
    """Pertanyaan fasilitas"""
    sentences = []
    
    id_templates = [
        "Fasilitas apa saja di {dest}?",
        "{dest} fasilitasnya apa aja?",
        "Ada fasilitas apa di {dest}?",
        "Fasilitas {dest} lengkap gak?",
        "{dest} ada toilet gak?",
        "{dest} ada tempat parkir?",
        "Ada mushola di {dest}?",
        "{dest} ada kantin gak?",
        "Fasilitas umum di {dest} apa aja?",
        "Ada wifi di {dest}?",
        "{dest} ramah anak gak?",
        "{dest} ada area bermain anak?",
        "Tempat makan di sekitar {dest} ada?",
        "{dest} ada spot foto gak?",
        "Fasilitas pendukung {dest}?",
        "{dest} ada tempat istirahat?",
        "Parkiran {dest} luas gak?",
        "Mau tanya fasilitas di {dest}",
        "Kak fasilitas {dest} apa aja ya?",
        "Info fasilitas {dest} dong",
    ]
    
    en_templates = [
        "What facilities are available at {dest}?",
        "Does {dest} have parking?",
        "Are there toilets at {dest}?",
        "Does {dest} have a mosque?",
        "What amenities does {dest} offer?",
        "Is {dest} child-friendly?",
        "Facilities at {dest}?",
        "Does {dest} have food stalls?",
        "Is there WiFi at {dest}?",
        "What's available at {dest}?",
    ]
    
    no_dest_id = [
        "Fasilitasnya apa aja?", "Ada fasilitas apa?", "Lengkap gak fasilitasnya?",
        "Ada toilet gak?", "Ada parkiran?", "Ada mushola?",
        "Fasilitasnya gimana?", "Apa aja fasilitas di sana?",
    ]
    no_dest_en = [
        "What facilities are there?", "What amenities are available?",
        "Is there parking?", "Are there restrooms?", "Facilities?",
    ]
    
    dest_cycle = get_dest_cycle()
    dest_cycle_en = get_dest_cycle_en()
    for t in id_templates:
        for _ in range(8):
            sentences.append((t.format(dest=next(dest_cycle)), "ask_facilities"))
    for t in en_templates:
        for _ in range(5):
            sentences.append((t.format(dest=next(dest_cycle_en)), "ask_facilities"))
    for s in no_dest_id:
        sentences.append((s, "ask_facilities"))
    for s in no_dest_en:
        sentences.append((s, "ask_facilities"))
    
    return sentences


def gen_ask_destination_info():
    """Pertanyaan info/deskripsi destinasi"""
    sentences = []
    
    id_templates = [
        "Info tentang {dest}",
        "Ceritakan tentang {dest}",
        "Apa itu {dest}?",
        "{dest} itu apa?",
        "Jelaskan tentang {dest}",
        "Deskripsi {dest}",
        "Mau tau tentang {dest}",
        "Kasih tau dong tentang {dest}",
        "{dest} itu tempat apa sih?",
        "Apa yang menarik dari {dest}?",
        "Kenapa {dest} terkenal?",
        "{dest} terkenal karena apa?",
        "Sejarah {dest} apa ya?",
        "Cerita sejarah {dest} dong",
        "Apa keunikan {dest}?",
        "Review tentang {dest}",
        "Gimana {dest}?",
        "Kayak apa sih {dest}?",
        "Seru gak {dest}?",
        "Menarik gak {dest}?",
        "Info lengkap {dest} dong",
        "Kasih info {dest} ya kak",
        "Mau tau sejarah {dest}",
        "Ceritain {dest} dong kak",
        "{dest} bagus gak?",
        "di dalem situ ada apa aja sih?",
        "ceritain dong kenapa tempat ini viral",
        "apa yang bikin {dest} ini spesial?",
        "kenapa orang-orang suka ke {dest}?",
        "di {dest} itu kita bisa ngapain aja?",
        "ada apa aja sih di {dest}?",
        "dulu tempat ini bekas apa?",
        "asal usul {dest} ini gimana ceritanya?",
    ]
    
    en_templates = [
        "Tell me about {dest}",
        "What is {dest}?",
        "Information about {dest}",
        "Describe {dest}",
        "What's special about {dest}?",
        "Why is {dest} famous?",
        "History of {dest}?",
        "What can I see at {dest}?",
        "Is {dest} worth visiting?",
        "What makes {dest} unique?",
        "Details about {dest}?",
        "What's {dest} like?",
    ]
    
    no_dest_id = [
        "Ceritakan tentangnya", "Info lebih lanjut dong", "Apa itu?",
        "Jelaskan dong", "Kasih tau deskripsinya", "Sejarahnya gimana?",
        "Menarik gak?", "Bagus gak tempatnya?", "Kayak apa sih tempatnya?",
    ]
    no_dest_en = [
        "Tell me more", "What is it?", "More information please",
        "Describe it", "What's it like?", "Is it worth visiting?",
    ]
    
    dest_cycle = get_dest_cycle()
    dest_cycle_en = get_dest_cycle_en()
    for t in id_templates:
        for _ in range(8):
            sentences.append((t.format(dest=next(dest_cycle)), "ask_destination_info"))
    for t in en_templates:
        for _ in range(5):
            sentences.append((t.format(dest=next(dest_cycle_en)), "ask_destination_info"))
    for s in no_dest_id:
        sentences.append((s, "ask_destination_info"))
    for s in no_dest_en:
        sentences.append((s, "ask_destination_info"))
    
    return sentences


def gen_ask_lrt_destinations():
    """Pertanyaan tentang destinasi yang bisa dijangkau LRT"""
    sentences = []
    
    id_list = [
        "Wisata apa yang bisa dijangkau LRT?",
        "Destinasi yang dekat stasiun LRT?",
        "Mau naik LRT, wisata apa yang dekat?",
        "Tempat wisata yang bisa diakses LRT?",
        "Wisata yang aksesnya lewat LRT apa aja?",
        "Ada wisata dekat LRT gak?",
        "Stasiun LRT terdekat ke tempat wisata mana?",
        "Wisata yang bisa pake LRT?",
        "Naik LRT bisa ke wisata mana aja?",
        "LRT Palembang lewat wisata mana?",
        "Tempat wisata yang dilalui LRT?",
        "Wisata accessible by LRT?",
        "Mau keliling Palembang naik LRT, kemana aja?",
        "Rute LRT melewati wisata apa?",
        "Stasiun LRT dekat tempat wisata?",
        "Bisa naik LRT ke BKB?",
        "Apakah Ampera dekat stasiun LRT?",
        "LRT berhenti dekat wisata mana?",
        "Wisata yang gampang dijangkau LRT?",
        "Kak wisata yang bisa naik LRT apa aja?",
        "Min info wisata dekat LRT dong",
        "Mau jalan-jalan pakai LRT bisa kemana?",
        "Destinasi wisata LRT Palembang?",
        "Wisata yang paling dekat LRT?",
        "Ada gak wisata yang bisa ditempuh LRT?",
        "Tempat-tempat wisata yang aksesibel LRT?",
        "Wisata apa aja yang dekat jalur LRT?",
        "Mau explore Palembang pakai LRT, kemana aja bisa?",
        "Rekomendasi wisata yang dekat stasiun LRT",
        "Wisata murah yang bisa naik LRT?",
        "Dari stasiun LRT bisa ke wisata apa?",
        "LRT Palembang stop di wisata mana?",
        "Apakah ada wisata yang bisa dijangkau LRT?",
        "Tolong kasih tau wisata dekat LRT",
        "Info wisata yang terjangkau LRT",
        "Wisata Palembang yang LRT friendly?",
        "Destinasi yang LRT accessible?",
        "Kemana aja bisa naik LRT untuk wisata?",
        "Apa {dest} bisa dijangkau LRT?".format(dest="BKB"),
        "Apakah {dest} dekat LRT?".format(dest="Ampera"),
        "Bisa naik LRT ke {dest}?".format(dest="Kambang Iwak"),
        "{dest} aksesnya lewat LRT bisa?".format(dest="Jakabaring"),
        "LRT bisa ke {dest} gak?".format(dest="Monpera"),
        "{dest} dekat stasiun LRT mana?".format(dest="SMB"),
    ]
    
    en_list = [
        "Which tourist spots are near LRT stations?",
        "Can I reach tourist places by LRT?",
        "What destinations are accessible by LRT?",
        "LRT accessible tourist spots in Palembang?",
        "Where can I go by LRT for sightseeing?",
        "Are there tourist attractions near LRT stations?",
        "Which places can I visit using LRT?",
        "Tourist spots along LRT route?",
        "Can I take LRT to visit tourist places?",
        "LRT friendly tourist destinations?",
        "Is BKB near an LRT station?",
        "Can I take LRT to Ampera Bridge?",
        "What tourism spots are on the LRT line?",
        "Palembang LRT tourism destinations?",
        "Best places to visit by LRT?",
    ]
    
    for s in id_list:
        sentences.append((s, "ask_lrt_destinations"))
    for s in en_list:
        sentences.append((s, "ask_lrt_destinations"))
    
    return sentences


def gen_ask_recommendation():
    """Permintaan rekomendasi wisata"""
    sentences = []
    
    id_list = [
        "Rekomendasi wisata Palembang dong",
        "Rekomendasikan tempat wisata di Palembang",
        "Ada rekomendasi wisata gak?",
        "Mau minta saran wisata Palembang",
        "Saran tempat wisata di Palembang?",
        "Wisata apa yang recommended di Palembang?",
        "Tempat wisata yang wajib dikunjungi di Palembang?",
        "Rekomendasikan destinasi wisata Palembang",
        "Kasih rekomendasi wisata dong kak",
        "Min rekomendasi wisata Palembang apa?",
        "Wisata terbaik di Palembang apa?",
        "Top wisata Palembang?",
        "Wisata paling bagus di Palembang?",
        "Tempat paling menarik di Palembang?",
        "Mau jalan-jalan di Palembang kemana ya?",
        "Enaknya kemana ya di Palembang?",
        "Palembang punya wisata apa aja?",
        "Ada wisata apa aja di Palembang?",
        "Saranin wisata di Palembang dong",
        "Bisa rekomendasikan tempat wisata?",
        "Mau liburan di Palembang kemana ya?",
        "Wisata Palembang yang populer apa?",
        "Destinasi favorit di Palembang?",
        "Wisata yang paling sering dikunjungi?",
        "Tempat wisata ikonik Palembang?",
        "Wisata instagramable Palembang?",
        "Wisata yang cocok untuk keluarga?",
        "Wisata romantis di Palembang?",
        "Wisata untuk anak-anak di Palembang?",
        "Tempat wisata murah di Palembang?",
        "Wisata gratis di Palembang ada gak?",
        "Wisata malam di Palembang kemana?",
        "Wisata sore hari di Palembang?",
        "Wisata weekend di Palembang?",
        "Mau cari wisata yang seru di Palembang",
        "Wisata apa yang cocok untuk foto-foto?",
        "Rekomendasi wisata untuk liburan singkat",
        "Wisata yang bisa dikunjungi sehari",
        "Kak rekomendasiin dong wisata di Palembang",
        "Min tolong saran wisata dong",
        "anakku suka main air, enaknya bawa kemana ya?",
        "aku lagi pengen makan enak sambil liat sungai",
        "bingung mau bawa keluarga jalan-jalan kemana",
        "ada ide tempat buat nongkrong sore?",
        "tempat yang bagus buat foto-foto keluarga di mana ya?",
        "aku butuh tempat healing yang tenang",
        "lagi pengen liat pemandangan kota dari atas",
    ]
    
    en_list = [
        "Recommend tourist places in Palembang",
        "What are the best places to visit in Palembang?",
        "Top tourist attractions in Palembang?",
        "Must-visit places in Palembang?",
        "Suggest tourist spots in Palembang",
        "Best things to do in Palembang?",
        "What should I visit in Palembang?",
        "Popular tourist spots in Palembang?",
        "Recommend places to see in Palembang",
        "Where should I go in Palembang?",
        "Tourist recommendations for Palembang?",
        "Best destinations in Palembang?",
        "What are the iconic places in Palembang?",
        "Suggest places for sightseeing in Palembang",
        "Famous places in Palembang?",
        "Things to do in Palembang?",
        "Best places for tourists in Palembang?",
        "What to see in Palembang?",
        "Where to go in Palembang?",
        "Palembang tourism recommendations?",
    ]
    
    for s in id_list:
        sentences.append((s, "ask_recommendation"))
    for s in en_list:
        sentences.append((s, "ask_recommendation"))
    
    return sentences


def gen_ask_category():
    """Pertanyaan wisata berdasarkan kategori"""
    sentences = []
    
    for cat in CATEGORIES_ID:
        id_temps = [
            f"Wisata {cat} di Palembang apa aja?",
            f"Ada {cat} di Palembang?",
            f"Rekomendasi {cat} Palembang",
            f"Tempat {cat} di Palembang?",
            f"Mau cari {cat} di Palembang",
            f"Info {cat} Palembang dong",
            f"Kak ada {cat} gak di Palembang?",
            f"Destinasi {cat} Palembang?",
            f"Yang termasuk {cat} apa aja?",
            f"Palembang punya {cat} apa?",
        ]
        for s in id_temps:
            sentences.append((s, "ask_category"))
    
    for cat in CATEGORIES_EN:
        en_temps = [
            f"What {cat} are in Palembang?",
            f"Any {cat} in Palembang?",
            f"Recommend {cat} in Palembang",
            f"{cat} places in Palembang?",
            f"Where can I find {cat} in Palembang?",
            f"List of {cat} in Palembang?",
        ]
        for s in en_temps:
            sentences.append((s, "ask_category"))
    
    extra_id = [
        "Wisata alam di Palembang ada apa aja?",
        "Mau ke tempat bersejarah di Palembang",
        "Ada wisata kuliner gak di Palembang?",
        "Wisata religi Palembang apa aja?",
        "Tempat wisata budaya Palembang?",
        "Taman di Palembang ada apa aja?",
        "Museum di Palembang apa aja?",
        "Masjid bersejarah di Palembang?",
        "Wisata air di Palembang ada?",
        "Wisata sungai Palembang?",
        "Wisata edukasi di Palembang?",
        "Wisata outdoor Palembang?",
        "Wisata indoor Palembang?",
        "Destinasi kategori alam?",
        "Tempat bersejarah apa aja?",
        "Ada taman kota di Palembang?",
        "Wisata yang bernuansa Islami?",
        "Tempat wisata bertema budaya Melayu?",
        "Wisata Tionghoa di Palembang?",
        "Wisata Arab di Palembang?",
    ]
    extra_en = [
        "Nature spots in Palembang?",
        "Historical places in Palembang?",
        "Food tourism in Palembang?",
        "Religious sites in Palembang?",
        "Cultural attractions in Palembang?",
        "Parks in Palembang?",
        "Museums in Palembang?",
    ]
    
    for s in extra_id:
        sentences.append((s, "ask_category"))
    for s in extra_en:
        sentences.append((s, "ask_category"))
    
    return sentences


def gen_ask_hidden_gems():
    """Pertanyaan hidden gems"""
    sentences = []
    
    id_list = [
        "Hidden gems di Palembang apa?",
        "Wisata tersembunyi di Palembang?",
        "Tempat wisata yang jarang dikunjungi di Palembang?",
        "Ada hidden gems gak di Palembang?",
        "Wisata yang belum banyak orang tau di Palembang?",
        "Tempat wisata unik di Palembang?",
        "Wisata anti mainstream Palembang?",
        "Destinasi rahasia Palembang?",
        "Wisata off the beaten path Palembang?",
        "Tempat yang belum terkenal di Palembang?",
        "Wisata yang belum viral di Palembang?",
        "Tempat keren tapi belum terkenal di Palembang?",
        "Wisata sepi tapi bagus di Palembang?",
        "Hidden gem Palembang yang recommended?",
        "Ada wisata yang jarang orang tau?",
        "Mau ke tempat yang unik dan sepi",
        "Wisata yang belum mainstream?",
        "Tempat wisata yang gak rame?",
        "Rekomendasi hidden gems dong kak",
        "Kak ada hidden gems gak?",
        "Min kasih tau hidden gems Palembang",
        "Wisata Palembang yang belum banyak orang kunjungi?",
        "Ada tempat wisata baru di Palembang?",
        "Wisata baru yang belum terkenal?",
        "Tempat wisata alternatif Palembang?",
        "Destinasi tersembunyi Palembang ada gak?",
        "Wisata yang gak pasaran di Palembang?",
        "Tempat yang underrated di Palembang?",
        "Wisata Palembang yang aesthetic tapi sepi?",
        "Ada gak wisata yang masih asri di Palembang?",
    ]
    
    en_list = [
        "Hidden gems in Palembang?",
        "Off the beaten path places in Palembang?",
        "Underrated tourist spots in Palembang?",
        "Secret places in Palembang?",
        "Lesser known attractions in Palembang?",
        "Unique places to visit in Palembang?",
        "Non-touristy places in Palembang?",
        "Hidden spots in Palembang?",
        "Undiscovered places in Palembang?",
        "Alternative tourist spots in Palembang?",
        "Any hidden gems to visit?",
        "Places that tourists don't usually go?",
        "Quiet but beautiful places in Palembang?",
        "Unexplored attractions in Palembang?",
        "Local favorites in Palembang?",
    ]
    
    for s in id_list:
        sentences.append((s, "ask_hidden_gems"))
    for s in en_list:
        sentences.append((s, "ask_hidden_gems"))
    
    return sentences


def gen_provide_feedback():
    """Kalimat feedback/apresiasi"""
    sentences = []
    
    id_list = [
        "Bagus infonya", "Mantap", "Keren", "Oke sip",
        "Informasinya membantu", "Sangat membantu", "Berguna banget",
        "Terima kasih infonya sangat berguna", "Wah keren infonya",
        "Suka sama jawabannya", "Jawabannya lengkap", "Oke bagus",
        "Infonya akurat", "Mantap kak", "Keren banget infonya",
        "Membantu banget", "Jelas banget penjelasannya", "Oke paham",
        "Noted kak", "Siap kak", "Oke kak noted",
        "Makasih infonya kak", "Membantu banget kak",
        "Wah makasih infonya lengkap", "Oke sip mantap",
        "Infonya oke banget", "Lengkap banget infonya",
        "Penjelasannya bagus kak", "Jawabannya memuaskan",
        "Bot nya pintar ya", "Chatbot nya bagus",
        "Aplikasinya keren", "Suka sama aplikasinya",
        "Infonya detail banget", "Jawaban yang sangat membantu",
        "Oke terima kasih jawabannya", "Sip lanjut kak",
        "Jawaban yang memuaskan", "Oke mantul", "Mantul kak",
        "Good info", "Nice info", "Nais", "Jos",
        "Keren sih ini botnya", "Canggih botnya",
        "Pinter ya botnya", "Jawabannya cepat",
        "Responnya cepat ya", "Informatif banget",
        "5 bintang deh", "Rating 5 untuk botnya",
        "Recommended banget aplikasinya", "Suka deh",
    ]
    
    en_list = [
        "Great info", "Very helpful", "Thanks for the info",
        "That's helpful", "Good answer", "Nice", "Perfect",
        "Exactly what I needed", "Very informative",
        "Useful information", "Great bot", "Smart bot",
        "Amazing response", "Very detailed", "Good job",
        "Well explained", "Clear explanation", "Understood",
        "Got it", "That makes sense", "Wonderful",
        "Excellent information", "5 stars", "Love this bot",
        "The answer is very detailed", "Very accurate",
    ]
    
    for s in id_list:
        sentences.append((s, "provide_feedback"))
    for s in en_list:
        sentences.append((s, "provide_feedback"))
    
    return sentences


def gen_ask_unrelated():
    """Pertanyaan di luar domain wisata Palembang"""
    sentences = []
    
    id_list = [
        "Siapa presiden Indonesia?", "Cuaca hari ini gimana?",
        "Resep nasi goreng dong", "Cara masak rendang gimana?",
        "Berapa harga iPhone terbaru?", "Kapan piala dunia?",
        "Siapa pemenang Liga Champions?", "Kurs dollar hari ini berapa?",
        "Cara daftar CPNS gimana?", "Lowongan kerja di Palembang?",
        "Bagaimana cara diet?", "Tips menurunkan berat badan?",
        "Resep pempek palembang", "Cara bikin cuko pempek",
        "Apa itu cryptocurrency?", "Cara main saham?",
        "Kenapa langit biru?", "Kapan kiamat?",
        "Siapa penemu lampu?", "Rumus pythagoras?",
        "Ceritakan tentang dinosaurus", "Apa itu AI?",
        "Bagaimana cara belajar coding?", "Tips belajar bahasa Inggris?",
        "Film terbaru 2024 apa?", "Anime yang bagus apa?",
        "Lagu hits 2024 apa?", "Siapa BTS?",
        "Cara menghilangkan jerawat?", "Tips perawatan kulit?",
        "Apa obat sakit kepala?", "Cara mengobati flu?",
        "Jadwal kereta api Jakarta?", "Tiket pesawat ke Bali berapa?",
        "Wisata Bali yang bagus?", "Hotel murah di Jakarta?",
        "Universitas terbaik Indonesia?", "Cara daftar kuliah?",
        "Apa itu machine learning?", "Python atau Java lebih bagus?",
        "Bagaimana cara memasak telur?", "Resep kue bolu?",
        "Harga emas hari ini?", "Prediksi cuaca besok?",
        "Siapa gubernur Palembang?", "Jadwal sholat hari ini?",
        "Kode pos Palembang berapa?", "Rumah sakit terdekat?",
        "Apotek 24 jam di Palembang?", "Taxi online di Palembang?",
        "Aku lagi sedih nih", "Aku bosan", "Kamu siapa?",
        "Kamu robot ya?", "Apakah kamu AI?", "Kamu punya perasaan?",
        "Ceritakan joke dong", "Kasih pantun dong",
        "Nyanyiin lagu dong", "Gambar kucing dong",
        "Apa arti cinta?", "Bagaimana cara move on?",
        "Curhatan aku dong", "Aku mau cerita",
        "Halo kamu cantik", "Kamu pintar ya", "Kamu bodoh",
        "1+1 berapa?", "Akar kuadrat dari 144?",
        "Translate bahasa Inggris dong", "Apa bahasa Inggrisnya kucing?",
        "kamu bot beneran ya?",
        "kok jawabannya lambat",
        "ini jawaban template ya?",
        "siapa yang bikin kamu?",
        "kamu bisa jawab apa aja sih?",
        "kamu manusia apa robot?",
        "aku lagi ngomong sama komputer ya?",
        "adminnya mana?",
    ]
    
    en_list = [
        "Who is the president of Indonesia?", "What's the weather today?",
        "How to cook fried rice?", "What's the price of iPhone?",
        "When is the World Cup?", "What's cryptocurrency?",
        "How to learn programming?", "What's the meaning of life?",
        "Tell me a joke", "Who invented electricity?",
        "What's 2+2?", "How to lose weight?",
        "Best movies of 2024?", "What's the capital of France?",
        "How to make money online?", "What's Bitcoin?",
        "Tell me about dinosaurs", "What is AI?",
        "How to cook an egg?", "Recipe for chocolate cake?",
        "Flight tickets to Bali?", "Hotels in Jakarta?",
        "Best universities in Indonesia?", "How to get a visa?",
        "What's the time?", "I'm bored", "Who are you?",
        "Are you a robot?", "Do you have feelings?",
        "Sing me a song", "Tell me a story",
        "What's love?", "How to move on?",
    ]
    
    for s in id_list:
        sentences.append((s, "ask_unrelated"))
    for s in en_list:
        sentences.append((s, "ask_unrelated"))
    
    return sentences


def swap_destination_in_text(text):
    """Ganti entitas destinasi dalam kalimat dengan destinasi acak dari ALL_DEST"""
    for ent in sorted(ALL_DEST, key=len, reverse=True):
        pattern = r'\b' + re.escape(ent) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            new_ent = random.choice(ALL_DEST)
            return re.sub(pattern, new_ent, text, flags=re.IGNORECASE)
    return text

def clean_stacked_fillers(text):
    """Membersihkan filler bertumpuk, duplikasi filler, dan tanda baca di tengah filler"""
    t = text
    # Fix repeated consecutive words (e.g. "sih sih" -> "sih", "ya ya" -> "ya")
    t = re.sub(r'\b(sih|ya|dong|deh|nih|kak|min|bang)\s+\1\b', r'\1', t, flags=re.IGNORECASE)
    # Fix ? or . followed by trailing filler (e.g. "? dong" -> "?", "? ya kak" -> "?")
    t = re.sub(r'([\?\!\.])\s*(ya|ya kak|dong|dong kak|nih|sih|deh|please|thanks)[\?\.\!]?$', r'\1', t, flags=re.IGNORECASE)
    # Fix trailing double fillers (e.g. "sih deh" -> "sih")
    t = re.sub(r'\b(sih|ya|dong|deh|nih)\s+(sih|ya|dong|deh|nih)\b', r'\1', t, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t).strip()

def balance_dataset(all_sentences, target_per_class=615):
    """Seimbangkan jumlah kalimat per intent dengan variasi cerdas & distribusi merata (round-robin)"""
    from collections import defaultdict
    import random
    
    by_label = defaultdict(list)
    for text, label in all_sentences:
        by_label[label].append((clean_stacked_fillers(text), label))
    
    prefixes_id = ["Kak ", "Min ", "Bang ", "Permisi ", "Halo ", "Maaf mau tanya ", "Mau nanya dong ", "Eh ", ""]
    suffixes_id = ["", " ya", " ya kak", " dong", " dong kak", " nih", " sih", " deh"]
    prefixes_en = ["Hey ", "Excuse me, ", "Hi, ", "Hello, ", "Please ", "Can you tell me ", "I want to know ", ""]
    suffixes_en = ["", " please", " thanks", "?"]
    
    en_words = {"what", "where", "how", "when", "is", "are", "do", "can", "tell", "the", "to", "in", "of", "for", "at", "it", "i", "you", "this", "that", "not", "with", "have", "has", "an", "a"}
    
    balanced = []
    for label, items in sorted(by_label.items()):
        unique_texts = list(dict.fromkeys(clean_stacked_fillers(t) for t, _ in items))
        random.shuffle(unique_texts)
        
        unique_tuples = [(t, label) for t in unique_texts]
        
        if len(unique_tuples) >= target_per_class:
            balanced.extend(random.sample(unique_tuples, target_per_class))
        else:
            balanced.extend(unique_tuples)
            remaining = target_per_class - len(unique_tuples)
            existing_texts = set(unique_texts)
            extra = []
            
            candidates_per_text = []
            for text in unique_texts:
                text_vars = []
                words_in_text = set(text.lower().split())
                is_en = len(words_in_text & en_words) >= 2
                prefixes = prefixes_en if is_en else prefixes_id
                suffixes = suffixes_en if is_en else suffixes_id
                
                bases = [text]
                if text.lower() != text:
                    bases.append(text.lower())
                if text.endswith("?"):
                    bases.append(text[:-1].strip())
                elif not text.endswith((".", "!", "?")):
                    bases.append(text + "?")
                
                for base in bases:
                    for pfx in prefixes:
                        for sfx in suffixes:
                            variant = clean_stacked_fillers(pfx + base + sfx)
                            variant_swapped = swap_destination_in_text(variant)
                            if variant_swapped and variant_swapped not in existing_texts and variant_swapped not in text_vars:
                                text_vars.append(variant_swapped)
                random.shuffle(text_vars)
                candidates_per_text.append(text_vars)
            
            # Interleave variasi secara ROUND-ROBIN antar semua teks dasar
            var_idx = 0
            while len(extra) < remaining:
                added_any = False
                for cand_list in candidates_per_text:
                    if var_idx < len(cand_list):
                        v = cand_list[var_idx]
                        if v not in existing_texts:
                            extra.append((v, label))
                            existing_texts.add(v)
                            added_any = True
                            if len(extra) >= remaining:
                                break
                var_idx += 1
                if not added_any:
                    break
                    
            balanced.extend(extra[:remaining])
            
    return balanced


def main():
    print("Generating dataset Intent v2 (bilingual, 13 intents)...")
    
    all_data = []
    generators = [
        ("greet", gen_greet),
        ("goodbye", gen_goodbye),
        ("ask_ticket_price", gen_ask_ticket_price),
        ("ask_operating_hours", gen_ask_operating_hours),
        ("ask_location_access", gen_ask_location_access),
        ("ask_facilities", gen_ask_facilities),
        ("ask_destination_info", gen_ask_destination_info),
        ("ask_lrt_destinations", gen_ask_lrt_destinations),
        ("ask_recommendation", gen_ask_recommendation),
        ("ask_category", gen_ask_category),
        ("ask_hidden_gems", gen_ask_hidden_gems),
        ("provide_feedback", gen_provide_feedback),
        ("ask_unrelated", gen_ask_unrelated),
    ]
    
    for name, gen_fn in generators:
        data = gen_fn()
        print(f"  {name}: {len(data)} kalimat (sebelum balancing)")
        all_data.extend(data)
    
    print(f"\nTotal sebelum balancing: {len(all_data)}")
    
    # Balance ke ~615 per kelas × 13 = ~8000
    balanced = balance_dataset(all_data, target_per_class=615)
    
    # Shuffle
    random.shuffle(balanced)
    
    # Hapus duplikat final
    seen = set()
    final = []
    for text, label in balanced:
        key = (text.strip(), label)
        if key not in seen:
            seen.add(key)
            final.append((text.strip(), label))
    
    # Simpan
    output_path = "ml/data/raw/intents_bilingual_v2.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in final:
            writer.writerow([text, label])
    
    # Print statistik
    from collections import Counter
    counts = Counter(label for _, label in final)
    print(f"\nTotal final: {len(final)} kalimat")
    print("\nDistribusi per intent:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    
    print(f"\nDataset disimpan di: {output_path}")


if __name__ == "__main__":
    main()
