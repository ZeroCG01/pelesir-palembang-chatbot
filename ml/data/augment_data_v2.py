import csv
import re
import random
import os

ID_DIALECT_MAP = {
    r'\bapa\b': ['apo', 'apa'],
    r'\btidak\b': ['dak', 'gak', 'enggak', 'ndak'],
    r'\bbagaimana\b': ['cakmano', 'gimana'],
    r'\bsangat\b': ['nian', 'banget', 'sekali'],
    r'\bkamu\b': ['kau', 'kamu'],
    r'\baku\b': ['aku', 'ambo', 'saya'],
    r'\bbagus\b': ['mantap', 'keren', 'bagus'],
    r'\bke\b': ['ke', 'k', 'ka'],
    r'\bdari\b': ['dari', 'dar'],
    r'\bdimana\b': ['dimano', 'dimana', 'dmana'],
    r'\bmakan\b': ['makan', 'maman'],
    r'\benak\b': ['lemak', 'enak'],
    r'\bsaudara\b': ['kakak', 'kak', 'mang', 'mamang', 'bik'],
    r'\bbisa\b': ['pacak', 'bisa', 'bis'],
    r'\bharga\b': ['hargo', 'harga'],
    r'\bberapa\b': ['berapo', 'berapa', 'brapa'],
    r'\bkenapa\b': ['ngapo', 'kenapa', 'knp'],
    r'\btapi\b': ['tapi', 'tp'],
    r'\bkalau\b': ['kalo', 'kalu', 'men'],
    r'\bsaja\b': ['bae', 'aja'],
    r'\bya\b': ['yo', 'ya'],
    r'\btidak ada\b': ['katek', 'dak katek', 'ga ada'],
    r'\bbohong\b': ['bota', 'bohong'],
    r'\bbodoh\b': ['bengak', 'bodo'],
    r'\bbelum\b': ['belom', 'blom'],
    r'\bsudah\b': ['sudah', 'dah', 'udah'],
    r'\bpergi\b': ['pegi', 'pergi'],
    r'\blihat\b': ['jingok', 'lihat']
}

EN_SLANG_MAP = {
    r'\bwhat is\b': ['whats', 'what s'],
    r'\bdo not\b': ['dont', 'don t'],
    r'\byou\b': ['u', 'ya'],
    r'\bare\b': ['r'],
    r'\bcannot\b': ['cant'],
    r'\bcan not\b': ['cant'],
    r'\bwhere is\b': ['wheres'],
    r'\bthere is\b': ['theres'],
    r'\bit is\b': ['its', 'it s'],
    r'\bthat is\b': ['thats'],
    r'\bhow is\b': ['hows'],
    r'\bi am\b': ['im', 'i m'],
    r'\bwill not\b': ['wont'],
    r'\bhave to\b': ['gotta'],
    r'\bgoing to\b': ['gonna'],
    r'\bwant to\b': ['wanna']
}

TRAIN_ENTITIES = [
    "BKB", "Benteng Kuto Besak", "Ampera", "Jembatan Ampera", "Kambang Iwak", 
    "Pulau Kemaro", "Bukit Siguntang", "Museum Balaputra Dewa", "Al Quran Al Akbar",
    "Masjid Agung", "Punti Kayu", "Kampung Kapitan", "Kampung Arab", "Monpera"
]

OOD_ENTITIES = [
    "Candi Borobudur", "Pantai Kuta", "Central Park", "Monas", "Gunung Bromo",
    "Danau Toba", "Raja Ampat", "Menara Eiffel", "Universal Studio",
    "Candi Prambanan", "Malioboro", "Ancol", "Taman Safari", "Tanah Lot"
]

def replace_entities(text):
    for entity in TRAIN_ENTITIES:
        # Search for entity. Use \b if it's alphanumeric.
        pattern = r'\b' + re.escape(entity) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            if random.random() < 0.5: # 50% chance to swap entity
                text = re.sub(pattern, random.choice(OOD_ENTITIES), text, flags=re.IGNORECASE)
    return text

def drop_vowels(text, prob=0.1):
    vowels = 'aeiouAEIOU'
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] in vowels and random.random() < prob:
            chars[i] = ''
    return "".join(chars)

def inject_dialect(text):
    def repl_func(replacements):
        def inner_repl(match):
            word = random.choice(replacements)
            if match.group(0).isupper():
                return word.upper()
            elif match.group(0).istitle():
                return word.title()
            else:
                return word
        return inner_repl

    for pattern, replacements in ID_DIALECT_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, repl_func(replacements), text, flags=re.IGNORECASE)
            
    for pattern, replacements in EN_SLANG_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, repl_func(replacements), text, flags=re.IGNORECASE)
            
    return text

def augment_data(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    augmented_rows = []
    
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            augmented_rows.append(row) # Keep original
            
            # Augmentation logic
            original_text = row['text']
            
            # 1. Entity Swapping
            swapped_text = replace_entities(original_text)
            
            # 2. Dialect Injection
            dialect_text = inject_dialect(swapped_text)
            
            # 3. Typo simulation (vowel drop)
            if random.random() < 0.3: # only 30% of data gets typos
                final_text = drop_vowels(dialect_text, prob=0.15)
            else:
                final_text = dialect_text
                
            if final_text != original_text:
                new_row = row.copy()
                new_row['text'] = final_text
                augmented_rows.append(new_row)
                
    with open(output_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmented_rows)
        
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(current_dir, "raw", "intents_bilingual.csv")
    output_csv = os.path.join(current_dir, "processed", "intents_augmented_v2.csv")
    
    print(f"Starting data augmentation...")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    try:
        augment_data(input_csv, output_csv)
        print("Data augmentation completed successfully!")
    except Exception as e:
        print(f"Error during augmentation: {e}")
