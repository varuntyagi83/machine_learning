!pip install langcodes
!pip install language_data

from transformers import MarianMTModel, MarianTokenizer
from langcodes import Language

model_name = "Helsinki-NLP/opus-mt-en-roa"
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)

supported_language_codes = [code[2:-2] for code in tokenizer.supported_language_codes]

def detect_language_name(language_code):
    try:
        language_name = Language.get(language_code).language_name()
        return language_name
    except ValueError:
        return f"Language name not found for code: {language_code}"

# Mapping language codes to language names
language_mapping = {code: detect_language_name(code) for code in supported_language_codes}

# Display the results
for code, name in language_mapping.items():
    print(f"Language Code: {code}, Language Name: {name}")

src_text = [
    ">>fra<< I want to travel to france for a holiday",
    ">>por<< Have you ever visited portugal before?",
    ">>spa<< Spain is an amazing country to live in",
    ">>ita<< I love Italian cuisine. It is one of my favorite",
    ">>ind<< Indonesia lies in the south asia region"
]

model_name = "Helsinki-NLP/opus-mt-en-roa"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Iterate over both source texts and translations
for source, translation in zip(src_text, tgt_text):
    language_code = source.split(">>")[1][:3]
    
    # Detect language using the extracted language code
    detected_language = language_code
    
    # Convert language code to language name using langcodes
    language_name = Language.get(detected_language).language_name('en')
    
    print(f"The translation in {language_name} is: {translation}")
