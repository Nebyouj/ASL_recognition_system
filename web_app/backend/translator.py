from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self):
        self.models = {}

    def load_model(self, target_lang):
        model_name_map = {
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "ar": "Helsinki-NLP/opus-mt-en-ar",
            "de": "Helsinki-NLP/opus-mt-en-de"
        }

        if target_lang not in model_name_map:
            return None
        
        if target_lang not in self.models:
            model_name = model_name_map[target_lang]

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            self.models[target_lang] = (tokenizer, model)

        return  self.models[target_lang]
    
    def translate(self, text, target_lang):
        loaded = self.load_model(target_lang)
        if loaded is None:
            return text  # ✅ return original instead of None so frontend doesn't break

        tokenizer, model = loaded
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs, max_length=512)  # ✅ set a cap
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def translate_with_model(self, text, model_name):
        if not model_name:
            return text
        if model_name not in self.models:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self.models[model_name] = (tokenizer, model)
        tokenizer, model = self.models[model_name]
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs, max_length=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)