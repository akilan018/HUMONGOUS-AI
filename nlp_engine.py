# nlp_engine.py

import json
import random
import spacy

class NlpEngine:
    def __init__(self, intents_file):
        print("Loading NLP engine...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            with open(intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)['intents']
            print("✅ NLP engine and intents loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading NLP engine: {e}")
            self.nlp = None; self.intents = []

    def _preprocess_text(self, text):
        if not self.nlp: return set()
        doc = self.nlp(text.lower())
        return {token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()}

    def get_intent(self, user_message):
        if not self.intents: return None
        user_keywords = self._preprocess_text(user_message)
        if not user_keywords: return next((i for i in self.intents if i['tag'] == 'fallback'), None)

        best_match_score = 0.0
        best_match_intent = None

        print("\n--- INTENT SCORING ANALYSIS ---")
        print(f"User Keywords: {user_keywords}")

        for intent in self.intents:
            max_pattern_score = 0.0
            for pattern in intent['patterns']:
                pattern_keywords = self._preprocess_text(pattern)
                if not pattern_keywords: continue
                
                intersection = len(user_keywords.intersection(pattern_keywords))
                union = len(user_keywords.union(pattern_keywords))
                score = intersection / union if union > 0 else 0.0
                
                if score > max_pattern_score: max_pattern_score = score
            
            print(f"  - Intent: '{intent['tag']}', Score: {max_pattern_score:.2f}")

            if max_pattern_score > best_match_score:
                best_match_score = max_pattern_score
                best_match_intent = intent
        
        CONFIDENCE_THRESHOLD = 0.25
        print(f"==> Best Match Found: '{best_match_intent['tag'] if best_match_intent else 'None'}' with score {best_match_score:.2f}")
        
        if best_match_score >= CONFIDENCE_THRESHOLD:
            print("--- END SCORING (Confident Match) ---")
            return best_match_intent
        else:
            print(f"Confidence score is below threshold of {CONFIDENCE_THRESHOLD}. Using fallback.")
            print("--- END SCORING (Fallback) ---")
            return next((i for i in self.intents if i['tag'] == 'fallback'), None)

    def get_response(self, intent):
        if not intent or not intent['responses']: return "I'm sorry, I'm having trouble understanding right now."
        return random.choice(intent['responses'])
