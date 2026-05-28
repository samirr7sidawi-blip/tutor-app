import google.generativeai as genai

# PASTE YOUR KEY HERE
api_key = "AIza..." 

genai.configure(api_key=api_key)

print("--- 2026 Model Check ---")
try:
    # This lists the new Gemini 2.5 and 3.0 models available to you
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Active: {m.name}")
except Exception as e:
    print(f"❌ Connection Error: {e}")