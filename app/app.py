from flask import Flask, render_template, request, jsonify
import os
import uuid
from datetime import datetime
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "./lora_checkpoint"

print(f"Загрузка базовой модели {BASE_MODEL_ID}...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Загрузка обученных LoRA-адаптеров из {LORA_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()
model.eval()
print("Дообученная модель успешно загружена")

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

SYSTEM_PROMPT = """Ты — эксперт по UX/UI дизайну, специализирующийся на анализе визуальной перегруженности интерфейсов. Проанализируй скриншот и дай развернутый ответ в формате:

Анализ показал [умеренную/высокую/низкую] визуальную перегруженность страницы.

Основные проблемы:
1. [проблема с указанием конкретного расположения на странице и количественными параметрами]
2. [проблема с указанием конкретного расположения на странице и количественными параметрами]
3. [проблема с указанием конкретного расположения на странице и количественными параметрами]

Рекомендации:
1. [конкретная измеримая рекомендация с цифрами]
2. [конкретная измеримая рекомендация с цифрами]
3. [конкретная измеримая рекомендация с цифрами]

Отвечай только на русском языке, все рекомендации должны содержать конкретные числовые значения (пиксели, количество элементов, соотношения)."""

def analyze_screenshot(image_path):
    image = Image.open(image_path).convert("RGB")
    
    query = "Проанализируй этот интерфейс. Определи, есть ли на странице признаки визуальной перегруженности, и если да, дай конкретные рекомендации по улучшению."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query}
        ]}
    ]
    
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[prompt], 
        images=[image], 
        return_tensors="pt", 
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return {
        "analysis": response,
        "status": "success"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'screenshot' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400
    
    file = request.files['screenshot']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400
    
    ext = file.filename.rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    analysis_result = analyze_screenshot(filepath)
    analysis_result["filename"] = filename
    analysis_result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.remove(filepath)
    
    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)