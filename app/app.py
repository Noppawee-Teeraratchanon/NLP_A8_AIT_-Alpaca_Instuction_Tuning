from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


app = Flask(__name__)


def generate_output(text,saved_model,tokenizer):
    
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate text with the model
    output_ids = saved_model.generate(input_ids, max_length=150)

    # Decode the generated output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


@app.route('/', methods=['GET', 'POST'])
def index():
    input = None
    output_text = None

    model_name_or_path = "./saved_model" 
    device = "cpu"

    # Load saved model
    saved_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if request.method == 'POST':
        # Clear the cache
        input = None
        output_text = None

        input = request.form['input']
        output_text = generate_output(input,saved_model,tokenizer)
        
    return render_template('index.html', input=input, output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
