import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "gpt2"
HISTORY_FILE = "data/chat_history.json"

os.makedirs("data", exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

print("Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Modelo carregado!")

def gerar_resposta(pergunta):
    inputs = tokenizer.encode(pergunta + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

def salvar_historico(user_input, resposta):
    with open(HISTORY_FILE, "r") as f:
        historico = json.load(f)
    historico.append({"user": user_input, "bot": resposta})
    with open(HISTORY_FILE, "w") as f:
        json.dump(historico, f, indent=4)

print("Bem-vindo à sua IA! Digite 'sair' para encerrar.")
while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        print("Encerrando chat. Até mais!")
        break
    resposta = gerar_resposta(pergunta)
    print(f"IA: {resposta}")
    salvar_historico(pergunta, resposta)
