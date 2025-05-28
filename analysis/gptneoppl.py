import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

def compute_perplexity_sliding_window(model, tokenizer, texts, stride=512):
    device = next(model.parameters()).device
    model.eval()

    nll_sum = 0.0
    n_tokens = 0

    for text in tqdm(texts):
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_len = encodings.input_ids.size(1)
        max_length = model.config.max_position_embeddings
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be smaller on last step

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            batch_size = target_ids.size(0)
            num_valid_tokens = (target_ids != -100).sum().item()
            num_loss_tokens = num_valid_tokens - batch_size

            nll_sum += neg_log_likelihood.item() * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    return ppl

def main():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = dataset[:100]["text"]

    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # switch to "cuda" if you have GPU

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)

    print("\n--- Final Output ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    ppl = compute_perplexity_sliding_window(model, tokenizer, texts)
    print(f"Perplexity on WikiText-2 validation: {ppl:.2f}")

if __name__ == "__main__":
    main()
