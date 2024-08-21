from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments, Trainer
import torch
from peft import PeftModel

def load_summarized_model(model_path = 'model/peft-summarize-text-checkpoint-local/'):
    tokenizer = AutoTokenizer.from_pretrained("pengold/t5-vietnamese-summarization")
    original_model = AutoModelForSeq2SeqLM.from_pretrained("pengold/t5-vietnamese-summarization")
    peft_model = PeftModel.from_pretrained(original_model,
                                        model_path,
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    return tokenizer, peft_model

def summarize_document(text):
    tokenizer, peft_model = load_summarized_model()
    prompt = f"""
    Summarize the following text:
    Text: 
    {text}

    Summary:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=1000, num_beans = 1))
    summarized_document = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    return summarized_document