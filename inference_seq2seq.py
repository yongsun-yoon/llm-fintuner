import hydra
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from prompts import construct_seq2seq_prompt

@hydra.main(version_base='1.3', config_path='conf', config_name='seq2seq.yaml')
def main(cfg):
    print(cfg)
    peft_config = PeftConfig.from_pretrained(cfg.ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, cfg.ckpt_dir)
    model.eval().requires_grad_(False).to(cfg.inference_device)
    # model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    
    while True:
        instruction_text = input('instruction을 입력하세요. : ')
        input_text = input('input을 입력하세요. : ')
        if not instruction_text and not input_text:
            break
            
        prompt = construct_seq2seq_prompt(instruction_text, input_text)
        inputs = tokenizer(prompt, max_length=cfg.data.input_max_length, truncation=True, return_tensors='pt')
        inputs = inputs.to(cfg.inference_device)
        
        outputs = model.generate(**inputs, max_new_tokens=cfg.data.output_max_length)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)
    
if __name__ == '__main__':
    main()