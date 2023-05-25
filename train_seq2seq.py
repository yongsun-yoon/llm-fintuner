import json
import hydra
import wandb
from tqdm import tqdm
from glob import glob

import torch
import lightning as L
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    get_scheduler,
    BatchEncoding
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

from prompts import construct_seq2seq_prompt


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_max_length, output_max_length):
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.data = load_dataset('csv', data_dir='data')['train']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction_text = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        prompt = construct_seq2seq_prompt(instruction_text, input_text)
        
        input_ids = self.tokenizer(prompt, max_length=self.input_max_length, truncation=True).input_ids        
        labels = self.tokenizer(output_text, max_length=self.output_max_length, truncation=True).input_ids
        return torch.tensor(input_ids), torch.tensor(labels)
    
    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    
    def get_dataloader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        


@hydra.main(version_base='1.3', config_path='conf', config_name='seq2seq.yaml')
def main(cfg):
    print(cfg)
    
    # data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset = dataset = Seq2SeqDataset(tokenizer, cfg.data.input_max_length, cfg.data.output_max_length)
    dataloader = dataset.get_dataloader(cfg.data.batch_size, shuffle=True)
    
    # model
    fabric = L.Fabric(**cfg.fabric)
    fabric.launch()
    fabric.seed_everything(cfg.seed)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)
    
    peft_config = LoraConfig(
        task_type = TaskType.SEQ_2_SEQ_LM, 
        inference_mode = False, 
        r = cfg.lora.r, 
        lora_alpha = cfg.lora.alpha, 
        lora_dropout = cfg.lora.dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    num_training_steps = len(dataloader) * cfg.train.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.scheduler.warmup_ratio)
    scheduler = get_scheduler(cfg.scheduler.type, optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    
    # setup
    wrapped_model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    if fabric.is_global_zero:
        wandb.init(project=cfg.wandb_project_name)
    
    for ep in range(cfg.train.num_epochs):
        pbar = tqdm(dataloader, disable=not fabric.is_global_zero)
        optimizer.zero_grad()
        
        for st, batch in enumerate(pbar):
            fabric.barrier()
            
#             is_accumulating = (st % cfg.train.gradient_accumulation_steps) != 0
#             with fabric.no_backward_sync(wrapped_model, enabled=is_accumulating):
#                 outputs = wrapped_model(**batch)
#                 loss = outputs.loss
#                 fabric.backward(loss)
                
#             if not is_accumulating:
#                 optimizer.step()
#                 optimizer.zero_grad()

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()
            
            if fabric.is_global_zero:
                log = {'loss': loss.item()}
                pbar.set_postfix(log)
                wandb.log(log)
        
        if fabric.is_global_zero:
            tokenizer.save_pretrained(cfg.ckpt_dir)
            model.save_pretrained(cfg.ckpt_dir)
            
            
            

    
if __name__ == '__main__':
    main()