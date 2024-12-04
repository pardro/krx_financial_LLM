import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM
from huggingface_hub import login

from data_utils import load_tokenizer

class DataCollatorForSupervisedDataset(object):
    def __init__(self,
                 tokenizer):
        self.tokenizer = tokenizer

    def __call__(self,
                 instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def model_load(
    model_id,
    hf_token=None
):
    if hf_token is not None:
        login(token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        device_map="auto",
    )
    model.config.use_cache = False
    tokenizer = load_tokenizer(model_id, hf_token)

    return model, tokenizer

def train_args_load(
    path,
    batch_size,
    epochs,
    max_seq_length
):
    training_args = SFTConfig(
        output_dir=path+'save',
        report_to='wandb',
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size['per_device_train_batch_size'],
        per_device_eval_batch_size=batch_size['per_device_eval_batch_size'],
        gradient_accumulation_steps=batch_size['gradient_accumulation_steps'],
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
        max_seq_length=max_seq_length,
        weight_decay=0.01,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        logging_steps=10,
        save_total_limit=4,
        metric_for_best_model='eval_loss',
        gradient_checkpointing=True,
        greater_is_better=False,
        packing=False
    )
    return training_args

def trainer_load(
    model_id,
    train_dataset,
    valid_dataset,
    training_args,
    hf_token=None
):
    model, tokenizer = model_load(model_id, hf_token)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    return trainer
