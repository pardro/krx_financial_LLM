import json, torch, re, os, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self,
                 data_lst,
                 tokenizer):
        IGNORE_INDEX = -100
        self.data_lst = data_lst
        self.input_lst = []
        self.label_lst = []
        self.input_token_len = []

        for data in tqdm(data_lst):
            source = tokenizer(
                data['prompt'],
                add_special_tokens=False,
                return_tensors="pt",
                )
            target = tokenizer(
                data['target'] + tokenizer.eos_token,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
                )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat(
                (
                    source['input_ids'][0],
                    target["input_ids"][0]
                )
            )
            labels = torch.concat(
                (
                    torch.LongTensor([IGNORE_INDEX] * source['input_ids'][0].shape[0]),
                    target["input_ids"][0]
                )
            )

            self.input_lst.append(input_ids)
            self.label_lst.append(labels)
            self.input_token_len.append(len(input_ids))

    def __len__(self):
        return len(self.input_lst)

    def __getitem__(self,
                    idx):
        return self.input_lst[idx], self.label_lst[idx]

    def get_max_token_length(self):
        return max(self.input_token_len)

def load_from_jsonl(
    filename
):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(
    filename,
    data
):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

def load_tokenizer(
    model_id,
    hf_token=None
):
    if hf_token is not None:
        login(token=hf_token)
    else:
        print("You're not logged in to HuggingFace.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer

def data_load(
    path,
    file_path,
    model_id,
    hf_token=None
):
    data = [y for x in file_path for y in load_from_jsonl(path + x)]
    tr_lst, va_lst = train_test_split(data, test_size=0.05, random_state=42)
    random.shuffle(tr_lst)
    # save_to_jsonl(path + 'va_lst.jsonl', va_lst)

    tokenizer = load_tokenizer(model_id, hf_token)

    train_dataset = CustomDataset(tr_lst, tokenizer)
    valid_dataset = CustomDataset(va_lst, tokenizer)

    train_max_token_len = train_dataset.get_max_token_length()
    valid_max_token_len = valid_dataset.get_max_token_length()

    from datasets import Dataset

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.input_lst,
        "labels": train_dataset.label_lst,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.input_lst,
        "labels": valid_dataset.label_lst,
        })
    print('train_data : {}'.format(format(len(tr_lst), ',')))
    print('valid_data : {}'.format(format(len(va_lst), ',')))
    print('-='*20)
    print('train_max_token_len : {}'.format(format(train_max_token_len, ',')))
    print('valid_max_token_len : {}'.format(format(valid_max_token_len, ',')))

    return train_dataset, valid_dataset