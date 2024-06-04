import torch
from torch.utils.data import Dataset

class NLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, source_column1, source_column2, target_column):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.source_column1 = source_column1
        self.source_column2 = source_column2
        self.target_column = target_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        premise_text = str(self.data.loc[index, self.source_column1])
        hypothesis_text = str(self.data.loc[index, self.source_column2])
        target_label = int(self.data.loc[index, self.target_column])

        premise_text = ' '.join(premise_text.split())
        hypothesis_text = ' '.join(hypothesis_text.split())

        encoding = self.tokenizer(
            text=premise_text,
            text_pair=hypothesis_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'source_ids': input_ids.to(dtype=torch.long),
            'source_mask': attention_mask.to(dtype=torch.long),
            'target_label': torch.tensor(target_label, dtype=torch.long),
            'target_ids': encoding['input_ids'].squeeze(),  # Include target_ids here
            'target_ids_y': encoding['input_ids'].squeeze()  # Include target_ids_y here
        }

    def custom_collate_fn(self, batch):
        input_ids = [item['source_ids'] for item in batch]
        attention_mask = [item['source_mask'] for item in batch]
        target_labels = [item['target_label'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]  # Add target_ids here
        target_ids_y = [item['target_ids_y'] for item in batch]  # Add target_ids_y here

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            'source_ids': input_ids_padded,
            'source_mask': attention_mask_padded,
            'target_labels': torch.stack(target_labels),
            'target_ids': torch.stack(target_ids),  # Include target_ids here
            'target_ids_y': torch.stack(target_ids_y)  # Include target_ids_y here
        }
