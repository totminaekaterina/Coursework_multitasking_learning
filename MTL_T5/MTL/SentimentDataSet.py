import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    Custom dataset for Sentiment Analysis
    """

    def __init__(self, dataframe, tokenizer, max_length, source_col, target_col):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.source_col = source_col
        self.target_col = target_col
        self.valid_labels = {0, 1, 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = str(self.data.loc[index, self.source_col])
        target_label = int(self.data.loc[index, self.target_col])

        # Ensure target_label is one of the valid labels
        if target_label not in self.valid_labels:
            raise ValueError(f"Invalid target label {target_label} at index {index}. Expected one of {self.valid_labels}.")

        # Cleaning data to ensure it is in string type
        source_text = ' '.join(source_text.split())

        # Encode source text
        encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Placeholder for target_ids and target_ids_y
        target_ids = torch.zeros_like(input_ids)
        target_ids_y = torch.zeros_like(input_ids)

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'attention_mask': attention_mask.to(dtype=torch.long),
            'source_ids': input_ids.to(dtype=torch.long),
            'source_mask': attention_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),  # Placeholder for target_ids
            'target_ids_y': target_ids_y.to(dtype=torch.long),  # Placeholder for target_ids_y
            'target_label': torch.tensor(target_label, dtype=torch.long)
        }

    def custom_collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        source_ids = [item['source_ids'] for item in batch]
        source_mask = [item['source_mask'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]  # Include target_ids
        target_ids_y = [item['target_ids_y'] for item in batch]  # Include target_ids_y
        target_label = [item['target_label'] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        source_ids_padded = torch.nn.utils.rnn.pad_sequence(source_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        source_mask_padded = torch.nn.utils.rnn.pad_sequence(source_mask, batch_first=True, padding_value=0)
        target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)  # Pad target_ids
        target_ids_y_padded = torch.nn.utils.rnn.pad_sequence(target_ids_y, batch_first=True, padding_value=self.tokenizer.pad_token_id)  # Pad target_ids_y

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'source_ids': source_ids_padded,
            'source_mask': source_mask_padded,
            'target_ids': target_ids_padded,  # Include target_ids
            'target_ids_y': target_ids_y_padded,  # Include target_ids_y
            'target_label': torch.stack(target_label)
        }
