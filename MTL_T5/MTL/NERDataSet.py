import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    """
    Custom dataset for Named Entity Recognition (NER)
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_col, target_col):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_col]
        self.target_text = self.data[target_col]
        self.label_map = {'O': 0, 'B-LOC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-ORG': 5, 'I-PER': 6, 'B-MISC': 7, 'I-MISC': 8, 'PAD': 9}

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Cleaning data to ensure it is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        # Encode source and target texts
        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length",
            return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, pad_to_max_length=True, truncation=True, padding="max_length",
            return_tensors='pt')

        # Extract input and output tensors
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        # Encode NER labels
        ner_labels = [self.label_map[label] for label in target_text.split()]

        # Pad or truncate labels to fixed length
        if len(ner_labels) < self.target_len:
            ner_labels += [self.label_map['PAD']] * (self.target_len - len(ner_labels))
        else:
            ner_labels = ner_labels[:self.target_len]

        ner_labels = torch.tensor(ner_labels)

        # Convert labels to indices
        label_indices = [self.convert_label_to_index(label) for label in target_text.split()]
        label_indices = torch.tensor(label_indices)

        # Create one-hot representation
        ner_labels_one_hot = torch.nn.functional.one_hot(label_indices, num_classes=10)

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_mask.to(dtype=torch.long),
            'ner_labels': ner_labels_one_hot.to(dtype=torch.long)  # Return one-hot encoded NER labels
        }

    def custom_collate_fn(self, batch):
        # Получить списки исходных и целевых тензоров
        source_ids = [item['source_ids'] for item in batch]
        source_mask = [item['source_mask'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]
        target_mask = [item['target_ids_y'] for item in batch]
        ner_labels = [item['ner_labels'] for item in batch]

        # Паддинг исходных и целевых тензоров до максимальной длины в батче
        source_ids_padded = torch.nn.utils.rnn.pad_sequence(source_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        source_mask_padded = torch.nn.utils.rnn.pad_sequence(source_mask, batch_first=True, padding_value=0)
        target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        target_mask_padded = torch.nn.utils.rnn.pad_sequence(target_mask, batch_first=True, padding_value=0)
        ner_labels_padded = torch.nn.utils.rnn.pad_sequence(ner_labels, batch_first=True, padding_value=9)  # Assuming PAD label is 9

        return {
            'source_ids': source_ids_padded,
            'source_mask': source_mask_padded,
            'target_ids': target_ids_padded,
            'target_ids_y': target_mask_padded,
            'ner_labels': ner_labels_padded
        }
    def convert_label_to_index(self, label):
        if label == 'O':
            return 0
        elif label.startswith('B-'):
            return self.label_map[label]  # Use the label map for conversion
        elif label.startswith('I-'):
            return self.label_map[label]  # Use the label map for conversion
        else:
            return 0  # Assume 'O' for other cases

