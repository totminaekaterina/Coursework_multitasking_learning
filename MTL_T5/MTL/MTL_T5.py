import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import cuda
from tqdm import tqdm
from MTLDataSet import MTLDataSet
from NERDataSet import NERDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from rich.table import Column, Table
from rich import box
from rich.console import Console
import argparse
from rouge_score import rouge_scorer
from NLIDataSet import NLIDataset
from STSDataset import STSDataset
from SentimentDataSet import SentimentDataset
import matplotlib.pyplot as plt



class T5MTL(torch.nn.Module):
    def __init__(self, model_params: dict, task_type: str) -> None:
        super(T5MTL, self).__init__()
        self.task_type = task_type
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
        self.encoder = T5Model.from_pretrained(model_params["MODEL"]).to(self.device)
        self.decoder = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"]).to(self.device)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        # Encoding phase
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_state = encoder_outputs.last_hidden_state

        # Decoding phase
        if decoder_input_ids is not None:
            outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_last_hidden_state, labels=labels)
        else:
            outputs = self.decoder.generate(input_ids=None, encoder_hidden_states=encoder_last_hidden_state)
        return outputs


class T5Trainer:
    def __init__(self, model_params: dict,
                 source_column: str = None, target_column: str = None,
                 output_dir=None, task_type: str = None, source_column1: str = None,
                 source_column2: str = None, valid_labels: int = None) -> None:
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.source_column2 = source_column2
        self.source_column1 = source_column1
        self.task_type = task_type
        self.model_params = model_params
        self.source_column = source_column
        self.target_column = target_column
        self.valid_labels = valid_labels
        self.model = T5MTL(model_params, task_type)
        self.train_losses = []
        self.validation_losses = []
        if output_dir is not None:
            self.output_dir = os.path.join(
                Path(os.path.dirname(os.path.realpath(__file__))), output_dir)
        else:
            self.output_dir = os.path.join(
                Path(os.path.dirname(os.path.realpath(__file__))), 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        print(f"device is set to {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
        self.model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"]).to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=model_params["LEARNING_RATE"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3,
                                                                    verbose=True)

        # For logging
        self._console = Console(record=True)
        self._training_logger = Table(Column("Epoch", justify="center"),
                                      Column("Steps", justify="center"),
                                      Column("Loss", justify="center"),
                                      title="Training Status", pad_edge=False, box=box.ASCII)

        if self.task_type == 'paraphrase_task':
            self.source_column1 = None
            self.source_column2 = None
            self.source_column = 'sentence1'
            self.target_column = 'sentence2'
            self.valid_labels = None

        elif self.task_type == 'qg_task':
            self.source_column1 = None
            self.source_column2 = None
            self.source_column = 'sentence1'
            self.target_column = 'sentence2'
            self.valid_labels = None

        elif self.task_type == 'sum_task':
            self.source_column1 = None
            self.source_column2 = None
            self.source_column = 'generated_text'
            self.target_column = 'summary'
            self.valid_labels = None

        elif self.task_type == 'ner_task':
            self.source_column1 = None
            self.source_column2 = None
            self.source_column = 'text'
            self.target_column = 'tag'
            self.valid_labels = None

        elif self.task_type == 'sentiment_task':
            self.source_column1 = None
            self.source_column2 = None
            self.source_column = 'text'
            self.target_column = 'sentiment'
            self.valid_labels = {0, 1, 2}

        elif self.task_type == 'nli_task':
            self.source_column1 = 'premise'
            self.source_column2 = 'hypothesis'
            self.target_column = 'label'
            self.source_column == None
            self.valid_labels = None

        elif self.task_type == 'sts_task':
            self.source_column == None
            self.source_column1 = 'sentence1'
            self.source_column2 = 'sentence2'
            self.target_column = 'similarity_score'
            self.valid_labels = None

    def run(self, dataframe, task_type):
        # Set random seeds and deterministic pytorch for reproducibility
        torch.manual_seed(self.model_params["SEED"])  # pytorch random seed
        np.random.seed(self.model_params["SEED"])  # numpy random seed
        torch.backends.cudnn.deterministic = True

        # logging
        self._console.log(f"[Data]: Reading data...\n")

        print("Source column1:", self.source_column1)
        print("Source column2:", self.source_column2)
        print("Source column:", self.source_column)
        print("Target column:", self.target_column)

        if task_type in ['paraphrase_task', 'qg_task', 'sum_task', 'titlegen_task', 'ner_task', 'sentiment_task']:
            dataframe = dataframe[[self.source_column, self.target_column]]
        elif task_type in ['nli_task', 'sts_task']:
            dataframe = dataframe[[self.source_column1, self.source_column2, self.target_column]]

        # Creation of Dataset and Dataloader
        # Splitting dataset into train-val-test
        perm = np.random.permutation(dataframe.index)
        m = len(dataframe.index)
        train_end = int(self.model_params["TRAIN_RATIO"] * m)
        validate_end = int(self.model_params["VALID_RATIO"] * m) + train_end
        train_dataset = dataframe.iloc[perm[:train_end]]
        validation_dataset = dataframe.iloc[perm[train_end:validate_end]]
        test_dataset = dataframe.iloc[perm[validate_end:]]

        train_dataset = train_dataset.reset_index(drop=True)
        validation_dataset = validation_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        del train_end, validate_end, m, perm

        self._console.print(f"FULL Dataset: {dataframe.shape}")
        self._console.print(f"TRAIN Dataset: {train_dataset.shape}")
        self._console.print(f"VALIDATION Dataset: {validation_dataset.shape}")
        self._console.print(f"TEST Dataset: {test_dataset.shape}\n")

        if self.task_type in ['paraphrase_task', 'qg_task', 'sum_task', 'titlegen_task']:
            training_set = MTLDataSet(train_dataset, self.tokenizer,
                                      self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                      self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                      self.source_column, self.target_column)
            validation_set = MTLDataSet(validation_dataset, self.tokenizer,
                                        self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                        self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                        self.source_column, self.target_column)
            test_set = MTLDataSet(test_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                  self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                  self.source_column, self.target_column)

        elif self.task_type == 'ner_task':
            training_set = NERDataset(train_dataset, self.tokenizer,
                                      self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                      self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                      self.source_column, self.target_column)
            validation_set = NERDataset(validation_dataset, self.tokenizer,
                                        self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                        self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                        self.source_column, self.target_column)
            test_set = NERDataset(test_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                  self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                  self.source_column, self.target_column)

        elif self.task_type == 'sentiment_task':
            training_set = SentimentDataset(train_dataset, self.tokenizer,
                                            self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                            self.source_column, self.target_column)
            validation_set = SentimentDataset(validation_dataset, self.tokenizer,
                                              self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                              self.source_column, self.target_column)
            test_set = SentimentDataset(test_dataset, self.tokenizer,
                                        self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                        self.source_column, self.target_column)

        elif self.task_type == 'nli_task':
            # For the NLI task, use the NLIDataset class
            training_set = NLIDataset(train_dataset, self.tokenizer,
                                      self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                      self.source_column1, self.source_column2, self.target_column)
            validation_set = NLIDataset(validation_dataset, self.tokenizer,
                                        self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                        self.source_column1, self.source_column2, self.target_column)
            test_set = NLIDataset(test_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                  self.source_column1, self.source_column2, self.target_column)

        elif self.task_type == 'sts_task':
            # For the NLI task, use the NLIDataset class
            training_set = STSDataset(train_dataset, self.tokenizer,
                                      self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                      self.source_column1, self.source_column2, self.target_column)
            validation_set = STSDataset(validation_dataset, self.tokenizer,
                                        self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                        self.source_column1, self.source_column2, self.target_column)
            test_set = STSDataset(test_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"],
                                  self.source_column1, self.source_column2, self.target_column)

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.model_params["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
        }

        validation_params = {
            'batch_size': self.model_params["VALID_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
        }

        test_params = {
            'batch_size': self.model_params["TEST_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
        }
        # Creation of Dataloaders for training and validation.
        if self.task_type == 'ner_task':
            training_loader = DataLoader(training_set, collate_fn=training_set.custom_collate_fn, **train_params)
            validation_loader = DataLoader(validation_set, collate_fn=validation_set.custom_collate_fn,
                                           **validation_params)
            test_loader = DataLoader(test_set, collate_fn=test_set.custom_collate_fn, **test_params)

        elif self.task_type in ['paraphrase_task', 'qg_task', 'titlegen_task', 'sum_task']:
            training_loader = DataLoader(training_set, **train_params)
            validation_loader = DataLoader(validation_set, **validation_params)
            test_loader = DataLoader(test_set, **test_params)

        elif self.task_type == 'sentiment_task':
            training_loader = DataLoader(training_set, **train_params)
            validation_loader = DataLoader(validation_set, **validation_params)
            test_loader = DataLoader(test_set, **test_params)

        elif self.task_type == 'nli_task':
            training_loader = DataLoader(training_set, collate_fn=training_set.custom_collate_fn, **train_params)
            validation_loader = DataLoader(validation_set, collate_fn=validation_set.custom_collate_fn,
                                           **validation_params)
            test_loader = DataLoader(test_set, collate_fn=test_set.custom_collate_fn, **test_params)

        elif self.task_type == 'sts_task':
            training_loader = DataLoader(training_set, collate_fn=training_set.custom_collate_fn, **train_params)
            validation_loader = DataLoader(validation_set, collate_fn=validation_set.custom_collate_fn,
                                           **validation_params)
            test_loader = DataLoader(test_set, collate_fn=test_set.custom_collate_fn, **test_params)

        # Train and validate
        self._console.log(f'[Initiating Fine Tuning]...\n')
        self.train(training_loader, validation_loader)

        # evaluating the test dataset and generate prediction report
        self._console.log(f"[Initiating Test]...\n")
        predictions, actuals = self.test(test_loader, task_type)


        # Сохранение предсказаний в CSV в зависимости от типа задачи
        if self.task_type == 'ner_task':
            final_df = pd.DataFrame({
                'actual': actuals,
                'tags': predictions
            })
            final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)

        elif self.task_type in ['paraphrase_task', 'qg_task', 'titlegen_task', 'sum_task']:
            final_df = pd.DataFrame({
                'actual': actuals,
                'generated': predictions
            })
            final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)

        elif self.task_type == 'sentiment_task':
            final_df = pd.DataFrame({
                'actual': actuals,
                'sentiment': predictions
            })
            final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)
        elif self.task_type in ['sts_task', 'nli_task']:
            final_df = pd.DataFrame({
            'actual': actuals,
            'prediction': predictions
            })
            final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)

        self._console.save_text(os.path.join(self.output_dir, 'logs.txt'))

        self._console.log(f"[TEST Completed.]\n")
        self._console.print(
            f"""[Model] Model saved @ {os.path.join(self.output_dir, "model_files")}\n""")
        self._console.print(
            f"""[TEST] Generation on Test data saved @ {os.path.join(self.output_dir, 'predictions.csv')}\n""")
        self._console.print(
            f"""[Logs] Logs saved @ {os.path.join(self.output_dir, 'logs.txt')}\n""")

    def plot_loss_curves(self, task_type):
        """
        Plot the training and validation loss curves.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        # Convert CUDA tensors to CPU before converting to NumPy
        validation_losses_cpu = [val_loss.cpu().numpy() for val_loss in self.validation_losses]
        plt.plot(range(1, len(self.validation_losses) + 1), validation_losses_cpu, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_plot_filename = f'{task_type}_loss_plot.png'
        plt.savefig(loss_plot_filename)  # Save the plot as a PNG file
        plt.show()

    def train(self, train_loader, validation_loader):
        """
        After each epoch of training, validate with the validation set and save the best model.
        """
        best_val_loss = np.inf
        patience_for_early_stopping = 0
        model_save_path = os.path.join(self.output_dir, "model_files")
        os.makedirs(model_save_path, exist_ok=True)

        # Set model to training mode
        self.model.train()

        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            epoch_train_loss = 0  # Initialize the loss for the epoch

            for i, batch in enumerate(tqdm(train_loader)):
                ids = batch['source_ids'].to(self.device, dtype=torch.long)
                mask = batch['source_mask'].to(self.device, dtype=torch.long)

                y = batch['target_ids'].to(self.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()

                # Labels for computing the sequence classification/regression loss
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100

                target_mask = batch['target_ids_y'].to(self.device, dtype=torch.long)
                target_mask = target_mask[:, :-1].contiguous()

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=ids, attention_mask=mask,
                                     decoder_input_ids=y_ids, labels=lm_labels,
                                     decoder_attention_mask=target_mask)
                loss = outputs[0]

                # Accumulate the loss for the epoch
                epoch_train_loss += loss.item()

                if i % 500 == 0:
                    self._training_logger.add_row(
                        str(epoch), str(i), str(loss))
                    self._console.print(self._training_logger)

                loss.backward()
                self.optimizer.step()

            # Validation and save the best model
            self._console.log(f"[Initiating Validation on epoch {epoch}]...\n")
            validation_loss = self.validate(validation_loader)
            self._console.log(f"Validation loss: {validation_loss} | Prev best Validation loss: {best_val_loss}\n")

            # Accumulate the losses
            self.train_losses.append(epoch_train_loss)
            self.validation_losses.append(validation_loss)

            if validation_loss < best_val_loss:
                patience_for_early_stopping = 0
                # Save the best model
                self._console.log(f"Better validation loss, so saving the model...\n")
                self.model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)
                best_val_loss = validation_loss
            else:
                patience_for_early_stopping += 1

            # Early stopping if validation loss is not decreasing within the given patience number
            if patience_for_early_stopping == self.model_params['EARLY_STOPPING_PATIENCE']:
                break

            # LR scheduling based on validation loss
            self.scheduler.step(validation_loss)

        # Plot the loss curves after all epochs
        self.plot_loss_curves(task_type)

    def validate(self, loader):
        """
        sum the validation loss over all batches and return the average validation loss
        """
        self.model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader):
                ids = batch['source_ids'].to(self.device, dtype=torch.long)
                mask = batch['source_mask'].to(self.device, dtype=torch.long)

                y = batch['target_ids'].to(self.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()

                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100

                target_mask = batch['target_ids_y'].to(self.device, dtype=torch.long)
                target_mask = target_mask[:, :-1].contiguous()

                outputs = self.model(input_ids=ids, attention_mask=mask,
                                     decoder_input_ids=y_ids, labels=lm_labels,
                                     decoder_attention_mask=target_mask)
                loss = outputs[0]

                total_valid_loss += loss

        avg_valid_loss = total_valid_loss / len(loader)
        return avg_valid_loss

    def test(self, loader, task_type):
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                generated_ids = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    min_length=1,
                    max_length=self.model_params['MAX_TARGET_TEXT_LENGTH'],
                    num_beams=5,
                    repetition_penalty=2.5,
                    length_penalty=0.8,
                    early_stopping=True
                )

                preds = [self.tokenizer.decode(g, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True) for t in y]

                if i % 50 == 0:
                    self._console.print(f'Completed {i}')

                predictions.extend(preds)
                actuals.extend(target)

        if task_type in ['paraphrase_task', 'qg_task', 'titlegen_task', 'sum_task']:
            rouge_scores = self.compute_rouge_scores(actuals, predictions)
            self._console.log("ROUGE scores:")
            log_text = "ROUGE scores:\n"
            for metric, score in rouge_scores.items():
                log_text += f'{metric}: {score}\n'
                self._console.log(f'{metric}: {score}')
            with open(os.path.join(self.output_dir, 'logs.txt'), 'a') as f:
                f.write(log_text)

        if task_type in ['ner_task', 'sentiment_task', 'sts_task', 'nli_task']:
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, average='weighted', zero_division='warn')
            recall = recall_score(actuals, predictions, average='weighted', zero_division='warn')
            f1 = f1_score(actuals, predictions, average='weighted', zero_division='warn')

            self._console.log("Metrics:")
            self._console.log(f"Accuracy: {accuracy}")
            self._console.log(f"Precision: {precision}")
            self._console.log(f"Recall: {recall}")
            self._console.log(f"F1 Score: {f1}")

            with open(os.path.join(self.output_dir, 'logs.txt'), 'a') as f:
                f.write(f"Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

        return actuals, predictions

    def compute_rouge_scores(self, references, predictions):
        rouge_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores = self.compute_rouge_metric(metric, references, predictions)
            rouge_scores[metric] = scores
        return rouge_scores

    def compute_rouge_metric(self, metric, references, predictions):
        scorer = self.rouge_scorer
        precision_scores = []
        recall_scores = []
        fmeasure_scores = []
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            precision_scores.append(scores[metric].precision)
            recall_scores.append(scores[metric].recall)
            fmeasure_scores.append(scores[metric].fmeasure)
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_fmeasure = sum(fmeasure_scores) / len(fmeasure_scores)
        return f'Score(precision={avg_precision}, recall={avg_recall}, fmeasure={avg_fmeasure})'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pass the TASK name')
    parser.add_argument('-t', '--task', help='e.g. paraphrase_task', required=True)
    parser.add_argument('-d', '--dataset', help='e.g. paraphrase', required=True)
    args = vars(parser.parse_args())

    TASK = args['task']
    DATASET = args['dataset']
    assert TASK in {'paraphrase_task', 'qg_task', 'sum_task', 'titlegen_task', 'ner_task', 'sentiment_task', 'nli_task',
                    'sts_task'}
    assert DATASET in {'paraphrase', 'ner', 'sentiment', 'nli', 'titlegen', 'sum', 'qg', 'sts'}
    del parser, args
    print(f"{DATASET} will be trained for {TASK} task")

    dataset_dir_path = Path(DATASET)
    dataset_path = os.path.join(dataset_dir_path, f'{DATASET}.csv')

    df = pd.read_csv(dataset_path, index_col=False, sep=";", encoding="utf-8-sig").astype(str)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head(10))
    print(f"Length: {len(df)}")

    model_params = {
        "MODEL": "./ner_task_ner/model_files",
        "TRAIN_BATCH_SIZE": 8,
        "VALID_BATCH_SIZE": 8,
        "TEST_BATCH_SIZE": 8,
        "TRAIN_RATIO": 0.8,
        "VALID_RATIO": 0.1,
        "TRAIN_EPOCHS": 5,
        "EARLY_STOPPING_PATIENCE": 4,
        "LEARNING_RATE": 3e-5,
        "MAX_SOURCE_TEXT_LENGTH": 512,
        "MAX_TARGET_TEXT_LENGTH": 512,
        "SEED": 42
    }

    t5_trainer = T5Trainer(model_params=model_params, output_dir=f'{TASK}_{DATASET}', task_type=TASK)
    task_type = TASK  #
    t5_trainer.run(dataframe=df, task_type=task_type)
