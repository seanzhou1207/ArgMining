"""
This script trains sentence transformers with a triplet loss function.
As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.datasets import NoDuplicatesDataLoader

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers import util
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List


import csv
import logging
import os
import sys

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-uncased'


def get_training_examples(df, eval_df, loss):
    
    logger.info("Read Triplet train dataset")
    train_examples = []
    for idx, row in df.iterrows():
        if loss == 'ContrastiveLoss':
            train_examples.append(InputExample(texts=[row['Premise'], row['Conclusion']], label=row['label']))
        elif loss == 'MultipleNegativesRankingLoss':
            if row['label'] == 1:
                train_examples.append(InputExample(texts=[row['Premise'], row['Conclusion']], label=1))
        else:
            train_examples.append(InputExample(texts=[row['anchor'], row['pos'], row['neg']], label=0))
            
    
    dev_samples = []
    for idx, row in eval_df.iterrows():
        dev_samples.append(InputExample(texts=[row['Premise'], row['Conclusion']], label=row['label']))
    
    return train_examples, dev_samples

def train_mt_model(nov_df, nov_eval_df, val_df, val_eval_df, output_path, model_name, num_epochs=3, train_batch_size=16, model_suffix='', \
                data_file_suffix='', max_seq_length=256, 
                special_tokens=[], novelty_loss='Triplet', validity_loss='Triplet', sentence_transformer=False, evaluation_steps=5):
    
    output_path = output_path + model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if sentence_transformer:
        word_embedding_model = SentenceTransformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
    else:
        word_embedding_model = models.Transformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
    
    
    if len(special_tokens) > 0:
        word_embedding_model.tokenizer.add_tokens(special_tokens, special_tokens=True)
        word_embedding_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    novelty_train_examples, novelty_dev_samples   = get_training_examples(nov_df, nov_eval_df, novelty_loss)
    validity_train_examples, validity_dev_samples = get_training_examples(val_df, val_eval_df, validity_loss)
    

    print('Len of training: {}'.format(len(novelty_train_examples + validity_train_examples)))
    print('Len of Dev: {}'.format(len(novelty_dev_samples + validity_dev_samples)))
    
    if novelty_loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        novelty_train_dataloader = NoDuplicatesDataLoader(novelty_train_examples, batch_size=train_batch_size)
        # Our training loss
        novelty_loss_func = losses.MultipleNegativesRankingLoss(model)
    elif novelty_loss == 'ContrastiveLoss':
        novelty_train_dataloader = DataLoader(novelty_train_examples, shuffle=True, batch_size=train_batch_size)
        novelty_loss_func = losses.ContrastiveLoss(model)
    else:
        novelty_train_dataloader = DataLoader(novelty_train_examples, shuffle=True, batch_size=train_batch_size)
        novelty_loss_func = losses.TripletLoss(model)
        
    if validity_loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        validity_train_dataloader = NoDuplicatesDataLoader(validity_train_examples, batch_size=train_batch_size)
        # Our training loss
        validity_loss_func = losses.MultipleNegativesRankingLoss(model)
    elif validity_loss == 'ContrastiveLoss':
        validity_train_dataloader = DataLoader(validity_train_examples, shuffle=True, batch_size=train_batch_size)
        validity_loss_func = losses.ContrastiveLoss(model)
    else:
        validity_train_dataloader = DataLoader(validity_train_examples, shuffle=True, batch_size=train_batch_size)
        validity_loss_func = losses.TripletLoss(model)
        
    

    evaluator = BinaryClassificationEvaluator.from_input_examples(validity_dev_samples + novelty_dev_samples, batch_size=train_batch_size, name='sts-dev')

    warmup_steps = int((len(novelty_train_dataloader) + len(validity_train_dataloader)) * num_epochs * 0.1) #10% of train data

    print('Evaluating before start learning.....')
    model.evaluate(evaluator)
    print('====== Start Training =======')
    
    # Train the model
    model.fit(train_objectives=[(validity_train_dataloader, validity_loss_func), (novelty_train_dataloader, novelty_loss_func)],
              evaluator=evaluator,
              epochs=num_epochs,
              save_best_model=True,
              checkpoint_save_steps=evaluation_steps,
              optimizer_params={'lr':5e-06},
              checkpoint_save_total_limit=3,
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)
    
    return model, evaluator

            
def train_model(df, eval_df, output_path, model_name, num_epochs=3, train_batch_size=16, model_suffix='', \
                data_file_suffix='', max_seq_length=256, 
                special_tokens=[], loss='Triplet', sentence_transformer=False, evaluation_steps=5, lr=5e-06):
    
    output_path = output_path + model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if sentence_transformer:
        word_embedding_model = SentenceTransformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
    else:
        word_embedding_model = models.Transformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
    
    
    if len(special_tokens) > 0:
        word_embedding_model.tokenizer.add_tokens(special_tokens, special_tokens=True)
        word_embedding_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    train_examples, dev_samples = get_training_examples(df, eval_df, loss)

    print('Len of training: {}'.format(len(train_examples)))
    print('Len of Dev: {}'.format(len(dev_samples)))
    
    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=train_batch_size)
        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss == 'ContrastiveLoss':
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model)
    else:
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model)
    

    evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

    print('Evaluating before start learning.....')
    model.evaluate(evaluator)
    print('====== Start Training =======')
    
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              save_best_model=True,
              checkpoint_save_steps=evaluation_steps,
              optimizer_params={'lr':lr},
              checkpoint_save_total_limit=3,
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)
    
    return model, evaluator


def predict_labels(df, best_model, clm1, clm2, output_clm, threshold):

    encoded_premises = best_model.encode(df[clm1].tolist())
    encoded_conclusions = best_model.encode(df[clm2].tolist())
    
    scores = [util.pytorch_cos_sim(x[0], x[1]).item() for x in zip(encoded_conclusions, encoded_premises)]
    labels = [1 if x > threshold else -1 for x in scores]
    
    df[output_clm] = labels
    
    return df