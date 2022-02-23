import pdb
import os
import json 
import argparse
from tqdm import tqdm

import torch 
import datasets
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BookSumDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, model, tokenizer, args):
    self.args = args
    self.model_max_length = tokenizer.model_max_length
    self.tokenizer = tokenizer
    self.pad_token_id = self.tokenizer.encode([self.tokenizer.pad_token])[0]
    self.dataset = dataset

    for entry in tqdm(dataset):
      text = entry['text']
      text_ids = self.tokenizer.encode(text, truncation=False, return_tensors='pt')
      chunks = []
      chunk_start = 0
      chunk_end = int(self.model_max_length)
      ### TODO: We need to make the chunks overlap to have more info!
      while chunk_start < len(text_ids[0]):
        chunk_end = min(chunk_end, len(text_ids[0]))
        chunk_ids = torch.full((1, self.model_max_length), self.pad_token_id)
        chunk_ids[0, :chunk_end-chunk_start] = text_ids[0][chunk_start:chunk_end]
        chunk_ids = torch.unsqueeze(chunk_ids, 0)
        chunks.append(chunk_ids)
        chunk_start += int(self.model_max_length)
        chunk_end += int(self.model_max_length)
      chunks_ids = torch.cat(chunks, dim=0)
      chunks_ids = chunks_ids.squeeze(1)
      chunks_ids = chunks_ids.to(device) # size=(6, 1024)
      
      chunk_summaries = []
      for i in range(0, len(chunks_ids), self.args.mini_batch_size):
        generated_ids = model.generate(
                                       chunks_ids[i:i+self.args.mini_batch_size], 
                                       num_beams=self.args.num_beams, 
                                       max_length=self.args.max_output_len, 
                                       early_stopping=True)
        chunk_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
        chunk_summaries.extend(chunk_summary)
      entry['downsized_text'] = '\n'.join(chunk_summaries)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    ID = self.list_IDs[index]
    X = torch.load('data/' + ID + '.pt')
    y = self.labels[ID]
    return X, y


def get_parser():
  parser = argparse.ArgumentParser(description="Test Summarization Models")
  parser.add_argument("--seed", type=int, default=1234, help="Seed")
  parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
  parser.add_argument("--max_input_len", type=int, default=1024, help="maximum num of wordpieces in the input")
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
  parser.add_argument("--mini_batch_size", type=int, default=10, help="Batch size")
  parser.add_argument("--num_beams", type=int, default=4, help="Batch size")
  return parser



if __name__=='__main__':
  parser = get_parser()
  args = parser.parse_args()

  print('Load the BART model...')
  model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
  print('Load the BART tokenizer...')
  model = model.to(device)
  print('Load the BART tokenizer...')
  tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

  print('Load the full BookSum dataset...')
  booksum_dataset = datasets.load_dataset('/home/ss2753/tune-LED/booksum_datasets.py', 'chapters')

  print('Create the train dataset...')
  train_dataset = BookSumDataset(booksum_dataset['train'], model, tokenizer, args)






