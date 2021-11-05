from tqdm import tqdm
import pdb
import os
import argparse
import torch
import json 
import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from ignite.metrics import Rouge


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SummarizationDatasetTest(Dataset):
	def __init__(self, dataset, tokenizer, args):
		self.tokenizer = tokenizer
		self.args = args
		texts2ids = {}
		self.ids2summaries = {}
		self.ids = []
		for id in tqdm(range(len(dataset))):
			text = dataset[id]['text']
			summary = dataset[id]['summary']
			if text not in texts2ids:
				texts2ids[text] = id
				self.ids.append(id)
				self.ids2summaries[id] = []
			else:
				id = texts2ids[text]
			self.ids2summaries[id].append(summary)
		self.ids2texts = {}
		for text in texts2ids:
			self.ids2texts[texts2ids[text]] = text

		assert len(self.ids) == len(self.ids2texts) == len(self.ids2summaries)
		assert set(self.ids) == set(self.ids2texts)
		assert set(self.ids) == set(self.ids2summaries)

	def __len__(self):
		return len(self.ids2texts)

	def __getitem__(self, idx):
		id = self.ids[idx]
		text = self.ids2texts[id]
		summaries = self.ids2summaries[id]

		if "t5" in self.args.model_path:
			text = "summarize: " + text		
			text = text.strip().replace("\n","")

		input_ids = self.tokenizer.encode(text, truncation=False, max_length=None)
		input_ids = torch.tensor(input_ids)
		output_ids = []
		for summary in summaries:
			summary_ids = self.tokenizer.encode(summary, truncation=False, max_length=None)
			summary_ids = torch.tensor(summary_ids)
			output_ids.append(summary_ids)
		return input_ids, output_ids

	@staticmethod
	def collate_fn(batch):
		pad_token_id = 1
		input_ids, output_ids = list(zip(*batch))
		input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
		return input_ids, output_ids


def get_parser():
	parser = argparse.ArgumentParser(description="Test Summarization Models")
	parser.add_argument("--seed", type=int, default=1234, help="Seed")
	parser.add_argument("--max_output_len", type=int, default=120, help="maximum num of wordpieces in the summary")
	parser.add_argument("--output_file", type=str, default='./results/output.jsonl', help="Location of output file")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	parser.add_argument("--num_beams", type=int, default=4, help="Batch size")
	parser.add_argument("--model_path", default="facebook/bart-large-cnn", 
		choices=['facebook/bart-large-cnn', 't5-small', 't5-base', 't5-large'], help="Model to evaluate")
	return parser

def get_model(model_path):
	if 'led' in model_path:
		model = LEDForConditionalGeneration.from_pretrained(model_path)
		tokenizer = LEDTokenizer.from_pretrained(model_path)
	if 't5' in model_path:
		model = T5ForConditionalGeneration.from_pretrained(model_path)
		tokenizer = T5Tokenizer.from_pretrained(model_path)
	if 'bart' in model_path:
		model = BartForConditionalGeneration.from_pretrained(model_path)
		tokenizer = BartTokenizer.from_pretrained(model_path)
	return model, tokenizer

if __name__=="__main__":
	parser = get_parser()
	args = parser.parse_args()
	print('Evaluate results for')
	print(args)

	print('Load model...')
	model, tokenizer = get_model(args.model_path)
	model = model.to(device)

	print('Set up the dataset and data loader...')
	dataset = datasets.load_dataset('/home/ss2753/tune-LED/booksum_datasets.py', 'chapters')
	test_dataset = dataset['test']
	test_dataset = SummarizationDatasetTest(test_dataset, tokenizer, args)

	test_dataloader = DataLoader(test_dataset, batch_size=1, 
		shuffle=False, collate_fn=SummarizationDatasetTest.collate_fn)

	print("Setting up the rouge metric...")
	rouge = Rouge(variants=["L", 2], multiref="average")
	pad_token_id = tokenizer.encode([tokenizer.pad_token])[0]
	counter = 0
	print("Start evaluating the test data...")
	with open(args.output_file, 'w') as f:
		for (texts_ids, summaries_ids) in tqdm(test_dataloader, total=len(test_dataloader)):
			counter += 1
			# get batches of tokens corresponding to the exact model_max_length
			chunk_start = 0
			chunk_end = tokenizer.model_max_length  # == 1024 for Bart
			inputs_batch_lst = []
			
			while chunk_start < len(texts_ids[0]):
				chunk_end = min(chunk_end, len(texts_ids[0]))
				inputs_batch = torch.full((1, tokenizer.model_max_length), pad_token_id)
				inputs_batch[0, 0:chunk_end-chunk_start] = texts_ids[0][chunk_start:chunk_end]

				inputs_batch = torch.unsqueeze(inputs_batch, 0)
				inputs_batch_lst.append(inputs_batch)
				chunk_start += tokenizer.model_max_length  # == 1024 for Bart
				chunk_end += tokenizer.model_max_length  # == 1024 for Bart

			input_ids = torch.cat(inputs_batch_lst, dim=0)
			input_ids = input_ids.squeeze(1)
			input_ids = input_ids.to(device)

			summary_batch_lst = []
			for i in range(0, len(input_ids), args.batch_size):
				generated_ids = model.generate(input_ids[i:i+args.batch_size], num_beams=args.num_beams, max_length=args.max_output_len, early_stopping=True)
				summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
				summary_batch_lst.extend(summary_batch)
			predictions = '\n'.join(summary_batch_lst)
			references = [tokenizer.batch_decode(summary_ids, skip_special_tokens=True) for summary_ids in summaries_ids]

			out = json.dumps({"references":references[0], "prediction":predictions[0]})
			f.write(out+'\n')

			predictions = predictions[0].split()
			references = [ref.split() for ref in references[0]]
			rouge.update(([predictions], [references]))

		score = rouge.compute()
		print(score)
		out = json.dumps(score)
		f.write(out+'\n')
		f.flush()

'''
python test-chunking.py --max_output_len 120 \
	--output_file results/t5-large-chunking.jsonl \
	--model_path t5-large
'''