"""
Text, Image Pre_processing functions are defined here
"""

import torch
from transformers import *

MODELS = {
    "bert_pretrained": [BertModel, BertTokenizer, 'bert-base-uncased']
}


class TextPreprocessor:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model, self.tokenizer, self.pretrained_weights = MODELS[model_name]
        self.tokenizer = self.tokenizer.from_pretrained(self.pretrained_weights)
        self.model = self.model.from_pretrained(self.pretrained_weights).to(self.device)

    def get_embedding(self, text: str):
        input_tensor = torch.LongTensor([self.tokenizer.encode(text)]).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # last hidden state with shape: [batch_size(1), sequence_length, hidden_size(768)]
        output_tensor = output_tensor[0]  # Returning only last hidden state
        return output_tensor


if __name__ == '__main__':
    processor = TextPreprocessor('bert_pretrained')
    input_text = "Hi My name is Dokyoung."
    output = processor.get_embedding(input_text)  # get tensor with shape: [batch_size(1), seq_length, hidden_size(768)]

