import torch
import torch.nn as nn
import transformers

NUM_LABELS = 5
HIDDEN_SIZE = 256


class BERTClassifier(torch.nn.Module):
    def __init__(
        self, bert_model_name, dropout, freeze_bert, extra_layers, include_index=False
    ):
        super(BERTClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_model_name)
        self.include_index = include_index
        clf_in_size = 2 * self.bert.config.hidden_size + int(include_index)
        if extra_layers:
            # add layers to increase expressiveness
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(clf_in_size, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, NUM_LABELS),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(clf_in_size, NUM_LABELS),
            )

        if freeze_bert:
            # freeze bert layers
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x):
        # remove abstract index from bert input
        if self.include_index:
            abstract_index = x["abstract_index"]
            del x["abstract_index"]

        last_hidden_state, pooler_output = self.bert(**x, return_dict=False)

        # mean pool last hidden states and concatenate with CLS embedding
        pooled_hidden = torch.mean(last_hidden_state, 1)

        if self.include_index:
            clf_in = torch.cat((pooled_hidden, pooler_output, abstract_index), 1)
        else:
            clf_in = torch.cat((pooled_hidden, pooler_output), 1)

        logits = self.classifier(clf_in)
        return logits
