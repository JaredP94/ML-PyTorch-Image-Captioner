import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # |-- configure network parameters --|
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # |-- configure network architecture --|
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM takes embedded word vectors (of embed_size) as inputs and outputs hidden states of hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # linear layer that maps the hidden state output dimension to the number of words (vocab) we want as output, vocab_size
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # initialise weights for training
        self.init_weights()
        
    def init_weights(self):
        ''' Initialize weights for embeddings and fully connected layers '''
        # Interesting discussion on why initialising with 0's or 1's does not help initial training accuracy:
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
    
    def forward(self, features, captions):        
        # initialize the hidden state
        self.hidden = self.init_hidden(features.size(0))
        
        # embed captions through embeddings layer
        embedded_captions = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embedded_captions), 1)
        
        # feed embeddings through LSTM layer
        outputs, self.hidden = self.lstm(embeddings)
        
        # return outputs of fully connected layer
        return self.fc(outputs)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "  
        predictions = []
        
        for idx in range(max_len):
            # feed inputs through LSTM layer
            outputs, states = self.lstm(inputs, states)
            
            # feed outputs through fully connected layer
            outputs = self.fc_out(outputs)
            outputs = outputs.squeeze(1)
            
            # calculate the most likely word
            word_idx = outputs.argmax(1)
            if word_idx.item() == 1:
                break
          
            # add predicted word to resulting sentence
            predictions.append(word_idx.item())
            
            # next input becomes embedded word of current iteration
            inputs = self.embed(word_idx)
            inputs = inputs.unsqueeze(1)
            
        # return predicted caption    
        return predictions 
    