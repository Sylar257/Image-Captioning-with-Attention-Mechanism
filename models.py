# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:08:50 2019

@author: xgli001
"""

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained=True) # download weights
        
        # Remove linear and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
        
        self.fine_tune()
    
    def forward(self, images):
        
        """
        images: images: a tensor of dimensions (batch_size, 3, image_size, image_size)
        returns: encoded images
        """
        out = self.resnet(images)       # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)   # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0,2,3,1)      # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 
        through 4 of the encoder.
        
        fine_tune: True ==> allow
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If Fine-tuning is on, convolutional blocks 2 though 4 are trained
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                
class Attention(nn.Module):
    """
    Attension Network.
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        endocer_dim: feature size of encoded images
        decoder_dim: number of neurons for DecoderRNN
        attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim) # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim,1)               # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # softmax layer to calculate weights
        
    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: encoded images (batch_size, num_pixels, encoder_dim)
        decoder_hidden: previous decoder output (batch_size, decoder_dim)
        
        return: attention weighted encoding, weights
        """
        att_enc = self.encoder_att(encoder_out)     # (batch_size, num_pixels, attention_dim)
        att_dec = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att_enc+att_dec.unsqueeze(1))).squeeze(2) # (batch_size, num_pixels)
        # alpha is the pixel-wise score that are applied to the same pixel location across all feature channels
        alpha = self.softmax(att)       # (batch_size, num_pixels)
        # attention_weighted_encoding is the attention score assigned to each channel, feature channel-wise
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha
    
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        attention: size of attention network
        embed_dim: embedding size
        decoder_dim: size of DecoderRNN
        vocab_size: size of vocabulary
        encoder_dim: feature size of encoded images
        dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim) # we specify our attention network here
        # create embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        # create decoderRNN the input_size = (embeding dimmension+encoder dimmension)
        self.decode_step = nn.LSTMCell(embed_dim+encoder_dim,decoder_dim,bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)   # linear layer to find initial hidden state of LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim)   # linear layer to find initial cell state of LSTM
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)        # linear layer to find scores over vocabulary
        self.init_weights() # Initialize some layers with the uniform distribution
        
    def init_weights(self):
        """
        Initializes some parameters with values from a uniform distribution for better convergence
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings
        embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
        
    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer?
        Only makes sense to not-allow if using a pre-trained embeddings.
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
            
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the DecoderRNN based on the encoded images.
        encoder_out: the encoded images, (batch_size, num_pixels, encoder_dim)
        return:  hidden_state, cell_state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # average across the pixel channel
        h = self.init_h(mean_encoder_out)       # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)       # (batch_size, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation
        
        encoder_out: encoded images, (batch_size, enc_image_size, enc_image_size, encoder_dim)
        encoded_captions: encoded captions, (batch_size, max_caption_length)
        caption_lengths: caption lengths,   (batch_size, 1)
        
        return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size,-1,encoder_dim) # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by descreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        # We finishing generating as soon as we see <end> token
        # So, decoding lengths are (actual length -1)
        decode_lengths = (caption_lengths -1).tolist()      # (batch_size)
        
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        
        # At each time-step, decode by "feature-channel-wise attention-weighted" encoder's 
        # output based on decoder's previous hidden state output
        # then, generate a new word in the decoder with the previous word and attention 
        # weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l>t for l in decode_lengths])
            # attention_weighted_encoding is the attention score assigned to each feature channel (batch_size_t, encoder_dim)
            # alpha is the pixel-wise score
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate*attention_weighted_encoding
            # pass data through LSTM
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t,:], attention_weighted_encoding],dim=1),
                                    (h[:batch_size_t],c[:batch_size_t]))        # (batch_size, decoder_dim)
            preds = self.fc(self.dropout(h)) # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
