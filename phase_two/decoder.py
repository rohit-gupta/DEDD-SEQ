import torch
import torch.nn as nn
from einops import rearrange
from model.tests.transformer_vitbb import PositionalEncoder

class XDecoder(nn.Module):
    def __init__(self, config, device):
        super(XDecoder, self).__init__()
        
        self.config = config
        self.gps_dim = 512
        self.n_features = 512
        
        self.batch_size = self.config['train_batch_size']
        self.views = self.config['train_views']

        self.num_encoder_layers = self.config['n_encoder'] 
        self.num_decoder_layers = self.config['n_decoder'] 

        self.pos_enc = PositionalEncoder(512, 0.1, seq_len=16)
        
        #self.queries_mask = torch.tril(torch.ones(self.config['train_views'], self.config['train_views'])).to(device)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features, nhead=8)
        
        if self.config['n_encoder'] != 0:
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_encoder_layers)
        else:
            self.encoder = nn.Identity()
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.n_features, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_decoder_layers)
        
        self.device = device
     
    def forward(self, img_features, gps_features, mode='train'):
        if mode == 'eval':
            batch_size = 1
            views = img_features.shape[0]
            #self.queries_mask = torch.tril(torch.ones(self.views, self.views)).to(self.device)
        elif mode == 'train':
            batch_size = self.config['train_batch_size']
            views = self.config['train_views']
            

        # Recives img_features ([Batch*Frames, 512]) so reshape to first ([Batch, Frames, 512])
        img_features = torch.reshape(img_features, (batch_size, views, -1))

        #Pos Emb
        #img_features = self.pos_enc(img_features)
        
        # Reshape img_features from ([Batch, Frames, 512]) to ([Frames, Batch, 512])
        #img_features = rearrange(torch.reshape(img_features, (self.batch_size, self.views, -1)), 'Batch Frames Dim -> Frames Batch Dim')
        img_features = img_features.permute((1,0,2))
        
        # Recives gps_features ([Batch*Frames, 512]) so reshape to first ([Batch, Frames, 512])
        gps_features = torch.reshape(gps_features, (batch_size, views, -1))

        #Pos Emb
        #gps_features = self.pos_enc(gps_features)
        
        # Reshape gps_features from ([Batch, Frames, 512]) to ([Frames, Batch, 512])
        #gps_features = rearrange(torch.reshape(gps_features, (self.batch_size, self.views, -1)), 'Batch Frames Dim -> Frames Batch Dim') #[:-1,:,:]
        gps_features = gps_features.permute((1,0,2))

        

        # gps_features are queries
        queries = gps_features


        
        # pass img_features to encoder of XDecoder and get img_features for decoder layer with shape ([Frames, Batch, 512])
        memory = self.encoder(img_features)

        # pass queries (gps_features), encoded img_features, and queries mask to decoder of XDecoder to get decoded gps_features
        out = self.decoder(queries, memory)
        
        # Reshape decoded gps_features from ([Frames, Batch, 512]) to ([Batch, Frames, 512])
        #out = rearrange(out, 'Frames Batch Dim -> Batch Frames Dim') 
        out = out.permute((1,0,2))
        # Reshape further decoded gps_features from ([Frames, Batch, 512]) to ([Batch*Frames, 512])
        out = torch.reshape(out, (batch_size*views, -1))

        #self.batch_size = self.config['train_batch_size']
        #self.views = self.config['train_views']
        #self.queries_mask = torch.tril(torch.ones(self.views, self.views)).to(self.device)
        
        return out
    

# Setting up a simple test scenario
def main():
    config = {'train_views': 2, 'train_batch_size': 3}  # Simplified for clarity
    device = torch.device('cuda')  # Using CPU for simplicity

    decoder = XDecoder(config, device).to(device)
    img_features = torch.randn(config['train_batch_size'] * config['train_views'], 8).to(device)
    gps_features = torch.randn(config['train_batch_size'] * config['train_views'], 8).to(device)
    
    print('img_features ', img_features)
    print('gps_features ', gps_features)
    
    output_features = decoder(img_features, gps_features, mode='train')

if __name__ == "__main__":
    main()
