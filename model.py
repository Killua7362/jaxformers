from utils import Module,Module_list
import jax
from modules import MultiHeadAttention,FeedForward,TransformerLayer,PosEmbeddings,LayerNorm,Linear
import jax.numpy as jnp
from loss import CrossEntropy
class Transformer(Module):
    def __init__(self,rnd_keys,n_vocab,emb_size,n_layers,heads,d_ff):
        super().__init__()
        self.n_vocab = n_vocab
        self.emb_size = emb_size
        self.loss_func = CrossEntropy()
        
        layers = []
        for i in range(n_layers):
            rnd_key,mha_key,ffn_key = jax.random.split(rnd_keys,3)
            attn = MultiHeadAttention(mha_key,heads,emb_size)
            ffn = FeedForward(ffn_key,emb_size,d_ff)
            layers.append(TransformerLayer(emb_size,attn,ffn))
        self.layers = Module_list(layers)
        
        rnd_key,emb_key,out_key = jax.random.split(rnd_key,3)
        self.embeddings = PosEmbeddings(emb_key,n_vocab,emb_size)
        self.norm = LayerNorm([emb_size])
        self.output = Linear(out_key,emb_size,n_vocab)
    
    def __call__(self,x):
        seq_len = len(x)
        mask = jnp.tril(jnp.ones((seq_len,seq_len),bool))
        x = self.embeddings(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x,mask)
        return self.output(self.norm(x))
    
    def get_loss(self,x):
        output = self(x)
        return self.loss_func(output[:-1],x[1:])
    
    def sample(self,seq,length):
        for i in range(length):
            idx = jnp.argmax(self(seq)[-1])
            seq = jnp.concatenate((seq,idx[None]))
        return seq