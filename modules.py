from modules import Module
import jax 
import jax.numpy as jnp

class Embeddings(Module):
    def __init__(self,rnd_key,n_emb,n_dim):
        super().__init__()
        self.embeddings = jax.random.normal(rnd_key,(n_emb,n_dim))
    
    def __call__(self,ids):
        return self.embeddings[ids,:]

class PosEmbeddings(Module):
    def __init__(self,rnd_key,n_vocab,emb_size,max_len=4096):
        super().__init__()
        self.embeddings = Embeddings(rnd_key,n_vocab,emb_size)
        self.p_coeffecient = 1/emb_size ** 0.5
        self.positional_encode = jnp.zeros((max_len,emb_size))
    def __call__(self,x):
        pe = self.positional_encode[:x.shape[0]]
        return self.embeddings(x) * self.p_coeffecient + pe

class Linear(Module):
    def __init__(self,rnd_key,in_features,out_features):
        super().__init__()
        rnd_range = 1/ in_features ** 0.5
        self.weight = jax.random.uniform(rnd_key,(in_features,out_features),minval=-rnd_range,maxval=rnd_range)
        self.bias = jax.zeros((out_features,))
        
    def __call__(self,x):
        return jnp.multiply(self.weight,x) + self.bias

class LayerNorm(Module):
    def __init__(self,normalized_shape,*,eps=1e-5,ele_affine=True):
        super().__init__()
        self.eps = eps
        self.ele_affine = ele_affine
        self.norm_shape = normalized_shape
        
        if ele_affine:
            self.gain = jnp.ones(normalized_shape)
            self.bias= jnp.zeros(normalized_shape) 
    
    def __call__(self,x):
        assert self.norm_shape == x.shape[-len(self.norm_shape):]
        axes = [-(i+1) for i in range(len(self.norm_shape))]
        mean = x.mean(axis=axes,keepdims=True)
        mean_2 = (x ** 2).mean(axis=axes,keepdims=True)
        var = mean_2 - mean ** 2
        x_norm = ( x - mean) / (var + self.eps) ** 0.5
        if self.ele_affine:
            x_norm = self.gain * x_norm + self.bias
        return x_norm
       
class PrepareForMultiHead(Module):
    def __init__(self,rnd_key,emb_size,heads,d_k):
        super().__init__() 
        self.linear = Linear(rnd_key,emb_size,d_k*heads)
        self.heads = heads
        self.d_k = d_k
    def __call__(self,x):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        return x.reshape(*head_shape,self.heads,self.d_k)

class MultiHeadAttention(Module):
    def __init__(self,rnd_key,heads,emb_size):
        super().__init__()
        _,*rnd_keys = jax.random.split(rnd_key,5)
        self.d_k = emb_size//heads
        self.heads = heads
        
        self.query = PrepareForMultiHead(rnd_key[0],emb_size,heads,self.d_k)
        self.key= PrepareForMultiHead(rnd_key[1],emb_size,heads,self.d_k)
        self.value= PrepareForMultiHead(rnd_key[2],emb_size,heads,self.d_k)
        
        self.output = Linear(rnd_key[3],emb_size,emb_size)
        self.scale = 1/self.d_k ** 0.5
    
    def __call__(self,*,query,key,value,mask=None):
        seq_len = len(query)
        if mask is not None:
            assert mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]
            
            mask = mask[:,:,None]
        query = self.query(query)
        key= self.key(key)
        value= self.value(value)
        scores = jnp.einsum('ihd,jhd->ijh',query,key) ##TODO
        scores *= self.scale
        if mask is not None:
            scores = scores + (mask == 0) * float('-inf')
        attn = jax.nn.softmax(scores,axis=1)
        x = jnp.einsum('ijh,jhd->ihd',attn,value)
        x = x.reshape(seq_len,-1)
        return self.output(x)

class FeedForward(Module):
    def __init__(self,rnd_keys,emb_size,d_ff,activation = jax.nn.relu):
        super().__init__()
        _ , *rnd_keys = jax.random.split(rnd_keys,5)
        
        self.layer1 = Linear(rnd_keys[0],emb_size,d_ff)
        self.layer2 = Linear(rnd_keys[1],d_ff,emb_size)
        self.activation = activation
        
    def __call__(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)

class TransformerLayer(Module):
    def __init__(self,emb_size,self_attn,feed_forward):
        super().__init__()
        self.size = emb_size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_self_attn = LayerNorm([emb_size])
        self.norm_ff = LayerNorm([emb_size])
    
    def __call__(self,x,mask):
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z,key=z,value=z,mask=mask)
        x = x + self_attn
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + ff
        return x
