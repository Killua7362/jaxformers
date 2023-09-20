import jax
from dataset import TinyShakeShere
from model import Transformer
from activation import Adam
import jax.numpy as jnp

def train():
    rnd_key = jax.random.PRNGKey(0)
    dataset = TinyShakeShere(rnd_key,seq_len=32,batch_size=128)
    model = Transformer(rnd_key,dataset.n_tokens,emb_size=128,n_layers=3,heads=8,d_ff=512)
    params = model.get_params()
    pure_sample_fn = jax.jit(model.purify(model.sample))
    pure_forward_fn = jax.jit(jax.vmap(model.purify(model.__call__),
                                       in_axes=(None, 0), out_axes=0))
    pure_loss_fn = jax.jit(jax.vmap(model.purify(model.get_loss),
                                    in_axes=(None, 0), out_axes=0))
    def get_loss(params,seq):
        return pure_loss_fn(params,seq).mean()
    grad_loss_fn = jax.jit(jax.grad(get_loss,argnums=0))
    optimizer = Adam(params)
    
    for epochs in range(32):
        for data in dataset:
            loss = get_loss(params,data)
            grads = grad_loss_fn(params,data)
            params = optimizer.step(params.grad)
            
        prompt = [dataset.stoi[c] for c in 'It ']
        sampled = pure_sample_fn(params,jnp.array(prompt))[len(prompt):]
        sampled = ''.join([dataset.itos[i] for i in sampled])
        sampled = sampled.replace('\n','\\n')
        print(sampled)