import jax 
import jax.numpy as jnp
class TinyShakeShere:
    def __init__(self,rnd_key,seq_len,batch_size):
        self.batch_size = batch_size
        _,self.rnd_key = jax.random.split(rnd_key)
        with open(str(path),'r') as f:
            self.text = f.read()
        tokens = sorted(list(set(self.text)))
        self.n_tokens = len(tokens)
        self.stoi = {t:i for i,t in enumerate(tokens)}
        self.itos = tokens
        data = jnp.array([self.stoi[s]] for s in list(self.text))
        self.n_batches = len(data) // (seq_len * batch_size)
        data = data[:self.n_batches * seq_len * batch_size]
        self.data = data.reshape((-1,seq_len))
        self.idx = jnp.arrange(len(self.data))
        
    def __iter__(self):
        self._iter_idx = 0
        self.rnd_key,rnd_key = jax.random.split(self.rnd_key)
        self.idx = jax.random.permutation(rnd_key,self.idx)
        return self

    def __len__(self):
        return self.n_batches
    
    def __next__(self):
        if self._iter_idx >= self.n_batches:
            raise StopIteration()

        idx = self.idx[self._iter_idx * self.batch_size:(self._iter_idx+1) * self.batch_size]
        self._iter_idx +=1
        return self.data[idx]
    
path = '/workspaces/peronal_projects/jaxformers/input.txt'