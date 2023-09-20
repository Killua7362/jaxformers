from typing import NamedTuple
import jax.numpy as jnp
import jax 
from functools import partial

class AdamState(NamedTuple):
    m:jnp.ndarray
    v:jnp.ndarray

class Adam:
    def __init__(self,params,lr=0.001,betas=(0.9,0.999),eps=1e-6):
        super().__init__()
        self.lr = lr
        self.betas= betas
        self.eps= eps
        self.states = jax.tree_multimap(self._init_state,params)
        self._step_jit = jax.jit(self._step)
        self._n_steps = 0
        self._update_state_jit = jax.jit(self._update_state)
        
    def _init_state(self,param):
        return AdamState(jnp.zeros_like(param),jnp.zeros_like(param))
    
    def step(self,params,grads):
        self._n_steps += 1
        self.states = jax.tree_multimap(self._update_state_jit,grads,self.states)
        return jax.tree_multimap(partial(self._step_jit,self._n_steps),params,self.states)
    
    def _step(self,n_steps,param,state):
        bias_correction = [1-beta ** n_steps for beta in self.betas]
        m,v = state 
        step_size = self.lr * (bias_correction[1] ** 0.5) /bias_correction[0]
        den = (v ** 0.5) + self.eps
        return param - step_size * m / den
    
    def _update_state(self,grad,state):
        m,v = state
        grad = jnp.clip(grad,-1,1)
        m = self.betas[0] * m + grad * (1-self.betas[0])
        v = self.betas[1] * v + (grad ** 2) *  (1-self.betas[1])
        return AdamState(m,v)
