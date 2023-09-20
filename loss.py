from utils import Module
import jax

class CrossEntropy(Module):
    def __init__(self):
        super().__init__()
        self._loss_vmap =  jax.vmap(self._loss,in_axes=(0,0))
        
    def _loss(self,output,target):
        return -jax.nn.log_softmax(output)[target]
    
    def __call__(self,output,target):
        return self._loss_vmap(output,target).mean()
    
