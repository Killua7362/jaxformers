from typing import TypeVar,List,Generic
import jax.numpy as jnp

class Module:
    def __init__(self):
        self.parmas = {}
        self.submodules = {}
    
    def __getattr__(self,attr_name):
        if attr_name in self.params:
            return self.parmas[attr_name]
        elif attr_name in self.submodules:
            return self.submodules[attr_name]
        else: 
            return self.__dict__[attr_name] 

    def __setattr__(self,key,value):
        if isinstance(value,Module):
            self.submodules[key] = value
        elif isinstance(value,jnp.ndarray):
            self.params[key]= value
        else: 
            self.__dict__[key] = value
            
    def _clear_params(self):
        self.parmas = {}
        for i in self.submodules.values():
            i._clear_params()
    
    def get_params(self):
        params = self.parmas.copy()
        for s_name,s in self.submodules.items():
            for k,v in s.get_params().items():
                params[s_name + '/' + k] = v
        return params

    def set_params(self,params):
        for name,value in params.items():
            self.set_param(name.split('/'),value)
    
    def set_param(self,param,value):
        if len(param) == 1:
            self.parmas[param[0]] = value
        else: 
            self.submodules[param[0]].set_param(param[1:],value) 
    
    def purify(self,method):
        def pure_method(params,*args):
            self._clear_params()
            self.set_params(params)
            result = method(*args)
            return result
        return pure_method

M = TypeVar('M',bound=Module)

class Module_list(Module,Generic[M]):
    _submodules:List[M]
    def __init__(self,modules):
        super().__init__()
        self._submodules = modules
    
    def __getitem__(self,idx):
        return self._submodules[idx]
        
    def __setitem__(self,key,value):
        pass
    
    def __len__(self):
        return len(self._submodules)
    
    def __getattr__(self, item):
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def _clear_params(self):
        self._params = {}
        for sm in self._submodules:
            sm._clear_params()

    def get_params(self):
        params = self._params
        for i, sm in enumerate(self._submodules):
            for name, value in sm.get_params().items():
                params[f'{i}/{name}'] = value
        return params

    def _set_param(self, param_path, value):
        self._submodules[int(param_path[0])]._set_param(param_path[1:], value)
    
