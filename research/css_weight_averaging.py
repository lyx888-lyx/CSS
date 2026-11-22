"""Frozen-trajectory CSS SWA/EMA parameter averaging (no new network).

The original CSS optimizer, model, PGM loss, data and train step
remain unchanged. Track CPU copies of end-of-epoch state dictionaries:
  EMA: epochs 100–200 inclusive, decay=0.90 once per epoch.
  SWA: uniform arithmetic mean of epochs 151–200 inclusive.
At epoch 200 save raw/EMA/SWA from the SAME original CSS run.
These are prespecified exploratory settings, not TEST-fitted.
"""
import torch

EMA_START=100
EMA_DECAY=0.90
SWA_START=151
FINAL_EPOCH=200


class EpochWeightAverager:
    def __init__(self,mode,start_epoch,decay=None):
        if mode not in ("ema","swa"):
            raise ValueError("mode must be ema or swa")
        if start_epoch<1:
            raise ValueError("start epoch must be >=1")
        if mode=="ema" and (decay is None or not 0.0<decay<1.0):
            raise ValueError("EMA decay must lie strictly between 0 and 1")
        self.mode=mode
        self.start=int(start_epoch)
        self.decay=decay
        self.count=0
        self.last_epoch=0
        self.shadow=None

    @torch.no_grad()
    def update(self,model,completed_epoch):
        epoch=int(completed_epoch)
        if epoch<=self.last_epoch or epoch<1:
            raise ValueError("epochs must be strictly increasing")
        self.last_epoch=epoch
        if epoch<self.start:
            return
        state=model.state_dict()
        if self.shadow is None:
            self.shadow={k:v.detach().cpu().clone() for k,v in state.items()}
            self.count=1
            return
        if set(state)!=set(self.shadow):
            raise ValueError("model state keys changed during averaging")
        next_count=self.count+1
        for key,tensor in state.items():
            new=tensor.detach().to(device="cpu")
            old=self.shadow[key]
            if old.shape!=new.shape or old.dtype!=new.dtype:
                raise ValueError(f"changed parameter/buffer spec: {key}")
            if old.is_floating_point():
                if self.mode=="ema":
                    old.mul_(self.decay).add_(new,alpha=1.0-self.decay)
                else:
                    old.add_((new-old)/next_count)
            else:
                # Integer counters are NOT arithmetically averaged.
                old.copy_(new)
        self.count=next_count

    def get_state(self):
        if self.shadow is None or self.count<1:
            raise RuntimeError("weight average not initialized")
        return {k:v.clone() for k,v in self.shadow.items()}

    def summary(self):
        return {
            "mode":self.mode,
            "start_epoch":self.start,
            "last_updated_epoch":self.last_epoch,
            "updates":self.count,
            "decay":float(self.decay) if self.mode=="ema" else None,
            "cpu_shadow":True,
        }


def frozen_cpu_state(model):
    """Detached deep-copy checkpoint; never modifies the trained network."""
    return {
        key:val.detach().cpu().clone()
        for key,val in model.state_dict().items()
    }


def verify_same_state_keys(*states):
    if not states:
        raise ValueError("at least one state dict is required")
    keys=set(states[0])
    for i,s in enumerate(states[1:],1):
        if set(s)!=keys:
            raise ValueError(f"state keys differ for {i}")
        for k in keys:
            if s[k].shape!=states[0][k].shape:
                raise ValueError(f"state shape differs: {k}")
    return True
