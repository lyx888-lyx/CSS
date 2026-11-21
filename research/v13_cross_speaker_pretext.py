"""V13: A-B-A predictive self-supervision without gold emotion labels.

Both arms use the same model, parameters, optimizer and training budget.
The extra task has NO test-time module or forward-pass cost:
  self  : predict A's current projected audio from previous A alone
  cross : additionally use mean intervening B projected audio

The predictor never receives current A's feature as an INPUT; the
current feature is only a stop-gradient target. This does not prove
causal emotion influence; it tests whether other-speaker evidence
helps representation learning beyond self-speaker continuity.
"""
import torch
from torch import nn
from torch.nn import functional as F

MODES=("self","cross")


def gather_aba_audio(projected_audio, speakers, valid_mask):
    """Return past-self, intervening-other-mean, current-target, eligible.

    Shapes:
        projected_audio [batch, time, hidden]
        speakers, valid_mask [batch, time]
    Self must have spoken before; >=1 different speaker utterance must
    occur strictly between self-previous and current. Future and
    padded turns never enter the predictor's SOURCE representation.
    No reference to gold emotion labels.
    """
    if projected_audio.ndim!=3:
        raise ValueError("projected_audio must be [B,L,D]")
    B,L,D=projected_audio.shape
    if speakers.shape!=(B,L) or valid_mask.shape!=(B,L):
        raise ValueError("speaker and mask must be [B,L]")
    if (speakers.device!=projected_audio.device or
            valid_mask.device!=projected_audio.device):
        raise ValueError("device mismatch")
    valid=valid_mask.bool()
    t=torch.arange(L,device=projected_audio.device)
    earlier=(t[None,:]<t[:,None])
    same=(speakers[:,:,None]==speakers[:,None,:])
    previous=(earlier[None,:,:]&same&
              valid[:,:,None]&valid[:,None,:])
    prev_ids=t[None,None,:].expand(B,L,L).masked_fill(~previous,-1).max(-1).values
    previous_audio=projected_audio.gather(
        dim=1,index=prev_ids.clamp_min(0).unsqueeze(-1).expand(B,L,D)
    )
    between=(earlier[None,:,:]
             & (t[None,None,:]>prev_ids[:,:,None])
             & (~same)&valid[:,:,None]&valid[:,None,:]
             & (prev_ids[:,:,None]>=0))
    count=between.sum(-1)
    eligible=(prev_ids>=0)&(count>0)&valid
    between_audio=(
        torch.bmm(between.to(projected_audio.dtype),projected_audio)
        /count.clamp_min(1).unsqueeze(-1)
    )
    return previous_audio,between_audio,projected_audio,eligible


class CrossSpeakerPredictivePretext(nn.Module):
    """Same-capacity self/cross predictors trained with InfoNCE."""

    def __init__(self,dim=1024,bottleneck=256,mode="cross",temperature=0.2):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"invalid mode {mode}")
        if temperature<=0:
            raise ValueError("temperature must be positive")
        self.mode=mode
        self.temperature=float(temperature)
        self.predictor=nn.Sequential(
            nn.Linear(dim*2,bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck,dim),
        )

    def forward(self, projected_audio, speakers, valid_mask):
        previous,between,current,eligible=gather_aba_audio(
            projected_audio,speakers,valid_mask
        )
        if int(eligible.sum().item())<2:
            return projected_audio.sum()*0.0
        x=previous[eligible]
        other=between[eligible]
        if self.mode=="self":
            other=torch.zeros_like(other)
        source=torch.cat([x,other],dim=-1)
        prediction=F.normalize(self.predictor(source),p=2,dim=-1)
        targets=F.normalize(current[eligible].detach(),p=2,dim=-1)
        scores=prediction @ targets.T / self.temperature
        identity=torch.arange(scores.shape[0],device=scores.device)
        return F.cross_entropy(scores,identity)


class V13LossHook:
    """Callable for the optional original train_IEMOCAP trainer hook."""

    def __init__(self,head,coefficient=0.05):
        if coefficient<=0:
            raise ValueError("coefficient must be positive")
        self.head=head
        self.coefficient=float(coefficient)
        self.last_aux_loss=None
        self.last_triplets=None

    def __call__(self,model,audio,qmask,umask):
        # Original trainer input audio is [L,B,1582], while projected
        # audio and the sequential masks below are batch-major.
        projected=model.acouf_input(audio.permute(1,2,0)).transpose(1,2)
        speakers=qmask.argmax(-1)
        _,_,_,eligible=gather_aba_audio(projected,speakers,umask.bool())
        raw=self.head(projected,speakers,umask.bool())
        self.last_aux_loss=float(raw.detach().item())
        self.last_triplets=int(eligible.sum().item())
        return raw*self.coefficient
