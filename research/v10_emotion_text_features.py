"""V10: frozen external emotion text features; no IEMOCAP labels in encoder."""
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

MODEL_ID = "SamLowe/roberta-base-go_emotions"
REVISION = "b3a4ef6cd1548efa18b8dc1483f28ad58538ab63"
EXTRA_DIM = 768
MODES = ("original", "current", "speaker_context")


def session_split(keys):
    """Prespecified session 4 held out; require original training sessions 1-4."""
    sessions = []
    for key in keys:
        m = re.match(r"Ses0([1-5])[FM]", str(key))
        if m is None:
            raise ValueError(f"unexpected IEMOCAP dialogue ID: {key}")
        sessions.append(int(m.group(1)))
    if set(sessions) != {1, 2, 3, 4}:
        raise ValueError(f"expected original TRAIN sessions 1-4, found {set(sessions)}; stop")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dialogue ID")
    return ([i for i,s in enumerate(sessions) if s != 4],
            [i for i,s in enumerate(sessions) if s == 4])


def text_inputs(sentences, speakers):
    """Return target and past-only history, no gold labels or future turns.

    History is up to four preceding retained utterances with explicit roles.
    Newest first prioritizes recent evidence under the token limit.
    """
    if len(sentences) != len(speakers):
        raise ValueError("utterances/speakers not aligned")
    if any(not isinstance(s, str) for s in sentences):
        raise ValueError("videoSentence must contain raw text strings")
    if any(sp not in ("M", "F") for sp in speakers):
        raise ValueError("IEMOCAP speaker IDs must be M/F")
    out = []
    for i, sentence in enumerate(sentences):
        history = []
        for j in range(i-1, max(-1, i-5), -1):
            role = "SELF" if speakers[j] == speakers[i] else "OTHER"
            history.append(f"{role}: {sentences[j]}")
        out.append((sentence.strip() or ".", " | ".join(history)))
    return out


def pack_tokens(tokenizer, target, history, max_length=256):
    """Preserve target tokens before truncating preceding context."""
    special = tokenizer.num_special_tokens_to_add(pair=bool(history))
    ids = tokenizer.encode(target, add_special_tokens=False,
                           truncation=True, max_length=128)
    if not history:
        return {"input_ids": tokenizer.build_inputs_with_special_tokens(ids)}
    budget = max_length - special - len(ids)
    if budget < 1:
        raise ValueError("no history token budget")
    past = tokenizer.encode(history, add_special_tokens=False,
                            truncation=True, max_length=budget)
    return {"input_ids": tokenizer.build_inputs_with_special_tokens(ids, past)}


@torch.no_grad()
def extract_text_features(ds, device, batch_size=32, local_only=False):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    kwargs = dict(revision=REVISION, local_files_only=local_only,
                  trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    encoder = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, use_safetensors=True, **kwargs).to(device).eval()
    encoder.requires_grad_(False)
    if encoder.config.hidden_size != EXTRA_DIM:
        raise RuntimeError("unexpected external encoder hidden dimension")
    cache = {"current": {}, "speaker_context": {}}
    for i, vid in enumerate(ds.keys):
        sentences = ds.videoSentence[vid]
        speakers = ds.videoSpeakers[vid]
        pairs = text_inputs(sentences, speakers)
        if not pairs:
            raise ValueError(f"empty dialogue: {vid}")
        # Only original TRAIN keys are iterated; no official test dialogue.
        for mode in cache:
            features = []
            for start in range(0, len(pairs), batch_size):
                rows = pairs[start:start+batch_size]
                items = [pack_tokens(tokenizer, t, h if mode=="speaker_context" else "")
                         for t,h in rows]
                batch = tokenizer.pad(items, padding=True, return_tensors="pt",
                                      return_attention_mask=True)
                batch = {k:v.to(device) for k,v in batch.items()}
                out = encoder(**batch, output_hidden_states=True, return_dict=True)
                x = out.hidden_states[-1][:,0].float().cpu()
                if not torch.isfinite(x).all():
                    raise RuntimeError("nonfinite external representation")
                features.append(x)
            cache[mode][vid] = torch.cat(features)
        if (i+1)%20==0 or i+1==len(ds.keys):
            print(f"Extracted {i+1}/{len(ds.keys)} original TRAIN dialogues", flush=True)
    return cache


class FeatureDataset(Dataset):
    """Append extra text features; preserve all other CSS inputs and indices."""
    def __init__(self, ds, mode, cache):
        if mode not in MODES:
            raise ValueError("invalid feature mode")
        self.ds, self.mode, self.cache = ds, mode, cache
        self.collate_fn = ds.collate_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        text, visual, audio, speaker, mask, label, vid = self.ds[i]
        if self.mode != "original":
            extra = self.cache[self.mode][vid]
            if extra.shape != (len(text), EXTRA_DIM):
                raise ValueError(f"external feature alignment mismatch: {vid}")
            # Per-turn norm match is label-free; no fit on validation statistics.
            extra = F.normalize(extra, dim=-1) * text.norm(dim=-1, keepdim=True)
            text = torch.cat([text, extra], dim=-1)
        return text, visual, audio, speaker, mask, label, vid


def expand_text_projection(model):
    """Preserve all common CSS initialization; new input weights start at zero."""
    original = model.textf_input
    if original.in_channels != 1024 or original.kernel_size != (1,) or original.bias is not None:
        raise ValueError("unexpected CSS text projection")
    expanded = nn.Conv1d(1024 + EXTRA_DIM, original.out_channels, 1, bias=False)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:,:1024].copy_(original.weight)
    model.textf_input = expanded
    return model
