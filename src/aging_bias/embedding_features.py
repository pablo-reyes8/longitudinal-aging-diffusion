"""Optional DINOv2 robustness space; never the primary bias descriptor."""
from pathlib import Path
import numpy as np
from .features import stable_hash


class DinoV2Encoder:
    def __init__(self,model_id='facebook/dinov2-small',*,local_files_only=True,device='cpu'):
        import torch
        from transformers import AutoImageProcessor,AutoModel
        self.processor=AutoImageProcessor.from_pretrained(model_id,local_files_only=local_files_only)
        self.model=AutoModel.from_pretrained(model_id,local_files_only=local_files_only).eval().requires_grad_(False).to(device)
        self.device=device;self.cache_id=model_id+':'+str(getattr(self.model.config,'_commit_hash',None))

    def encode_batch(self,images):
        import torch
        with torch.inference_mode():
            inputs={k:v.to(self.device) for k,v in self.processor(images=images,return_tensors='pt').items()}
            return self.model(**inputs).last_hidden_state[:,0].cpu().numpy()


class EmbeddingFeatureBackend:
    """PCA is fitted on aligned Colombian train + proxy images, never test.

    Feed common-preprocessed crops to fit(); FeatureStore applies exactly the
    same preprocessing before extract_aligned. Persist PCA and use its hash in
    every cache key; changing PCA must invalidate all evaluation features.
    """
    def __init__(self,encoder,n_components=16):
        self.encoder=encoder;self.n_components=int(n_components)
        self.mean=None;self.components=None;self.cache_id='unfitted-dinov2-pca'
        self.attribute_ids=tuple('embedding_'+str(i) for i in range(self.n_components))

    def fit(self,aligned_images,*,splits,batch_size=16):
        if set(splits)-{'train','proxy'} or len(splits)!=len(aligned_images):raise ValueError('PCA fit accepts train + proxy only, never test')
        encoded=np.concatenate([self.encoder.encode_batch(aligned_images[i:i+batch_size]) for i in range(0,len(aligned_images),batch_size)])
        if min(encoded.shape)<=self.n_components:raise ValueError('Insufficient train+proxy samples for PCA dimension')
        self.mean=encoded.mean(0);_,_,vt=np.linalg.svd(encoded-self.mean,full_matrices=False)
        self.components=vt[:self.n_components];self._update_id()
        return self

    def _update_id(self):
        self.cache_id=stable_hash(dict(encoder=self.encoder.cache_id,mean=self.mean.tolist(),components=self.components.tolist()))

    def extract_aligned(self,image,landmarks,allow_color=True):
        if self.components is None:raise ValueError('Fit PCA on train+proxy before reference/evaluation extraction')
        vec=(self.encoder.encode_batch([image])[0]-self.mean)@self.components.T
        return {'phi__'+a:float(v) for a,v in zip(self.attribute_ids,vec)}

    def save(self,path):np.savez_compressed(path,mean=self.mean,components=self.components)

    def load(self,path):
        with np.load(path,allow_pickle=False) as data:self.mean=data['mean'].copy();self.components=data['components'].copy()
        if self.components.shape[0]!=self.n_components:raise ValueError('PCA dimension mismatch')
        self._update_id();return self
