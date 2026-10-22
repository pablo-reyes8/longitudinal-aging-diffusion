"""Optional DINOv2 robustness space; never the primary bias descriptor."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .features import stable_hash


class DinoV2Encoder:
    def __init__(self,model_id='facebook/dinov2-small',*,local_files_only=True,device='cpu',cache_dir=None):
        import torch
        from transformers import AutoImageProcessor,AutoModel
        self.processor=AutoImageProcessor.from_pretrained(model_id,local_files_only=local_files_only,cache_dir=cache_dir)
        self.model=AutoModel.from_pretrained(model_id,local_files_only=local_files_only,cache_dir=cache_dir).eval().requires_grad_(False).to(device)
        self.device=device;self.cache_id=model_id+':'+str(getattr(self.model.config,'_commit_hash',None))

    def to(self,device):
        self.model.to(device);self.device=device
        return self

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
        return self.fit_vectors(encoded,splits=splits)

    def fit_vectors(self,encoded,*,splits):
        encoded=np.asarray(encoded,float)
        if set(splits)-{'train','proxy'} or len(splits)!=len(encoded):raise ValueError('PCA fit accepts train + proxy only, never test')
        if encoded.ndim!=2 or not np.isfinite(encoded).all():raise ValueError('PCA requires finite encoded vectors')
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

    def save(self,path):
        np.savez_compressed(path,mean=self.mean,components=self.components,encoder_id=np.array(self.encoder.cache_id))

    def load(self,path):
        with np.load(path,allow_pickle=False) as data:
            if 'encoder_id' not in data or str(data['encoder_id'])!=self.encoder.cache_id:
                raise ValueError('PCA encoder differs from saved reference encoder; refit on train+proxy')
            self.mean=data['mean'].copy();self.components=data['components'].copy()
        if self.components.shape[0]!=self.n_components:raise ValueError('PCA dimension mismatch')
        self._update_id();return self


class CachedEncoder:
    """Content-addressed CPU vectors; bounded batches, no repeated fit encoding."""
    def __init__(self,encoder,root):
        self.encoder=encoder;self.cache_id=encoder.cache_id
        self.root=Path(root)/stable_hash(self.cache_id);self.root.mkdir(parents=True,exist_ok=True)

    def to(self,device):
        if hasattr(self.encoder,'to'):self.encoder.to(device)
        return self

    def encode_batch(self,images):
        output=[None]*len(images);pending=[]
        for i,image in enumerate(images):
            digest=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()+str(image.size).encode()).hexdigest()
            path=self.root/(digest+'.npy')
            if path.is_file():output[i]=np.load(path,allow_pickle=False)
            else:pending.append((i,image,path))
        if pending:
            values=self.encoder.encode_batch([image for _,image,_ in pending])
            if len(values)!=len(pending):raise ValueError('Encoder batch length mismatch')
            for (i,_,path),value in zip(pending,values):
                value=np.asarray(value,float)
                if value.ndim!=1 or not np.isfinite(value).all():raise ValueError('Nonfinite/invalid embedding vector')
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as target:np.save(target,value,allow_pickle=False)
                temp.replace(path);output[i]=value
        return np.asarray(output)


def fit_reference_embedding(config,colombia,proxy,encoder,landmark_backend):
    """Fit/reuse bounded PCA using common crops of train+proxy, never test."""
    from dataclasses import asdict
    from .preprocessing import preprocess_image
    from src.quantitative_metrics.image_io import ImageResolver
    if config.embedding_batch_size<1 or config.embedding_fit_max_images<2:
        raise ValueError('Need positive embedding batch size and >=2 PCA fit images')
    root=Path(config.output_dir);root.mkdir(parents=True,exist_ok=True)
    encoder=CachedEncoder(encoder,root/'embedding_vector_cache')
    backend=EmbeddingFeatureBackend(encoder,config.embedding_components)
    candidates=[]
    for split,frame in [('train',colombia),('proxy',proxy)]:
        unique=frame.drop_duplicates('image_path')
        count=min(len(unique),config.embedding_fit_max_images//2)
        for row in unique.sample(count,random_state=config.seed).to_dict('records'):
            candidates.append((split,row['image_path']))
    resolver=ImageResolver();content=[]
    for split,path in candidates:
        _,digest=resolver.load(path);resolver._cache.clear();content.append((split,digest))
    task=config.features.landmark_model_path
    task_hash=hashlib.sha256(Path(task).read_bytes()).hexdigest() if task and Path(task).is_file() else None
    signature=stable_hash(dict(content=content,features=asdict(config.features),task=task_hash,encoder=encoder.cache_id,
        landmarks=getattr(landmark_backend,'cache_id',config.features.landmark_backend_id),components=config.embedding_components))
    pca=root/'embedding_pca.npz';meta=root/'embedding_pca.json'
    if config.resume and pca.is_file() and meta.is_file() and json.loads(meta.read_text()).get('signature')==signature:
        return backend.load(pca)
    vectors=[];splits=[];batch=[];batch_splits=[];failures=[]
    def flush():
        if not batch:return
        vectors.extend(encoder.encode_batch(batch));splits.extend(batch_splits)
        batch.clear();batch_splits.clear()
    for i,(split,path) in enumerate(candidates):
        try:
            image,_=resolver.load(path);resolver._cache.clear()
            prepared=preprocess_image(image,landmark_backend,config.features)
            if not prepared.covariates['pose_quality_ok'] or not prepared.covariates['expression_quality_ok']:
                raise ValueError('Pose/expression quality exclusion from PCA fit')
            batch.append(prepared.image);batch_splits.append(split)
        except (OSError,ValueError,TypeError) as exc:failures.append(dict(split=split,image_path=str(path),error=str(exc)))
        if len(batch)>=config.embedding_batch_size:flush()
        if (i+1)%100==0:print(f'[DINOv2 PCA] {i+1}/{len(candidates)} candidates, {len(failures)} exclusions',flush=True)
    flush()
    pd.DataFrame(failures,columns=['split','image_path','error']).to_csv(root/'embedding_fit_failures.csv',index=False)
    if set(splits)!={'train','proxy'}:raise ValueError('PCA needs quality-valid images from both train and proxy')
    backend.fit_vectors(vectors,splits=splits);backend.save(pca)
    meta.write_text(json.dumps(dict(signature=signature,encoder=encoder.cache_id,n_train=splits.count('train'),
        n_proxy=splits.count('proxy'),n_excluded=len(failures),fit_scope='Colombian train + FFHQ proxy only'),indent=2))
    return backend
