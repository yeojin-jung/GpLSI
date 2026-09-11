"""Sparse source contracts, immutable outer inputs, and training-only vocabularies.

No estimator receives the evaluation annotation table. All disk outputs of this
module are below ROOT/data in a joint_v2 namespace. Heavy materialization runs
through Slurm, not a login-node inventory command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, load_npz, save_npz

from .splits import array_hash, assign_roles, build_outer_splits
from .config import fingerprint

DATASETS=("visium_dlpfc","merfish_trem2_5xfad","xenium_uc")
CONTRACT_VERSION="joint_v2_data_1"
VISIUM_PANELS=(500,2000,5000,10000,15000)
SAFE_OBS=("obs_id","bio_id","section_id","graph_id","x","y","source_row")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def seed_namespace(seed: int, *parts) -> int:
    payload=json.dumps([int(seed),*map(str,parts)],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')


def integer_csr(matrix):
    a=csr_matrix(matrix);a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
    if np.any(~np.isfinite(a.data)) or np.any(a.data<0) or np.any(a.data!=np.rint(a.data)):
        raise ValueError("Count matrix must contain nonnegative finite integers")
    if a.data.max(initial=0)>np.iinfo(np.int32).max:
        raise ValueError("Count exceeds int32 capacity")
    return a.astype(np.int32)


def _atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+f'.partial.{os.getpid()}')
    partial.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    os.replace(partial,path)


def _file_records(directory,names):
    directory=Path(directory)
    return {name:{'sha256':sha256(directory/name),'size_bytes':(directory/name).stat().st_size}
            for name in names}


def _verify_files(directory,records):
    directory=Path(directory)
    for name,record in records.items():
        path=directory/name
        if not path.is_file() or path.stat().st_size!=record['size_bytes'] or sha256(path)!=record['sha256']:
            raise ValueError(f'Immutable artifact missing or checksum mismatch: {path}')


@contextmanager
def _lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def _source_metadata(root: Path,dataset: str):
    """Read metadata only (backed H5AD does not load count matrices)."""
    if dataset=='visium_dlpfc':
        path=root/'data/interim/visium_dlpfc/observations.tsv.gz'
        raw=pd.read_csv(path,sep='\t',low_memory=False)
        obs=pd.DataFrame({'obs_id':raw.sample_id.astype(str)+'::'+raw.observation_id.astype(str),
                          'bio_id':raw.subject.astype(str),'section_id':raw.sample_id.astype(str),
                          'position':raw.position,'replicate':raw.replicate,
                          'x':raw.array_col.astype(float),'y':raw.array_row.astype(float),
                          'source_row':np.arange(len(raw))})
        annotations=raw[[c for c in ['layer_guess','layer_guess_reordered','discard','in_tissue']
                         if c in raw]].copy()
        exclusions={'source_discard_flags_retained':int(raw.discard.sum()),
                    'explanation':'Preserve documented full fetched in-tissue cohort including discard flags'}
    else:
        import anndata as ad
        path=root/f'data/processed/{dataset}/{dataset}.h5ad'
        a=ad.read_h5ad(path,backed='r')
        raw=a.obs.copy();coords=np.asarray(a.obsm['spatial'])[:,:2]
        names=a.obs_names.astype(str).to_numpy();a.file.close()
        if dataset=='merfish_trem2_5xfad':
            obs=pd.DataFrame({'obs_id':dataset+'::'+names,'bio_id':raw.gen.astype(str).to_numpy(),
                              'section_id':raw.gen_fine.astype(str).to_numpy(),
                              'split_group':raw.gen_coarse.astype(str).to_numpy(),
                              'x':coords[:,0],'y':coords[:,1],'source_row':np.arange(len(raw))})
            annotations=raw[[c for c in ['gen_coarse','cluster','cluster_coarse','region_labels',
                              'region_labels_coarse','plaque_distance','leiden'] if c in raw]].reset_index(drop=True)
            exclusions={'source_QC':'existing raw-RNA count-native processed cohort; no additional QC'}
        else:
            unit=raw['24_01_17_HS'].astype(str).reset_index(drop=True)
            patient=unit.str.extract(r'^(HS\d+)_',expand=False)
            condition=raw['24_01_17_Condition'].astype(str).reset_index(drop=True)
            excluded=unit=='unassigned'
            if patient[~excluded].isna().any(): raise ValueError('Unparseable assigned Xenium patient')
            group=condition.map(lambda x:'healthy' if x=='HC' else
                                'nonresponder' if x.endswith('_NR') else
                                'responder' if x.endswith('_R') else '')
            if (group[~excluded]=='').any(): raise ValueError('Unknown Xenium patient condition')
            obs=pd.DataFrame({'obs_id':dataset+'::'+names,'bio_id':patient,
                              'section_id':unit,'source_graph_id':raw.Patient_ID_cores_combined.astype(str).to_numpy(),
                              'split_group':group,'x':coords[:,0],'y':coords[:,1],
                              'source_row':np.arange(len(raw))})
            annotations=raw[[c for c in ['24_01_08_EM_combined_from_leiden','24_01_17_Condition',
                       '24_05_29_Fine_annotations_Xenium_combined','25_06_11_Common_Coarse_Xenium_Combinedv3',
                       '25_06_11_Compartments'] if c in raw]].reset_index(drop=True)
            exclusions={'unassigned_observations':int(excluded.sum()),
                        'source_QC':'released annotated raw_counts cohort; unassigned pseudo-unit excluded'}
            obs=obs.loc[~excluded].reset_index(drop=True)
            annotations=annotations.loc[~excluded].reset_index(drop=True)
    original_graph=obs.source_graph_id if 'source_graph_id' in obs else obs.section_id
    obs['graph_id']=[json.dumps([dataset,str(b),str(s),str(g)],separators=(',',':'))
                     for b,s,g in zip(obs.bio_id,obs.section_id,original_graph)]
    annotations.insert(0,'obs_id',obs.obs_id.to_numpy())
    if obs.obs_id.duplicated().any() or np.any(~np.isfinite(obs[['x','y']].to_numpy(float))):
        raise ValueError('Nonunique observation IDs or invalid coordinates')
    return obs,annotations,path,exclusions


def inventory_cohort(root,dataset):
    root=Path(root);obs,annotations,path,exclusions=_source_metadata(root,dataset)
    columns=['bio_id','section_id','graph_id']+[c for c in ['position','replicate','split_group'] if c in obs]
    units=obs.groupby(columns,dropna=False,sort=True).size().rename('n_observations').reset_index()
    provenance=json.loads((root/f'data/processed/{dataset}/provenance.json').read_text())
    return {'dataset':dataset,'n_observations':len(obs),'n_biological_units':obs.bio_id.nunique(),
            'n_sections':obs.section_id.nunique(),'n_graph_strata':obs.graph_id.nunique(),
            'metadata_source':str(path),'legacy_provenance':provenance,
            'observation_order_sha256':array_hash(obs.obs_id),
            'annotation_fields':list(annotations.columns[1:]),'exclusions':exclusions,
            'units':units.to_dict(orient='records')}


def materialize_cohort(root,dataset):
    """One full raw-count CSR H5AD, metadata, and full source/mask provenance."""
    import anndata as ad
    root=Path(root);out=root/f'data/processed/joint_v2/{dataset}';out.mkdir(parents=True,exist_ok=True)
    with _lock(out/'.materialize.lock'):
        complete=out/'contract.json'
        if complete.exists():
            contract=json.loads(complete.read_text())
            if contract['contract_version']!=CONTRACT_VERSION: raise ValueError('Incompatible cohort contract')
            _verify_files(out,contract['artifacts'])
            return out/'counts.h5ad'
        obs,annotations,source,exclusions=_source_metadata(root,dataset)
        if dataset=='visium_dlpfc':
            matrix=root/'data/interim/visium_dlpfc/counts_genes_by_spots.mtx.gz'
            feature_file=root/'data/interim/visium_dlpfc/features.tsv.gz'
            counts=integer_csr(mmread(matrix).T)
            genes=pd.read_csv(feature_file,sep='\t').feature_id.astype(str).to_numpy()
            sources=[matrix,feature_file,source]
        else:
            original=ad.read_h5ad(source)
            counts=integer_csr(original.X[obs.source_row.to_numpy()])
            genes=original.var_names.astype(str).to_numpy();sources=[source]
        if counts.shape!=(len(obs),len(genes)) or len(set(genes))!=len(genes):
            raise ValueError('Count/observation/feature contract mismatch')
        fitobs=obs[list(SAFE_OBS)].set_index('obs_id',drop=False)
        object_=ad.AnnData(counts,obs=fitobs,var=pd.DataFrame(index=pd.Index(genes,name='feature_id')))
        object_.obsm['spatial']=obs[['x','y']].to_numpy(float)
        partial=out/f'counts.partial.{os.getpid()}.h5ad'
        object_.write_h5ad(partial,compression='gzip');os.replace(partial,out/'counts.h5ad')
        obs.to_parquet(out/'design_metadata.parquet',index=False)
        annotations.to_parquet(out/'evaluation_annotations.parquet',index=False)
        np.save(out/'source_row_mask.npy',obs.source_row.to_numpy())
        contract={'contract_version':CONTRACT_VERSION,'dataset':dataset,'shape':list(counts.shape),
                  'nnz':int(counts.nnz),'molecule_count':int(counts.sum()),
                  'sources':[{'path':str(s),'sha256':sha256(s)} for s in sources],
                  'output_sha256':sha256(out/'counts.h5ad'),
                  'observation_order_sha256':array_hash(obs.obs_id),'feature_order_sha256':array_hash(genes),
                  'excluded_or_retained_source_QC':exclusions,
                  'estimator_obs_fields':list(SAFE_OBS),'annotation_fields':list(annotations.columns[1:]),
                  'artifacts':_file_records(out,['counts.h5ad','design_metadata.parquet',
                                               'evaluation_annotations.parquet','source_row_mask.npy'])}
        _atomic_json(complete,contract)
    return out/'counts.h5ad'


def freeze_splits(root,dataset,outer_split_seed=26091001):
    root=Path(root);out=root/f'data/manifests/joint_v2/splits/{dataset}';out.mkdir(parents=True,exist_ok=True)
    with _lock(out/'.freeze.lock'):
        obs,_,_,_= _source_metadata(root,dataset)
        splits=build_outer_splits(obs,dataset,outer_split_seed)
        path=out/'splits.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old!=splits: raise ValueError('Refusing to overwrite a different frozen split manifest')
            _verify_files(out,json.loads((out/'artifacts.json').read_text()))
            return old
        for split in splits:
            roles=assign_roles(obs,split)
            columns=['obs_id','bio_id','section_id','graph_id','x','y']
            table=obs[columns].copy();table['role']=roles
            table.to_parquet(out/f"{split['split_id']}.parquet",index=False)
        _atomic_json(out/'inventory.json',inventory_cohort(root,dataset))
        _atomic_json(out/'artifacts.json',_file_records(out,
                     [f"{s['split_id']}.parquet" for s in splits]+['inventory.json']))
        _atomic_json(path,splits)
        return splits


def nested_thin(counts,retention,seed):
    """Sequential binomial coupling at prespecified levels; exact nested counts."""
    if retention not in (1.,.75,.5,.25): raise ValueError('Retention must be 1,.75,.50,.25')
    a=integer_csr(counts);values=a.data.copy();previous=1.
    for level in [.75,.5,.25]:
        if retention>level:break
        rng=np.random.default_rng(seed_namespace(seed,'nested_thinning',level))
        values=rng.binomial(values,level/previous).astype(np.int32);previous=level
    result=csr_matrix((values,a.indices.copy(),a.indptr.copy()),shape=a.shape)
    result.eliminate_zeros();return result


def molecule_split(counts,seed,training_fraction=.8):
    a=integer_csr(counts);rng=np.random.default_rng(seed)
    values=rng.binomial(a.data,training_fraction).astype(np.int32)
    fit=csr_matrix((values,a.indices.copy(),a.indptr.copy()),shape=a.shape)
    score=csr_matrix((a.data-values,a.indices.copy(),a.indptr.copy()),shape=a.shape)
    fit.eliminate_zeros();score.eliminate_zeros();return fit,score


def training_only_feature_ranking(counts,detection_fraction=.01):
    """Stable original-column ties; statistics use nonzero-total fit rows only.

    Detection minimum is max(1,floor(0.01*n_eligible_train)), retaining the
    historical floor convention while replacing its cohort-wide absolute cutoff.
    """
    a=integer_csr(counts);row_keep=np.asarray(a.sum(axis=1)).ravel()>0;a=a[row_keep]
    if not a.shape[0]:raise ValueError('No positive-count fitting observations')
    threshold=max(1,int(detection_fraction*a.shape[0]))
    detected=np.bincount(a.indices,minlength=a.shape[1])
    first=np.asarray(a.sum(axis=0,dtype=np.float64)).ravel()/a.shape[0]
    second=np.bincount(a.indices,weights=np.square(a.data.astype(np.float64)),minlength=a.shape[1])/a.shape[0]
    ratio=np.divide(np.maximum(second-first**2,0),first,out=np.zeros_like(first),where=first>0)
    eligible=np.flatnonzero((detected>=threshold)&(first>0))
    ranked=eligible[np.argsort(-ratio[eligible],kind='stable')]
    return {'ranked_indices':ranked,'detected':detected,'mean_raw_count':first,'variance_to_mean':ratio,
            'detection_threshold':threshold,'n_eligible_training_observations':int(a.shape[0]),
            'zero_training_row_mask':~row_keep}


def frequency_statistics(counts,biological_ids=None,alpha=.005):
    a=integer_csr(counts);totals=np.asarray(a.sum(axis=1)).ravel();keep=totals>0;a=a[keep];totals=totals[keep]
    if not len(totals):raise ValueError('No eligible training rows')
    mean_freq=np.asarray(a.multiply((1./totals)[:,None]).sum(axis=0)).ravel()/len(totals)
    pooled=np.asarray(a.sum(axis=0,dtype=np.float64)).ravel()/totals.sum()
    detected=np.bincount(a.indices,minlength=a.shape[1]);threshold=alpha*np.sqrt(np.log(max(a.shape))/(len(totals)*totals.mean()))
    positive=mean_freq>0;surviving=mean_freq>threshold
    result={'mean_row_normalized_frequency':mean_freq,'pooled_molecule_frequency':pooled,
            'prevalence':detected/len(totals),'detected':detected,'zero_frequency_count':int((~positive).sum()),
            'tran_threshold':float(threshold),'tran_strict_survival_count':int(surviving.sum()),
            'tran_top_10pct_fallback_triggered':bool(surviving.sum()<.1*a.shape[1]),
            'tran_effective_survival_count':int(max(surviving.sum(),np.ceil(.1*a.shape[1]))),
            'n_observations':len(totals),'n_molecules':int(totals.sum())}
    if biological_ids is not None:
        ids=np.asarray(biological_ids)[keep]
        result['per_biological_unit']={str(g):frequency_statistics(a[ids==g],alpha=alpha)
                                       for g in np.unique(ids)}
    return result


@dataclass
class TransferSet:
    adapt: csr_matrix
    score: csr_matrix
    obs: pd.DataFrame
    source_rows: np.ndarray
    zero_adaptation_mask: np.ndarray
    zero_score_mask: np.ndarray

    @property
    def observation_ids(self):return self.obs.obs_id.to_numpy()
    @property
    def graph_ids(self):return self.obs.graph_id.to_numpy()
    @property
    def biological_ids(self):return self.obs.bio_id.to_numpy()
    @property
    def coordinates(self):return self.obs[['x','y']].to_numpy(float)


@dataclass
class PreparedSplit:
    train_fit: csr_matrix
    train_score: csr_matrix
    train_rank_counts: csr_matrix
    full_feature_ids: np.ndarray
    feature_indices: np.ndarray
    feature_ids: np.ndarray
    reference_indices: np.ndarray
    reference_feature_ids: np.ndarray
    reference_train_fit: csr_matrix
    reference_train_score: csr_matrix
    eval_sets: dict[str,TransferSet]
    reference_eval_sets: dict[str,TransferSet]
    train_obs: pd.DataFrame
    source_train_rows: np.ndarray
    native_train_mask: np.ndarray
    ranking: dict
    split_metadata: dict

    @property
    def train_ids(self):return self.train_obs.obs_id.to_numpy()
    @property
    def train_coords(self):return self.train_obs[['x','y']].to_numpy(float)
    @property
    def train_graph_ids(self):return self.train_obs.graph_id.to_numpy()
    @property
    def train_bio_ids(self):return self.train_obs.bio_id.to_numpy()


def _counts_cache(root,dataset,split,molecule_seed,retention):
    """One immutable full-vocabulary split, reused across K/panels/estimators."""
    import anndata as ad
    cohort=root/f'data/processed/joint_v2/{dataset}'
    contract=json.loads((cohort/'contract.json').read_text())
    # Counts are immutable H5AD source artifacts; mask/design hashes enter every
    # split cache key. Evaluation annotations never participate in fitting keys.
    design_artifacts={name:record for name,record in contract['artifacts'].items()
                      if name in ['design_metadata.parquet','source_row_mask.npy']}
    _verify_files(cohort,design_artifacts)
    config={'contract_version':CONTRACT_VERSION,'cohort_sha256':contract['output_sha256'],
            'design_and_mask_artifacts':design_artifacts,
            'data_code_sha256':sha256(Path(__file__)),
            'splits_code_sha256':sha256(Path(__file__).with_name('splits.py')),
            'split_sha256':split['split_sha256'],'molecule_seed':int(molecule_seed),'retention':float(retention)}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=root/f'data/interim/joint_v2/count_splits/{dataset}/{key}'
    with _lock(out/'.cache.lock'):
        complete=out/'complete.json'
        if complete.exists():
            stored=json.loads(complete.read_text())
            if stored['config']!=config: raise ValueError('Incompatible count split cache')
            _verify_files(out,stored['artifacts'])
            return out,stored
        _verify_files(cohort,{'counts.h5ad':contract['artifacts']['counts.h5ad']})
        obs=pd.read_parquet(cohort/'design_metadata.parquet');roles=assign_roles(obs,split)
        original=ad.read_h5ad(cohort/'counts.h5ad')
        conservation=[]
        for role in sorted(set(roles)):
            rows=np.flatnonzero(roles==role)
            retained=nested_thin(original.X[rows],retention,
                                  seed_namespace(molecule_seed,dataset,split['split_id'],role,'retention'))
            fit,score=molecule_split(retained,seed_namespace(molecule_seed,dataset,split['split_id'],role,'split'))
            conservation.append({'role':role,'observations':len(rows),
                                 'retained_molecules':int(retained.sum()),
                                 'fit_or_adaptation_molecules':int(fit.sum()),'score_molecules':int(score.sum()),
                                 'zero_fit_or_adaptation_rows':int((np.asarray(fit.sum(axis=1)).ravel()==0).sum()),
                                 'zero_score_rows':int((np.asarray(score.sum(axis=1)).ravel()==0).sum()),
                                 'retention_stream_seed':seed_namespace(molecule_seed,dataset,split['split_id'],role,'retention'),
                                 'split_stream_seed':seed_namespace(molecule_seed,dataset,split['split_id'],role,'split')})
            for name,matrix in [('fit',fit),('score',score)]:
                partial=out/f'{role}_{name}.partial.{os.getpid()}.npz'
                save_npz(partial,matrix,compressed=True);os.replace(partial,out/f'{role}_{name}.npz')
            np.save(out/f'{role}_rows.npy',rows)
        names=[f'{role}_{kind}.{suffix}' for role in sorted(set(roles))
               for kind,suffix in [('fit','npz'),('score','npz'),('rows','npy')]]
        meta={'config':config,'roles':sorted(set(roles)),'feature_order_sha256':contract['feature_order_sha256'],
              'artifacts':_file_records(out,names),
              'count_conservation':conservation,
              'cache_key':key,'count_split':'nested retained counts -> independent 80/20 within each frozen role'}
        _atomic_json(complete,meta)
    return out,meta


def prepare_split(root,dataset,split_id,panel_requested=None,retention=1.,molecule_seed=26091002,
                  *,expected_split_hash=None):
    import anndata as ad
    root=Path(root);cohort=root/f'data/processed/joint_v2/{dataset}'
    splits=json.loads((root/f'data/manifests/joint_v2/splits/{dataset}/splits.json').read_text())
    split=next((s for s in splits if s['split_id']==split_id),None)
    if split is None:raise KeyError(split_id)
    frozen_payload={key:value for key,value in split.items() if key!='split_sha256'}
    observed_split_sha=hashlib.sha256(json.dumps(frozen_payload,sort_keys=True).encode()).hexdigest()
    if observed_split_sha!=split.get('split_sha256'):
        raise ValueError('Frozen outer split content does not match its saved checksum')
    if expected_split_hash is not None and fingerprint(split)!=expected_split_hash:
        raise ValueError('Frozen outer split differs from the scientific task manifest')
    out,meta=_counts_cache(root,dataset,split,molecule_seed,retention)
    obs=pd.read_parquet(cohort/'design_metadata.parquet')
    original=ad.read_h5ad(cohort/'counts.h5ad',backed='r')
    genes=original.var_names.astype(str).to_numpy();original.file.close()
    if array_hash(genes)!=meta['feature_order_sha256']:
        raise ValueError('Full feature order differs from the immutable count-cache vocabulary')
    full_fit=load_npz(out/'train_fit.npz');full_score=load_npz(out/'train_score.npz')
    rows=np.load(out/'train_rows.npy');ranking=training_only_feature_ranking(full_fit)
    if dataset=='visium_dlpfc':
        requested=2000 if panel_requested is None else int(panel_requested)
        if requested not in VISIUM_PANELS:raise ValueError('Unsupported Visium panel')
        indices=np.sort(ranking['ranked_indices'][:requested])
        reference=np.sort(ranking['ranked_indices'][:max(VISIUM_PANELS)])
    else:
        requested=len(genes) if panel_requested is None else int(panel_requested)
        if requested!=len(genes):raise ValueError('Targeted platforms retain only their native panel')
        indices=reference=np.arange(len(genes))
    if not len(indices):raise ValueError('No eligible genes')
    native_fit=full_fit[:,indices];keep=np.asarray(native_fit.sum(axis=1)).ravel()>0
    training_obs=obs.iloc[rows[keep]][list(SAFE_OBS)].reset_index(drop=True)
    native_eval={};reference_eval={}
    for role in meta['roles']:
        if role=='train':continue
        erows=np.load(out/f'{role}_rows.npy');efit=load_npz(out/f'{role}_fit.npz');escore=load_npz(out/f'{role}_score.npz')
        eobs=obs.iloc[erows][list(SAFE_OBS)].reset_index(drop=True)
        for vocab,target in [(indices,native_eval),(reference,reference_eval)]:
            adapt=efit[:,vocab];score=escore[:,vocab]
            target[role]=TransferSet(adapt,score,eobs,erows,
                                    np.asarray(adapt.sum(axis=1)).ravel()==0,
                                    np.asarray(score.sum(axis=1)).ravel()==0)
    split_meta={**meta,'outer_split':split,'panel_requested':requested,'panel_actual':len(indices),
                'reference_panel_actual':len(reference),'feature_order_sha256':array_hash(genes[indices]),
                'reference_feature_order_sha256':array_hash(genes[reference]),
                'n_native_zero_training_rows':int((~keep).sum()),
                'n_native_zero_training_score_rows':int((np.asarray(full_score[keep][:,indices].sum(axis=1)).ravel()==0).sum()),
                'detection_filter_rounding':'max(1,floor(0.01*n positive-full-vocabulary fitting-count observations))',
                'training_row_order_sha256':array_hash(training_obs.obs_id),
                'objective_weighting':'pooled count objective; no inverse-biological-size likelihood weights'}
    contributions=training_obs[['bio_id','section_id','graph_id']].copy()
    contributions['native_fit_molecules']=np.asarray(native_fit[keep].sum(axis=1)).ravel()
    contributions['reference_fit_molecules']=np.asarray(full_fit[keep][:,reference].sum(axis=1)).ravel()
    contributions['n_observations']=1
    split_meta['training_sample_contributions']=contributions.groupby(['bio_id','section_id','graph_id'],
        sort=True,observed=True)[['native_fit_molecules','reference_fit_molecules','n_observations']].sum().reset_index().to_dict('records')
    # Persist selection and count-defined masks independently of K and estimator.
    panel_path=out/f'panel_{requested}';panel_path.mkdir(exist_ok=True)
    with _lock(panel_path/'.panel.lock'):
        if (panel_path/'complete.json').exists():
            _verify_files(panel_path,json.loads((panel_path/'complete.json').read_text()))
            if json.loads((panel_path/'metadata.json').read_text())!=split_meta:
                raise ValueError('Frozen native panel metadata mismatch')
        else:
            partial=panel_path/f'masks_and_vocabulary.partial.{os.getpid()}.npz'
            eval_masks={}
            for role,evaluation in native_eval.items():
                eval_masks[f'{role}__native_zero_adaptation_mask']=evaluation.zero_adaptation_mask
                eval_masks[f'{role}__native_zero_score_mask']=evaluation.zero_score_mask
                eval_masks[f'{role}__reference_zero_adaptation_mask']=reference_eval[role].zero_adaptation_mask
                eval_masks[f'{role}__reference_zero_score_mask']=reference_eval[role].zero_score_mask
            np.savez_compressed(partial,native_train_mask=keep,
                                source_train_rows=rows,feature_indices=indices,reference_indices=reference,
                                ranked_indices=ranking['ranked_indices'],detected=ranking['detected'],
                                variance_to_mean=ranking['variance_to_mean'],**eval_masks)
            os.replace(partial,panel_path/'masks_and_vocabulary.npz')
            _atomic_json(panel_path/'metadata.json',split_meta)
            _atomic_json(panel_path/'complete.json',_file_records(panel_path,
                         ['masks_and_vocabulary.npz','metadata.json']))
    return PreparedSplit(native_fit[keep],full_score[keep][:,indices],full_fit[keep],genes,
                         indices,genes[indices],reference,genes[reference],full_fit[keep][:,reference],
                         full_score[keep][:,reference],native_eval,reference_eval,training_obs,
                         rows,keep,ranking,split_meta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','materialize','freeze'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--outer-split-seed',type=int,default=26091001)
    args=parser.parse_args()
    if args.action=='inventory':result=inventory_cohort(args.root,args.dataset)
    elif args.action=='freeze':result=freeze_splits(args.root,args.dataset,args.outer_split_seed)
    else:result=str(materialize_cohort(args.root,args.dataset))
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
