import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from gplsi_joint_v2.data import (frequency_statistics,integer_csr,molecule_split,
                                nested_thin,seed_namespace,training_only_feature_ranking)
from gplsi_joint_v2.graph import build_graph,neighbor_indices
from gplsi_joint_v2.splits import assign_roles,build_outer_splits


def visium_obs():
    records=[]
    for b,base in [('Br5292',151507),('Br5595',151669),('Br8100',151673)]:
        for section in range(4):
            for i in range(8):
                records.append(dict(obs_id=f'{base+section}::{i}',bio_id=b,section_id=str(base+section),
                                    graph_id=f'{b}::{base+section}',x=i//2,y=i%4,
                                    position=0 if section<2 else 300,replicate=section%2+1))
    return pd.DataFrame(records)


def biological_obs(dataset):
    records=[]
    if dataset=='merfish_trem2_5xfad':
        groups={'WT':['WT1','WT3','WT5'],'5xFAD':['5xFAD1','5xFAD3','5xFAD4','5xFAD5'],
                'Trem2':['Trem2_1','Trem2_3','Trem2_4','Trem2_5'],
                'Trem2_5xFAD':['Trem2_5xFAD1','Trem2_5xFAD3','Trem2_5xFAD4','Trem2_5xFAD5']}
    else:
        groups={'healthy':['HS31','HS33','HS35','HS37','HS39','HS40','HS41','HS42','HS43'],
                'responder':['HS32','HS34','HS36','HS38','HS44'],
                'nonresponder':['HS45','HS46','HS47','HS48','HS49','HS50']}
    for group,units in groups.items():
        for bio in units:
            sections=2 if ((dataset.startswith('merfish') and bio.endswith('1')) or
                           (dataset=='xenium_uc' and bio in ['HS32','HS34','HS36','HS48','HS50'])) else 1
            for section in range(sections):
                sid=f'{bio}::section{section}'
                for i in range(3):
                    records.append(dict(obs_id=f'{sid}::{i}',bio_id=bio,section_id=sid,
                                        graph_id=sid,x=i,y=i,split_group=group))
    return pd.DataFrame(records)


def test_visium_verified_position_rotation_and_complementary_halves():
    obs=visium_obs();splits=build_outer_splits(obs,'visium_dlpfc')
    assert len(splits)==7
    expected=[(['151507','151509'],['151508'],['151510']),
              (['151508','151510'],['151509'],['151507']),
              (['151508','151509'],['151510'],['151507'])]
    for split,(train,primary,additional) in zip(splits,expected):
        assignment=split['assignments'][0]
        assert (assignment['train_sections'],assignment['primary_test'],assignment['additional_test'])==(train,primary,additional)
        roles=assign_roles(obs,split)
        assert set(roles)=={'train','primary_test','additional_test'}
        assert obs.loc[roles=='train'].section_id.nunique()==6
    for a,b in [(3,4),(5,6)]:
        left=assign_roles(obs,splits[a]);right=assign_roles(obs,splits[b])
        np.testing.assert_array_equal(left=='train',right=='spatial_half')
        # Equal coordinates are never split between opposing halves.
        axis=splits[a]['axis']
        assert all(len(set(left[indices]))==1 for indices in obs.groupby(['section_id',axis]).indices.values())


@pytest.mark.parametrize('dataset,heldout,n',[('merfish_trem2_5xfad',2,15),('xenium_uc',3,20)])
def test_balanced_complete_biological_holdouts(dataset,heldout,n):
    obs=biological_obs(dataset);splits=build_outer_splits(obs,dataset)
    assert len(splits)==n
    assert set(Counter(b for s in splits for b in s['test_biological_ids']).values())=={heldout}
    selections={}
    for split in splits:
        roles=assign_roles(obs,split);held=set(split['test_biological_ids'])
        assert len(held)==heldout
        assert set(obs.loc[roles=='train'].bio_id).isdisjoint(held)
        for bio in held:
            assert not np.any(roles[obs.bio_id==bio]=='train')
        assert set(split['train_group_counts'])==set(obs.split_group)
        if dataset.startswith('merfish'):
            train=obs.loc[roles=='train']
            assert set(train.groupby('bio_id').section_id.nunique())=={1}
            for bio,section in split['selected_training_sections'].items():
                selections.setdefault(bio,[]).append(section)
    if selections:
        for bio,seq in selections.items():
            if obs.loc[obs.bio_id==bio].section_id.nunique()==2:
                assert all(a!=b for a,b in zip(seq,seq[1:]))


def test_split_and_graph_annotation_leakage_sentinels():
    obs=visium_obs();a=build_outer_splits(obs,'visium_dlpfc')
    obs['layer_guess']='fake';obs['score_count_canary']=np.arange(len(obs))*100000
    b=build_outer_splits(obs,'visium_dlpfc')
    assert a==b
    reordered=obs.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError,match='order mismatch'):assign_roles(reordered,a[0])


def test_count_conservation_nested_thinning_and_independent_namespaces():
    a=csr_matrix([[100,0,5],[30,22,9],[1,1,1]])
    previous=a
    for retain in [1.,.75,.5,.25]:
        thinned=nested_thin(a,retain,42)
        assert (thinned-previous).data.max(initial=0)<=0
        fit,score=molecule_split(thinned,78)
        assert (fit+score-thinned).nnz==0
        assert (thinned-nested_thin(a,retain,42)).nnz==0
        previous=thinned
    assert seed_namespace(42,'train')!=seed_namespace(42,'unseen_patient')
    assert seed_namespace(42,'train')==seed_namespace(42,'train')


def test_hvg_ranking_nested_sparse_dense_parity_and_ties():
    a=np.array([[4,4,0,0],[1,1,2,0],[0,0,4,0],[0,0,0,0]])
    dense=training_only_feature_ranking(a);sparse=training_only_feature_ranking(csr_matrix(a))
    for key in ['ranked_indices','detected','mean_raw_count','variance_to_mean']:
        np.testing.assert_array_equal(dense[key],sparse[key])
    order=dense['ranked_indices'].tolist()
    assert order.index(0)<order.index(1)
    assert set(order[:1])<=set(order[:2])<=set(order[:3])
    assert dense['detection_threshold']==1
    assert 3 not in order
    # A separate scoring matrix is never an argument to feature statistics.
    score=np.zeros_like(a);score[:,3]=100000
    np.testing.assert_array_equal(training_only_feature_ranking(a)['ranked_indices'],order)


def test_graph_no_cross_stratum_edges_self_exclusion_duplicate_coordinates():
    coords=np.array([[0,0],[0,0],[1,0],[2,0]]*2)
    ids=np.array(['A']*4+['B']*4)
    graph,meta=build_graph(coords,ids,k=3)
    r,c=graph.nonzero()
    assert np.all(ids[r]==ids[c]) and np.all(r!=c)
    assert (graph-graph.T).nnz==0
    nn,d=neighbor_indices(coords,ids,k=3)
    assert np.all(nn!=np.arange(8)[:,None])
    assert np.all(ids[nn]==ids[:,None])
    assert meta['n_edges_undirected']==12


def test_frequency_definitions_and_fallback_are_explicit():
    a=csr_matrix([[99,1,0],[0,1,0]])
    stats=frequency_statistics(a,['small','large'])
    np.testing.assert_allclose(stats['mean_row_normalized_frequency'],[.495,.505,0])
    np.testing.assert_allclose(stats['pooled_molecule_frequency'],[99/101,2/101,0])
    assert stats['zero_frequency_count']==1
    assert stats['n_observations']==2
    assert set(stats['per_biological_unit'])=={'small','large'}


def test_noninteger_counts_fail():
    with pytest.raises(ValueError):integer_csr([[1.,.5]])
    with pytest.raises(ValueError):integer_csr([[1.,-1.]])


def _tiny_cached_cohort(tmp_path):
    ad=pytest.importorskip('anndata');pytest.importorskip('pyarrow')
    import gplsi_joint_v2.data as module
    obs=visium_obs();obs['source_row']=np.arange(len(obs))
    root=tmp_path/'benchmark';cohort=root/'data/processed/joint_v2/visium_dlpfc';cohort.mkdir(parents=True)
    manifest=root/'data/manifests/joint_v2/splits/visium_dlpfc';manifest.mkdir(parents=True)
    splits=build_outer_splits(obs,'visium_dlpfc');(manifest/'splits.json').write_text(json.dumps(splits))
    counts=np.random.default_rng(7).poisson(3,size=(len(obs),625)).astype(np.int32)
    roles=assign_roles(obs,splits[0]);counts[roles=='train',624]=0
    counts[roles!='train',624]=100000 # Gene observed only in complete outer test sections.
    genes=np.array([f'g{i}' for i in range(counts.shape[1])])
    obj=ad.AnnData(csr_matrix(counts),obs=obs[list(module.SAFE_OBS)].set_index('obs_id',drop=False),
                   var=pd.DataFrame(index=genes))
    obj.write_h5ad(cohort/'counts.h5ad')
    obs.to_parquet(cohort/'design_metadata.parquet',index=False)
    pd.DataFrame({'obs_id':obs.obs_id,'layer_guess':'A'}).to_parquet(cohort/'evaluation_annotations.parquet',index=False)
    np.save(cohort/'source_row_mask.npy',obs.source_row)
    artifacts=module._file_records(cohort,['counts.h5ad','design_metadata.parquet',
                                           'evaluation_annotations.parquet','source_row_mask.npy'])
    contract={'contract_version':module.CONTRACT_VERSION,'output_sha256':module.sha256(cohort/'counts.h5ad'),
              'feature_order_sha256':module.array_hash(genes),'artifacts':artifacts}
    (cohort/'contract.json').write_text(json.dumps(contract))
    return root,splits[0]['split_id']


def test_prepare_split_end_to_end_outer_score_and_annotation_leakage(tmp_path,monkeypatch):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    initial=module.prepare_split(root,'visium_dlpfc',split_id,500)
    assert initial.train_fit.shape[1]==500
    assert 624 not in initial.feature_indices and 624 not in initial.reference_indices
    assert initial.reference_train_fit.shape[1]==624
    assert all('layer_guess' not in e.obs for e in initial.eval_sets.values())
    # Change the actual arrays presented by the loader only for outer D_score.
    # Artifact bytes stay intact: immutable input checks are independently tested.
    original_loader=module.load_npz
    def perturb_score(path):
        matrix=original_loader(path)
        return matrix*17 if str(path).endswith('_score.npz') and not str(path).endswith('train_score.npz') else matrix
    monkeypatch.setattr(module,'load_npz',perturb_score)
    annotations=root/'data/processed/joint_v2/visium_dlpfc/evaluation_annotations.parquet'
    frame=pd.read_parquet(annotations);frame['layer_guess']='REPLACED';frame.to_parquet(annotations,index=False)
    changed=module.prepare_split(root,'visium_dlpfc',split_id,500)
    assert (initial.train_fit-changed.train_fit).nnz==0
    assert (initial.train_score-changed.train_score).nnz==0
    assert (initial.train_rank_counts-changed.train_rank_counts).nnz==0
    np.testing.assert_array_equal(initial.feature_indices,changed.feature_indices)
    np.testing.assert_array_equal(initial.ranking['ranked_indices'],changed.ranking['ranked_indices'])
    for role in initial.eval_sets:
        assert (initial.eval_sets[role].adapt-changed.eval_sets[role].adapt).nnz==0
        assert (initial.eval_sets[role].score*17-changed.eval_sets[role].score).nnz==0
    larger=module.prepare_split(root,'visium_dlpfc',split_id,2000)
    assert set(initial.feature_indices)<=set(larger.feature_indices)
    np.testing.assert_array_equal(initial.reference_indices,larger.reference_indices)
    for role in initial.reference_eval_sets:
        a=initial.reference_eval_sets[role];b=larger.reference_eval_sets[role]
        np.testing.assert_array_equal(a.zero_adaptation_mask,b.zero_adaptation_mask)
        np.testing.assert_array_equal(a.zero_score_mask,b.zero_score_mask)


def test_count_cache_detects_corruption_and_rebuilds_interrupted_stage(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    initial=module.prepare_split(root,'visium_dlpfc',split_id,500)
    cache=root/'data/interim/joint_v2/count_splits/visium_dlpfc'/initial.split_metadata['cache_key']
    target=cache/'train_fit.npz';target.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum mismatch'):
        module.prepare_split(root,'visium_dlpfc',split_id,500)
    # Without a completion marker an interrupted stage is rebuilt, not a cache hit.
    (cache/'complete.json').unlink()
    restored=module.prepare_split(root,'visium_dlpfc',split_id,500)
    assert (initial.train_fit-restored.train_fit).nnz==0
    np.testing.assert_array_equal(initial.feature_indices,restored.feature_indices)


def test_runner_row_contract_injection_does_not_poison_shared_count_cache(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    initial=module.prepare_split(root,'visium_dlpfc',split_id,500)
    initial.split_metadata['row_contract']='/project/shared_prepare/train_observations.parquet'
    resumed=module.prepare_split(root,'visium_dlpfc',split_id,500)
    assert 'row_contract' not in resumed.split_metadata
    assert initial.split_metadata['training_row_order_sha256']==resumed.split_metadata['training_row_order_sha256']
    assert initial.split_metadata['cache_key']==resumed.split_metadata['cache_key']
    np.testing.assert_array_equal(initial.train_ids,resumed.train_ids)


def test_cached_counts_reject_changed_h5ad_gene_order_for_new_panel(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    module.prepare_split(root,'visium_dlpfc',split_id,500)
    import anndata as ad
    path=root/'data/processed/joint_v2/visium_dlpfc/counts.h5ad'
    object_=ad.read_h5ad(path);object_.var_names=object_.var_names[::-1];object_.write_h5ad(path)
    with pytest.raises(ValueError,match='feature order'):
        module.prepare_split(root,'visium_dlpfc',split_id,2000)


def test_cached_counts_reject_changed_frozen_split_even_without_prior_panel(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    module.prepare_split(root,'visium_dlpfc',split_id,500)
    path=root/'data/manifests/joint_v2/splits/visium_dlpfc/splits.json'
    splits=json.loads(path.read_text());splits[0]['outer_split_seed']+=1
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError,match='saved checksum'):
        module.prepare_split(root,'visium_dlpfc',split_id,2000)


def test_task_manifest_rejects_another_valid_frozen_split_version(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    path=root/'data/manifests/joint_v2/splits/visium_dlpfc/splits.json'
    expected=module.fingerprint(json.loads(path.read_text())[0])
    replacement=build_outer_splits(visium_obs(),'visium_dlpfc',outer_split_seed=26091002)
    path.write_text(json.dumps(replacement))
    with pytest.raises(ValueError,match='task manifest'):
        module.prepare_split(root,'visium_dlpfc',split_id,500,expected_split_hash=expected)


def test_detection_threshold_uses_positive_training_rows_and_preserves_floor():
    values=np.zeros((1000,4),dtype=np.int32)
    values[:299,0]=1
    values[0,1]=5
    values[:2,2]=10
    ranking=training_only_feature_ranking(csr_matrix(values))
    assert ranking['n_eligible_training_observations']==299
    assert ranking['detection_threshold']==2
    assert set(ranking['ranked_indices'])=={0,2}
    assert ranking['zero_training_row_mask'].sum()==701


def test_largest_requested_visium_panel_records_shortfall_without_padding(tmp_path):
    import gplsi_joint_v2.data as module
    root,split_id=_tiny_cached_cohort(tmp_path)
    prepared=module.prepare_split(root,'visium_dlpfc',split_id,15000)
    assert prepared.split_metadata['panel_requested']==15000
    assert prepared.split_metadata['panel_actual']==624
    assert prepared.train_fit.shape[1]==624
    assert prepared.split_metadata['reference_panel_actual']==624
    assert len(set(prepared.feature_ids))==624
    assert 'g624' not in prepared.feature_ids
