from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gplsi_joint_v2.metadata_evaluation import (_patient_group,_summaries,_plaque_rank_metrics,
                                              evaluate_metadata,paired_prepost_changes)


def test_animal_summaries_average_sections_not_cells():
    W=np.array([[1.,0.]]*9+[[0.,1.]]+[[.3,.7]])
    obs=pd.DataFrame({'bio_id':['A']*10+['B'],'section_id':['A1']*9+['A2','B1']})
    summary,columns=_summaries(W,obs,['WT']*10+['mutant'])
    np.testing.assert_allclose(summary.loc[summary.bio_id=='A',columns].to_numpy(),[[.5,.5]])
    with pytest.raises(ValueError,match='not constant'):
        _summaries(W,obs,['WT']*9+['mutant','mutant'])


def test_paired_patient_delta_and_source_response_mapping():
    obs=pd.DataFrame({'bio_id':['HS32']*4+['HS45'],
                      'section_id':['HS32_PRE']*3+['HS32_POST','HS45_POST'],
                      'graph_id':['core1','core1','core2','core3','core4']})
    W=np.array([[1,0],[1,0],[0,1],[.25,.75],[.8,.2]],dtype=float)
    conditions=['PRE_VDZ_R']*3+['POST_VDZ_R','POST_VDZ_NR']
    result=paired_prepost_changes(W,obs,conditions)
    assert len(result['paired_patients'])==1
    paired=result['paired_patients'][0]
    np.testing.assert_allclose(paired['pre'],[.5,.5])
    np.testing.assert_allclose(paired['post_minus_pre'],[-.25,.25])
    assert result['unpaired_patients']==['HS45']
    assert [_patient_group(c) for c in ['HC','PRE_VDZ_R','POST_VDZ_NR','other']]==['healthy','responder','nonresponder','']


def test_plaque_ranks_center_evaluation_by_section_and_flag_constants():
    W=np.array([[.1,.9],[.2,.8],[.3,.7],[.5,.5],[.5,.5],[.5,.5]])
    obs=pd.DataFrame({'bio_id':['A']*6,'section_id':['first']*3+['second']*3})
    result=_plaque_rank_metrics(W,obs,[1,2,3,1,2,3])
    valid=[r for r in result['per_section_topic'] if r['status']=='ok']
    assert len(valid)==2
    np.testing.assert_allclose([r['spearman_rho'] for r in valid],[1,-1])
    assert result['p_values'] is None


def test_metadata_reads_ids_and_never_changes_factors(tmp_path,monkeypatch):
    pytest.importorskip('pyarrow')
    import gplsi_joint_v2.metadata_evaluation as module
    trainobs=pd.DataFrame({'obs_id':['a','b','c','d'],'bio_id':['donor1']*2+['donor2']*2,
                           'section_id':['s1']*2+['s2']*2,'graph_id':['s1']*2+['s2']*2})
    testobs=pd.DataFrame({'obs_id':['e','f'],'bio_id':['donor1']*2,'section_id':['held']*2,'graph_id':['held']*2})
    annotations=pd.DataFrame({'obs_id':['f','e','d','c','b','a'],
                              'layer_guess':['L2','L1','L2','L1','L2','L1'],
                              'layer_guess_reordered':['L2','L1','L2','L1','L2','L1']})
    directory=tmp_path/'data/processed/joint_v2/visium_dlpfc';directory.mkdir(parents=True)
    annotations.to_parquet(directory/'evaluation_annotations.parquet',index=False)
    trainW=np.array([[1.,0],[0,1],[1,0],[0,1]])
    testW=np.array([[1.,0],[0,1]])
    calls=[]
    def fake_decoder(Xtrain,ytrain,groupstrain,Xtest,ytest,**kwargs):
        calls.append((list(ytrain),list(groupstrain),list(ytest)))
        return {'status':'test_decoder','balanced_accuracy':1.,'test_labels_used_for_tuning':False,
                'predictions':np.array(['L1','L2'])}
    monkeypatch.setattr(module,'grouped_label_transfer',fake_decoder)
    transfer=SimpleNamespace(obs=testobs,zero_adaptation_mask=np.zeros(2,dtype=bool))
    prepared=SimpleNamespace(train_obs=trainobs,eval_sets={'primary_test':transfer},reference_eval_sets={})
    traincopy=trainW.copy();testcopy=testW.copy()
    result=evaluate_metadata(tmp_path,'visium_dlpfc',prepared,trainW,
                             {'primary_test':{'W':testW,'inference_valid':np.ones(2,dtype=bool)}})
    np.testing.assert_array_equal(trainW,traincopy);np.testing.assert_array_equal(testW,testcopy)
    assert result['duplicated_visium_annotations']['equal']
    assert set(result['train'])=={'layer_guess'}
    assert result['outer']['primary_test']['fields']['layer_guess']['hard_agreement']['pooled']['NMI']==1
    assert calls==[(['L1','L2','L1','L2'],['donor1','donor1','donor2','donor2'],['L1','L2'])]
