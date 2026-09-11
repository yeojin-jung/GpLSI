"""External annotations evaluated after fitting, with biological group blocking."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

from .metrics import grouped_label_transfer,hard_label_metrics

CELL_FIELDS={
    'visium_dlpfc':['layer_guess'],
    'merfish_trem2_5xfad':['cluster','cluster_coarse','region_labels','region_labels_coarse'],
    'xenium_uc':['24_01_08_EM_combined_from_leiden','24_05_29_Fine_annotations_Xenium_combined',
                 '25_06_11_Common_Coarse_Xenium_Combinedv3','25_06_11_Compartments']}
CONDITION='24_01_17_Condition'


def _valid_labels(values):
    return ~np.isin(np.asarray(values,dtype=str),['','nan','None','<NA>'])


def _hard_by_unit(W,labels,groups):
    groups=np.asarray(groups,dtype=str)
    records=[]
    for bio in np.unique(groups):
        mask=groups==bio
        records.append({'biological_id':bio,**hard_label_metrics(W[mask],np.asarray(labels)[mask])})
    valid=[r for r in records if r['status']=='ok']
    return {'pooled':hard_label_metrics(W,labels),'per_biological_unit':records,
            'equal_biological_NMI':float(np.mean([r['NMI'] for r in valid])) if valid else None,
            'equal_biological_ARI':float(np.mean([r['ARI'] for r in valid])) if valid else None,
            'n_valid_biological_units':len(valid),'n_biological_units':len(records)}


def _decoder(Wtrain,ytrain,gtrain,Wtest,ytest,gtest,seed):
    result=grouped_label_transfer(Wtrain,ytrain,gtrain,Wtest,ytest,seed=seed)
    if 'predictions' in result:
        predictions=np.asarray(result.pop('predictions'),dtype=str)
        ytest=np.asarray(ytest,dtype=str);gtest=np.asarray(gtest,dtype=str)
        valid=_valid_labels(ytest);records=[]
        for bio in np.unique(gtest):
            rows=(gtest==bio)&valid
            records.append({'biological_id':bio,'labeled_observations':int(rows.sum()),
                            'test_classes':np.unique(ytest[rows]).tolist(),
                            'balanced_accuracy':float(balanced_accuracy_score(ytest[rows],predictions[rows]))
                            if rows.any() else None})
        defined=[r['balanced_accuracy'] for r in records if r['balanced_accuracy'] is not None]
        result['per_biological_unit']=records
        result['equal_biological_balanced_accuracy']=float(np.mean(defined)) if defined else None
    return result


def _summaries(W,obs,labels,*,section_column='section_id',keep_section=False):
    """Equal section/core means first; optional equal section mean per animal.

    Labels must be constant within each section and then each collapsed biological
    unit. Inconsistent labels fail explicitly rather than majority voting.
    """
    columns=[f'topic_{k}' for k in range(W.shape[1])]
    frame=obs[['bio_id',section_column]].reset_index(drop=True).copy()
    frame['label']=np.asarray(labels,dtype=str)
    for k,column in enumerate(columns):frame[column]=W[:,k]
    key=['bio_id',section_column]
    if np.any(frame.groupby(key,observed=True).label.nunique()>1):
        raise ValueError('Outcome labels are not constant within an evaluation section/unit')
    unit=frame.groupby(key,sort=True,observed=True)[columns].mean().reset_index()
    targets=frame.groupby(key,sort=True,observed=True).label.first().reset_index()
    unit=unit.merge(targets,on=key,validate='one_to_one')
    if keep_section:return unit,columns
    if np.any(unit.groupby('bio_id',observed=True).label.nunique()>1):
        raise ValueError('Outcome is not constant within biological unit')
    means=unit.groupby('bio_id',sort=True,observed=True)[columns].mean().reset_index()
    means['label']=means.bio_id.map(unit.groupby('bio_id',observed=True).label.first())
    return means,columns


def _outcome_transfer(train,test,columns,seed,*,endpoint):
    tr=train.loc[_valid_labels(train.label)].copy();te=test.loc[_valid_labels(test.label)].copy()
    if len(te)==0 or len(tr)==0:
        return {'endpoint':endpoint,'status':'no_eligible_biological_outcome_rows',
                'n_train_units':len(tr),'n_test_units':len(te)}
    result=_decoder(tr[columns].to_numpy(),tr.label.to_numpy(),tr.bio_id.to_numpy(),
                    te[columns].to_numpy(),te.label.to_numpy(),te.bio_id.to_numpy(),seed)
    result.update({'endpoint':endpoint,'n_train_units':len(tr),'n_test_units':len(te),
                   'training_unit_summaries':tr.to_dict('records'),'test_unit_summaries':te.to_dict('records'),
                   'unit_weighting':'equal section/unit means, then equal sections/timepoints within biological unit',
                   'small_sample_limitation':'limited biological units and test classes; repeated outer splits are dependent'})
    return result


def _plaque_rank_metrics(W,obs,values):
    numeric=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(float)
    records=[]
    for section in np.unique(obs.section_id):
        mask=(obs.section_id.to_numpy()==section)&np.isfinite(numeric)
        for topic in range(W.shape[1]):
            valid=mask&np.isfinite(W[:,topic]);x=W[valid,topic];y=numeric[valid]
            status='ok' if valid.sum()>=3 and np.ptp(x)>1e-14 and np.ptp(y)>0 else 'insufficient_or_constant_values'
            records.append({'section_id':str(section),'biological_id':str(obs.loc[mask,'bio_id'].iloc[0]) if mask.any() else None,
                            'topic':topic,'n_observations':int(valid.sum()),'status':status,
                            'spearman_rho':float(spearmanr(x,y).statistic) if status=='ok' else None})
    return {'endpoint':'within_section_plaque_distance_rank_association','per_section_topic':records,
            'distance_units':'source units not independently verified; rank association is unit-invariant',
            'p_values':None,'inference':'descriptive association; no cell-independent significance claim'}


def _patient_group(condition):
    value=str(condition)
    if value=='HC':return 'healthy'
    if value.endswith('_NR'):return 'nonresponder'
    if value.endswith('_R'):return 'responder'
    return ''


def paired_prepost_changes(W,obs,conditions):
    """One paired delta per patient, averaging cores within each time point."""
    frame=obs[['bio_id','section_id','graph_id']].reset_index(drop=True).copy()
    frame['condition']=np.asarray(conditions,dtype=str)
    frame['timepoint']=frame.condition.map(lambda x:'pre' if x.startswith('PRE_') else 'post' if x.startswith('POST_') else '')
    columns=[f'topic_{k}' for k in range(W.shape[1])]
    for k,column in enumerate(columns):frame[column]=W[:,k]
    frame=frame.loc[frame.timepoint!='']
    core=frame.groupby(['bio_id','timepoint','graph_id'],observed=True,sort=True)[columns].mean().reset_index()
    means=core.groupby(['bio_id','timepoint'],observed=True,sort=True)[columns].mean().reset_index()
    records=[];unpaired=[]
    for patient,rows in means.groupby('bio_id',sort=True):
        if set(rows.timepoint)!={'pre','post'}:
            unpaired.append(str(patient));continue
        pre=rows.loc[rows.timepoint=='pre',columns].to_numpy()[0]
        post=rows.loc[rows.timepoint=='post',columns].to_numpy()[0]
        records.append({'patient_id':str(patient),'pre':pre.tolist(),'post':post.tolist(),
                        'post_minus_pre':(post-pre).tolist()})
    return {'endpoint':'patient_paired_pre_post_topic_composition','paired_patients':records,
            'unpaired_patients':unpaired,'unit_weighting':'equal core mean within each patient/timepoint',
            'statistical_unit':'patient','standard_error':None}


def evaluate_metadata(root,dataset,prepared,W_train,foldin_results,*,reference=False,seed=26091003):
    """Read annotations only here; score complete outer folds with frozen A.

    foldin_results maps evaluation role to FoldInResult or a dict containing W
    and inference_valid. This function cannot mutate W, A, counts, or tuning.
    """
    root=Path(root);annotation_path=root/f'data/processed/joint_v2/{dataset}/evaluation_annotations.parquet'
    annotations=pd.read_parquet(annotation_path)
    if annotations.obs_id.duplicated().any():raise ValueError('Duplicate annotation observation IDs')
    annotations=annotations.set_index('obs_id');trainobs=prepared.train_obs.reset_index(drop=True)
    if not set(trainobs.obs_id)<=set(annotations.index):raise ValueError('Training annotation IDs missing')
    train=annotations.loc[trainobs.obs_id].reset_index(drop=True);wtrain=np.asarray(W_train,dtype=float)
    if len(wtrain)!=len(trainobs):raise ValueError('Training W/annotation rows misaligned')
    evaluation=prepared.reference_eval_sets if reference else prepared.eval_sets
    result={'dataset':dataset,'vocabulary':'common_reference' if reference else 'native',
            'annotation_fields_available':list(annotations.columns),'train':{},'outer':{},
            'labels_used_during_fitting':False,'labels_used_during_topic_alignment':False,
            'decoder_tuning':'training biological groups only; exact candidates and fold curves recorded',
            'interpretation':'training hard-label agreement is descriptive; transfer decoders scored only on outer data'}
    if dataset=='visium_dlpfc' and 'layer_guess_reordered' in annotations:
        a=annotations.layer_guess.fillna('').astype(str);b=annotations.layer_guess_reordered.fillna('').astype(str)
        result['duplicated_visium_annotations']={'equal':bool(a.equals(b)),
                                               'primary_field':'layer_guess','secondary_field_not_independent':'layer_guess_reordered'}
    fields=CELL_FIELDS[dataset]
    for field in fields:
        result['train'][field]=_hard_by_unit(wtrain,train[field].to_numpy(),trainobs.bio_id.to_numpy()) if field in train else {'status':'source_field_absent'}
    for role,transfer in evaluation.items():
        if role not in foldin_results:
            result['outer'][role]={'status':'foldin_result_missing'};continue
        fold=foldin_results[role]
        w=np.asarray(fold['W'] if isinstance(fold,dict) else fold.W,dtype=float)
        valid=np.asarray(fold['inference_valid'] if isinstance(fold,dict) else fold.inference_valid,dtype=bool)
        if w.shape!=(len(transfer.obs),wtrain.shape[1]) or valid.shape!=(len(transfer.obs),):
            raise ValueError('Test fold-in/annotation rows misaligned')
        valid=valid&~transfer.zero_adaptation_mask
        testobs=transfer.obs.loc[valid].reset_index(drop=True);w=w[valid]
        if not set(testobs.obs_id)<=set(annotations.index):raise ValueError('Test annotation IDs missing')
        test=annotations.loc[testobs.obs_id].reset_index(drop=True)
        endpoint={'status':'ok' if len(testobs) else 'no_valid_adaptation_inference',
                  'count_eligible_observations':int((~transfer.zero_adaptation_mask).sum()),
                  'available_inference_observations':len(testobs),
                  'inference_unavailable_observations':int((~transfer.zero_adaptation_mask&~valid).sum()),
                  'fields':{},'biological_outcomes':{}}
        result['outer'][role]=endpoint
        if not len(testobs):continue
        for field in fields:
            if field not in test:
                endpoint['fields'][field]={'status':'source_field_absent'};continue
            y=train[field].to_numpy();yt=test[field].to_numpy()
            endpoint['fields'][field]={'hard_agreement':_hard_by_unit(w,yt,testobs.bio_id.to_numpy()),
                'grouped_decoder':_decoder(wtrain,y,trainobs.bio_id.to_numpy(),w,yt,testobs.bio_id.to_numpy(),seed)}
        if dataset=='merfish_trem2_5xfad':
            if 'plaque_distance' in test:
                endpoint['plaque_distance']=_plaque_rank_metrics(w,testobs,test.plaque_distance)
            if 'gen_coarse' in test:
                train_summary,columns=_summaries(wtrain,trainobs,train.gen_coarse)
                test_summary,_=_summaries(w,testobs,test.gen_coarse)
                endpoint['biological_outcomes']['genotype']=_outcome_transfer(train_summary,test_summary,columns,seed,
                                                                           endpoint='animal_genotype')
                endpoint['biological_outcomes']['genotype']['evaluation_relationship']=role
        if dataset=='xenium_uc' and CONDITION in test:
            for name,transform,keep_section in [('condition',lambda x:x,True),
                                               ('patient_group',_patient_group,False),
                                               ('response',lambda x:_patient_group(x) if _patient_group(x)!='healthy' else '',False)]:
                trlabels=train[CONDITION].map(transform);telabels=test[CONDITION].map(transform)
                tr,columns=_summaries(wtrain,trainobs,trlabels,keep_section=keep_section)
                te,_=_summaries(w,testobs,telabels,keep_section=keep_section)
                endpoint['biological_outcomes'][name]=_outcome_transfer(tr,te,columns,seed,
                    endpoint='patient_timepoint_condition' if keep_section else f'patient_{name}')
            endpoint['paired_prepost']=paired_prepost_changes(w,testobs,test[CONDITION])
            endpoint['clinical_metadata_availability']={'condition':CONDITION,'response':'derived from source _R/_NR condition suffix',
                                                      'unrecorded_outcomes':'not invented or inferred'}
    return result
