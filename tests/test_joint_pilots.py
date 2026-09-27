from collections import Counter
import json
from pathlib import Path

import pytest

from gplsi_joint_v2.artifacts import atomic_json,commit_stage,record_failed_attempt
from gplsi_joint_v2.config import default_config,fingerprint
from gplsi_joint_v2.manifests import build_dag
from gplsi_joint_v2.pilots import (PILOT_TARGETS,_correctness_evidence,ancestral_closure,assess_pilots,
                                  select_pilot_dag,write_pilot_manifest)


def _splits():
    return [{'dataset':d,'split_id':t['split_id'],
             'protocol':'section_holdout' if d.startswith('visium') else
             'animal_holdout' if d.startswith('merfish') else 'patient_holdout'}
            for d,t in PILOT_TARGETS.items()]


def test_pilot_exact_scientific_grid_and_parent_closure():
    config=default_config();source={'source_tree_hash':'source'};contracts={d:{} for d in PILOT_TARGETS}
    manifests,dag=select_pilot_dag(_splits(),config,source,contracts)
    assert len(manifests['pilots'])==6
    assert len(dag['tasks'])==527
    assert Counter(t['stage'] for t in dag['tasks'])=={
        'prepare':3,'spectral':18,'geometry':90,'recovery':138,
        'reference_recovery':46,'competitor':48,'evaluation':184}
    assert Counter(t['dataset'] for t in dag['tasks'])=={
        'visium_dlpfc':237,'merfish_trem2_5xfad':145,'xenium_uc':145}
    ids={t['task_id'] for t in dag['tasks']}
    assert all(set(t['parents'])<=ids for t in dag['tasks'])
    assert not any(t['stage']=='diagnostics' for t in dag['tasks'])
    assert dag['pilot_design']['hidden_subsampling'] is False
    geometry=[t for t in dag['tasks'] if t['stage']=='geometry' and t['spec']['K']==20]
    assert len(geometry)==12
    assert {t['spec']['hunter'] for t in geometry}=={'spa_current'}
    assert {t['spec']['family'] for t in geometry}=={'document'}
    assert {t['spec']['preprocessing'] for t in geometry}=={'P0_raw','P3_tran_then_ke'}
    assert {t['spec']['control'] for t in geometry}=={'lambda_zero','selected'}
    assert {t['spec']['recovery'] for t in dag['tasks']
            if t['stage'] in ['recovery','reference_recovery','evaluation']}=={'A_full_Pois'}
    mapping={t['task_id']:t for t in dag['tasks']}
    assert all(any(mapping[p]['stage']=='recovery' for p in t['parents'])
               for t in dag['tasks'] if t['stage']=='evaluation' and t['spec']['vocabulary']=='native')
    # Pilot-tier membership changes no scientific content-addressed identity.
    production=build_dag({'core':manifests['pilots']},config,'source',contracts)
    assert ids<={t['task_id'] for t in production['tasks']}


def test_pilot_manifest_is_immutable_and_includes_source_snapshot(tmp_path):
    for split in _splits():
        directory=tmp_path/f"data/manifests/joint_v2/splits/{split['dataset']}";directory.mkdir(parents=True)
        atomic_json(directory/'splits.json',[split])
    config=default_config();identity={'source_tree_hash':'source','commit':'abc'}
    contracts={d:{} for d in PILOT_TARGETS}
    first=write_pilot_manifest(tmp_path,config,identity,contracts)
    second=write_pilot_manifest(tmp_path,config,identity,contracts)
    assert first==second
    assert Path(first['manifest']).exists()
    assert Path(first['source_identity']).exists()
    changed=write_pilot_manifest(tmp_path,config,{**identity,'source_tree_hash':'next'},contracts)
    assert changed['manifest']!=first['manifest']


def test_ancestral_closure_rejects_missing_parent():
    with pytest.raises(ValueError,match='Missing DAG'):
        ancestral_closure([{'task_id':'child','parents':['missing']}],['child'])


def _assessment_fixture(tmp_path,*,smoke_source='code',xml_failure=False,
                        correctness_source='code',correctness_config=None):
    config=default_config();reports=tmp_path/'reports/joint_v2';reports.mkdir(parents=True)
    xml=('<testsuites><testsuite tests="1" errors="0" failures="1"><testcase name="x"><failure/></testcase></testsuite></testsuites>'
         if xml_failure else '<testsuites><testsuite tests="1" errors="0" failures="0"><testcase name="x"/></testsuite></testsuites>')
    (reports/'correctness_1.xml').write_text(xml)
    atomic_json(reports/'correctness_1_source.json',
                {'source':{'source_tree_hash':correctness_source},
                 'config_hash':fingerprint(config) if correctness_config is None else correctness_config,
                 'job_id':'1'})
    smoke=tmp_path/'results/joint_v2/smoke/test/smoke_report.json'
    atomic_json(smoke,{'source':{'source_tree_hash':smoke_source},'config':config,
                      'status':'passed','substantive_failures':[]})
    atomic_json(reports/'SMOKE_STATUS.json',{'status':'passed','report':str(smoke)})
    tasks=[{'task_id':'prepare','stage':'prepare','dataset':'visium_dlpfc','variant':'prepare',
            'parents':[],'code_hash':'code','spec':{}},
           {'task_id':'geometry','stage':'geometry','dataset':'visium_dlpfc','variant':'svs',
            'parents':['prepare'],'code_hash':'code','spec':{'K':20}},
           {'task_id':'recovery','stage':'recovery','dataset':'visium_dlpfc','variant':'poisson',
            'parents':['prepare','geometry'],'code_hash':'code','spec':{'K':20}}]
    dag={'tasks':tasks,'config':config,'config_hash':fingerprint(config),'code_hash':'code',
         'pilot_design':{'hidden_subsampling':False}}
    manifest=tmp_path/'data/manifests/joint_v2/pilot_test/dag.json';atomic_json(manifest,dag)
    directory=tmp_path/'data/interim/joint_v2/stages/prepare/prepare'
    effective={'task':tasks[0],'runtime_seconds':10.,'peak_rss_bytes':1024**3,
               'slurm_job_id':'123','partition':'caslake','prepared':{}}
    atomic_json(directory/'effective.json',effective)
    commit_stage(directory,'prepare',effective,['effective.json'])
    return manifest,tasks


def test_resource_blocked_propagation_never_approves_production(tmp_path):
    manifest,tasks=_assessment_fixture(tmp_path)
    directory=tmp_path/'results/joint_v2/stages/geometry/geometry'
    class GeometryResourceBlocked(RuntimeError):pass
    record_failed_attempt(directory,GeometryResourceBlocked('exact enumeration exceeds budget'),
                          {'metadata':{'status':'resource_blocked','reason':'frozen exact budget'}})
    result=assess_pilots(tmp_path,manifest)
    assert result['status']=='ready_for_resource_review'
    assert result['stage_state_counts']=={'complete':1,'resource_blocked':1,'blocked_by_resource_parent':1}
    assert result['production_approved'] is False
    assert not (tmp_path/'reports/joint_v2/PRODUCTION_GATE.json').exists()
    assert result['observed_storage']['stages']['allocated_bytes']>0
    assert result['proposed_resource_requests']['visium_dlpfc/prepare']['memory_gb']==4


@pytest.mark.parametrize('failure',['runtime','checksum','xml','smoke_source',
                                    'correctness_source','correctness_config'])
def test_gate_failures_are_not_scientific_resource_blocks(tmp_path,failure):
    manifest,tasks=_assessment_fixture(tmp_path,smoke_source='different' if failure=='smoke_source' else 'code',
                                       xml_failure=failure=='xml',
                                       correctness_source='different' if failure=='correctness_source' else 'code',
                                       correctness_config='different' if failure=='correctness_config' else None)
    for task in tasks[1:]:
        directory=tmp_path/'results/joint_v2/stages'/task['stage']/task['task_id']
        effective={'task':task,'runtime_seconds':5.,'peak_rss_bytes':1024**3,'prepared':{}}
        if failure=='runtime' and task['stage']=='geometry':
            record_failed_attempt(directory,MemoryError('OOM is a retryable resource failure, not a successful exact-budget exclusion'),{})
        else:
            atomic_json(directory/'effective.json',effective)
            commit_stage(directory,task['task_id'],effective,['effective.json'])
            if failure=='checksum' and task['stage']=='geometry':(directory/'effective.json').write_text('corrupt')
    result=assess_pilots(tmp_path,manifest)
    assert result['status']!='ready_for_resource_review'
    assert result['production_approved'] is False
    if failure.startswith('correctness_'):
        assert result['correctness']['status']=='incompatible_correctness_source_or_config'


def test_correctness_requires_matching_filename_sidecar_and_latest_source(tmp_path):
    _assessment_fixture(tmp_path)
    reports=tmp_path/'reports/joint_v2'
    expected_config=fingerprint(default_config())
    assert _correctness_evidence(tmp_path,'code',expected_config)['passed']
    # A new green XML cannot borrow the previous job's matching source identity.
    xml=reports/'correctness_2.xml'
    xml.write_text((reports/'correctness_1.xml').read_text())
    result=_correctness_evidence(tmp_path,'code',expected_config)
    assert result['path']==str(xml)
    assert result['status']=='correctness_source_sidecar_missing'
    assert not result['passed']
    atomic_json(reports/'correctness_2_source.json',
                {'source':{'source_tree_hash':'another_snapshot'},
                 'config_hash':expected_config,'job_id':'2'})
    result=_correctness_evidence(tmp_path,'code',expected_config)
    assert not result['passed']
    assert result['source_hash_matches'] is False
    assert result['config_hash_matches'] is True
    assert result['status']=='incompatible_correctness_source_or_config'


def test_correctness_rejects_invalid_source_sidecar(tmp_path):
    _assessment_fixture(tmp_path)
    (tmp_path/'reports/joint_v2/correctness_1_source.json').write_text('{invalid')
    result=_correctness_evidence(tmp_path,'code',fingerprint(default_config()))
    assert not result['passed']
    assert result['status']=='invalid_correctness_source_sidecar'
