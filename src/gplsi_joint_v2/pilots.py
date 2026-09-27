"""Full-observation pilot DAG and evidence-based resource review gate.

Assessment never writes PRODUCTION_GATE.json or submits jobs. It produces the
concrete evidence and proposed requests the root controller must review first.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .artifacts import atomic_json,compatible_completed,sha256_file
from .config import fingerprint
from .manifests import build_base_manifests,build_dag,summarize_dag,write_manifests


PILOT_TARGETS={
    'visium_dlpfc':{'split_id':'section_rotation_01','panel_requested':15000,'primary_K':7},
    'merfish_trem2_5xfad':{'split_id':'leave_2_out_01','panel_requested':300,'primary_K':12},
    'xenium_uc':{'split_id':'leave_3_out_01','panel_requested':290,'primary_K':12}}


def ancestral_closure(tasks,selected_ids):
    mapping={t['task_id']:t for t in tasks};selected=set(selected_ids);pending=list(selected)
    while pending:
        key=pending.pop()
        if key not in mapping:raise ValueError(f'Missing DAG task/parent: {key}')
        for parent in mapping[key]['parents']:
            if parent not in selected:selected.add(parent);pending.append(parent)
    return [task for task in tasks if task['task_id'] in selected]


def select_pilot_dag(splits,config,source_identity,contracts):
    """Select six bases, then scientific leaves and their exact ancestor closure.

    Base/scientific specs match production: pilot selection never changes an
    estimator parameter, seed, sample mask, CV grid, or content-addressed task ID.
    """
    selected_splits=[]
    for dataset,target in PILOT_TARGETS.items():
        candidates=[s for s in splits if s['dataset']==dataset and s['split_id']==target['split_id']]
        if len(candidates)!=1:raise ValueError(f'Expected exactly one frozen {dataset}/{target["split_id"]}')
        if config['primary_K'][dataset]!=target['primary_K']:
            raise ValueError('Pilot primary K differs from prespecified platform reference resolution')
        selected_splits.append(candidates[0])
    all_bases=build_base_manifests(selected_splits,config)['core']
    bases=[b for b in all_bases if b['panel_requested']==PILOT_TARGETS[b['dataset']]['panel_requested']
           and b['K'] in [PILOT_TARGETS[b['dataset']]['primary_K'],20]]
    if len(bases)!=6:raise ValueError('Pilot configuration must include six requested primary/K20 bases')
    manifests={'pilots':bases}
    complete=build_dag(manifests,config,source_identity['source_tree_hash'],contracts)
    selected=[]
    for task in complete['tasks']:
        if task['stage']!='evaluation':continue
        spec=task['spec'];primary=spec['K']==PILOT_TARGETS[spec['dataset']]['primary_K']
        if primary:
            selected.append(task['task_id'])
        elif spec.get('method') in config['competitors']:
            selected.append(task['task_id'])
        elif (spec.get('family')=='document' and spec.get('hunter')=='spa_current'
              and spec.get('preprocessing') in ['P0_raw','P3_tran_then_ke']):
            selected.append(task['task_id'])
    dag=deepcopy(complete);dag['tasks']=ancestral_closure(complete['tasks'],selected)
    if any(t['stage']=='diagnostics' for t in dag['tasks']):raise AssertionError('Lambda-path diagnostic tasks leaked into pilots')
    dag['pilot_design']={
        'kind':'full_observation_platform_pilots','hidden_subsampling':False,
        'training_observations':'all count-eligible observations in specified outer training split',
        'targets':PILOT_TARGETS,'K20':20,
        'primary_K_methods':f"all {len(config['preprocessings'])} preprocessings, {len(config['hunters'])} document hunters ({', '.join(config['hunters'])}), P0 anchor SPA, exact-zero and selected controls, all eight competitors; Poisson A only",
        'K20_methods':'P0/P3 document SPA, both exact-zero and selected controls, all eight competitors; Poisson A only',
        'native_A_recovery':'A_full_Pois for every GpLSI and competitor W; internally fitted competitor A is not reported',
        'common_reference':'Visium recovery/evaluation on same reference vocabulary for every selected W',
        'diagnostics':'excluded from pilot DAG; unmeasured diagnostic resource requests require explicit review',
        'production_task_ids_reusable':True}
    return manifests,dag


def write_pilot_manifest(root,config,source_identity,contracts):
    root=Path(root);splits=[]
    for dataset in PILOT_TARGETS:
        path=root/f'data/manifests/joint_v2/splits/{dataset}/splits.json'
        splits.extend(json.loads(path.read_text()))
    manifests,dag=select_pilot_dag(splits,config,source_identity,contracts)
    identity=fingerprint({'code_hash':dag['code_hash'],'config_hash':dag['config_hash'],
                          'source_hashes':contracts,'pilot_design':dag['pilot_design']})[:20]
    directory=root/f'data/manifests/joint_v2/pilots_{identity}'
    path=write_manifests(directory,manifests,dag)
    source_path=directory/'source_identity.json'
    if source_path.exists() and json.loads(source_path.read_text())!=source_identity:
        raise ValueError('Pilot source identity manifest differs')
    if not source_path.exists():atomic_json(source_path,source_identity)
    return {'manifest':str(path),'source_identity':str(source_path),
            **summarize_dag(dag,manifests),'pilot_design':dag['pilot_design']}


def _stage_path(root,task):
    base='data/interim/joint_v2/stages' if task['stage']=='prepare' else 'results/joint_v2/stages'
    return Path(root)/base/task['stage']/task['task_id']


def _correctness_evidence(root,expected_code_hash=None,expected_config_hash=None):
    paths=list((Path(root)/'reports/joint_v2').glob('correctness_*.xml'))
    if not paths:return {'passed':False,'status':'correctness_xml_missing'}
    path=max(paths,key=lambda p:(p.stat().st_mtime_ns,p.name))
    try:
        tree=ET.parse(path).getroot()
        cases=list(tree.iter('testcase'))
        declared_errors=max([int(x.get('errors','0')) for x in tree.iter() if x.tag in ('testsuite','testsuites')]+[0])
        declared_failures=max([int(x.get('failures','0')) for x in tree.iter() if x.tag in ('testsuite','testsuites')]+[0])
        errors=max(declared_errors,len(list(tree.iter('error'))))
        failures=max(declared_failures,len(list(tree.iter('failure'))))
        skipped=len(list(tree.iter('skipped')))
        xml_passed=bool(cases) and errors==0 and failures==0 and skipped<len(cases)
        result={'passed':xml_passed,'status':'passed' if xml_passed else 'correctness_failed_or_empty',
                'path':str(path),'sha256':sha256_file(path),'testcases':len(cases),
                'errors':errors,'failures':failures,'skipped':skipped}
    except (OSError,ValueError,ET.ParseError) as exc:
        return {'passed':False,'status':'invalid_correctness_xml','path':str(path),'error':str(exc)}
    # Bind this exact XML to the source snapshot recorded before its test run.
    # Never borrow another job's sidecar or fall back to an older green XML.
    sidecar=path.with_name(path.stem+'_source.json')
    required=expected_code_hash is not None or expected_config_hash is not None
    result.update({'source_sidecar':str(sidecar),'source_hash_matches':None,
                   'config_hash_matches':None,'source_binding':'unverified'})
    if not sidecar.exists():
        if required:result.update(passed=False,status='correctness_source_sidecar_missing')
        return result
    try:
        identity=json.loads(sidecar.read_text())
        actual_code_hash=identity['source']['source_tree_hash']
        actual_config_hash=identity['config_hash']
        code_matches=(actual_code_hash==expected_code_hash) if expected_code_hash is not None else None
        config_matches=(actual_config_hash==expected_config_hash) if expected_config_hash is not None else None
        compatible=code_matches is not False and config_matches is not False
        result.update({'source_sidecar_sha256':sha256_file(sidecar),
                       'source_tree_hash':actual_code_hash,'config_hash':actual_config_hash,
                       'source_hash_matches':code_matches,'config_hash_matches':config_matches,
                       'source_binding':'exact_requested_hashes' if required and compatible else
                                        'incompatible' if not compatible else 'recorded_not_compared'})
        if not compatible:result.update(passed=False,status='incompatible_correctness_source_or_config')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result.update(passed=False,status='invalid_correctness_source_sidecar',error=str(exc))
    return result


def _smoke_evidence(root,code_hash,config_hash):
    path=Path(root)/'reports/joint_v2/SMOKE_STATUS.json'
    if not path.exists():return {'passed':False,'status':'smoke_status_missing'}
    try:
        status=json.loads(path.read_text());report_path=Path(status['report'])
        report=json.loads(report_path.read_text())
        exact_source=report['source']['source_tree_hash']==code_hash
        exact_config=fingerprint(report['config'])==config_hash
        failures=report.get('substantive_failures',[])
        passed=status.get('status')=='passed' and report.get('status')=='passed' and exact_source and exact_config and not failures
        return {'passed':passed,'status':'passed' if passed else 'failed_or_incompatible_smoke',
                'source_hash_matches':exact_source,'config_hash_matches':exact_config,
                'report':str(report_path),'report_sha256':sha256_file(report_path),
                'substantive_failures':failures}
    except (OSError,ValueError,KeyError) as exc:
        return {'passed':False,'status':'invalid_smoke_evidence','error':str(exc)}


def _physical_bytes(directory):
    files=[p for p in Path(directory).rglob('*') if p.is_file() and not p.is_symlink()]
    return {'allocated_bytes':sum(getattr(p.stat(),'st_blocks',math.ceil(p.stat().st_size/512))*512 for p in files),
            'apparent_bytes':sum(p.stat().st_size for p in files),'files':len(files)}


def _blocked_resource(details):
    evidence=details.get('metadata',{})
    explicit=(evidence.get('status')=='resource_blocked'
              or {'point_pair_comparisons','max_diameter_pairs'}<=set(evidence)
              or (evidence.get('maximum_heap',0)>2_000_000 and 'point_pair_comparisons' in evidence))
    return details.get('type')=='GeometryResourceBlocked' and explicit and bool(evidence.get('reason') or details.get('error'))


def _resource_requests(measurements,config):
    grouped=defaultdict(list)
    for row in measurements:
        if row.get('state')=='complete' and row.get('runtime_seconds',0)>0 and row.get('peak_rss_bytes',0)>0:
            grouped[row['dataset']+'/'+row['stage']].append(row)
    requests={};review=[]
    for key,rows in sorted(grouped.items()):
        memory=max(r['peak_rss_bytes'] for r in rows);seconds=max(r['runtime_seconds'] for r in rows)
        request_seconds=max(600,int(math.ceil((1.5*seconds+600)/60)*60))
        memory_gb=max(4,int(math.ceil(1.5*memory/(1024**3))))
        hours,rest=divmod(request_seconds,3600);minutes,sec=divmod(rest,60)
        requests[key]={'cpus':config['resources']['pilot_cpus'],'memory_gb':memory_gb,
                       'time':f'{hours:02d}:{minutes:02d}:{sec:02d}',
                       'partition':config['resources']['default_partition'],
                       'basis':'observed maximum process RSS and wall time across successful full-observation pilots',
                       'n_measurements':len(rows),'observed_peak_rss_bytes':memory,
                       'observed_max_stage_allocated_bytes':max(r['artifact_storage']['allocated_bytes'] for r in rows),
                       'observed_max_stage_apparent_bytes':max(r['artifact_storage']['apparent_bytes'] for r in rows),
                       'observed_max_runtime_seconds':seconds,'memory_buffer':1.5,
                       'time_buffer':1.5,'fixed_time_overhead_seconds':600,
                       'review_required':True,'whole_job_Slurm_MaxRSS_verified':False}
    # Full lambda-path stages were intentionally not piloted. This bound is a
    # proposal, not measured evidence or an automatic production launch gate.
    for dataset in PILOT_TARGETS:
        spectral=requests.get(dataset+'/spectral')
        if spectral:
            factor=1+len(config['hunters'])
            sec=max(600,int(math.ceil((spectral['observed_max_runtime_seconds']*factor*1.5+600)/60)*60))
            h,rest=divmod(sec,3600);m,s=divmod(rest,60)
            requests[dataset+'/diagnostics']={
                'cpus':config['resources']['pilot_cpus'],'memory_gb':spectral['memory_gb'],
                'time':f'{h:02d}:{m:02d}:{s:02d}','partition':config['resources']['default_partition'],
                'basis':'UNMEASURED diagnostic bound from spectral maximum times (1 + number of hunters)',
                'spectral_runtime_multiplier':factor,'diagnostic_stage_pilot_measurements':0,
                'estimated_max_runtime_seconds':spectral['observed_max_runtime_seconds']*factor,
                'estimated_stage_allocated_bytes':spectral['observed_max_stage_allocated_bytes']*factor,
                'review_required':True,'automatically_approved':False}
            review.append(dataset+'/diagnostics has no actual pilot; multiplier does not certify runtime or memory')
        else:review.append(dataset+'/spectral has no successful resource measurement')
    return requests,review


def _production_extrapolation(path,dag,requests,count_directory_measurements):
    if path is None:
        return {'status':'production_manifest_required_for_total_estimate',
                'per_stage_resource_and_disk_maxima':'available in proposed_resource_requests',
                'estimate_is_measurement':False}
    path=Path(path);production=json.loads(path.read_text())
    if production['code_hash']!=dag['code_hash'] or production['config_hash']!=dag['config_hash']:
        raise ValueError('Production extrapolation manifest must use exact pilot source and scientific configuration')
    counts=Counter(t['dataset']+'/'+t['stage'] for t in production['tasks']);rows=[];unmeasured=[]
    for key,count in sorted(counts.items()):
        profile=requests.get(key)
        if not profile:
            unmeasured.append(key);continue
        runtime=profile.get('observed_max_runtime_seconds',profile.get('estimated_max_runtime_seconds'))
        storage=profile.get('observed_max_stage_allocated_bytes',profile.get('estimated_stage_allocated_bytes'))
        rows.append({'profile':key,'expected_stage_tasks':count,
                     'estimated_allocated_cpu_hours':count*runtime*profile['cpus']/3600,
                     'estimated_stage_allocated_bytes':count*storage,
                     'basis':'pilot maximum per successful stage' if 'n_measurements' in profile else 'unmeasured diagnostic multiplier',
                     'same_class_different_outer_splits_K_panels':'resource extrapolation requiring review, not a guaranteed upper bound'})
    countgroups=defaultdict(set)
    for task in production['tasks']:
        if task['stage']=='prepare':
            s=task['spec'];countgroups[task['dataset']].add(tuple(s[k] for k in
                ['protocol','outer_split_id','retention','molecule_seed']))
    caches=[]
    for dataset,groups in countgroups.items():
        measured=[m['allocated_bytes'] for m in count_directory_measurements if m['dataset']==dataset]
        if not measured:unmeasured.append(dataset+'/full_vocabulary_count_cache');continue
        caches.append({'dataset':dataset,'unique_count_caches':len(groups),
                       'estimated_allocated_bytes':max(measured)*len(groups),
                       'basis':'largest observed whole-count-cache allocated size; no thinning size discount'})
    return {'status':'requires_resource_and_storage_review','manifest':str(path),'manifest_sha256':sha256_file(path),
            'per_stage':rows,'count_caches':caches,'unmeasured_profiles':unmeasured,
            'estimated_allocated_cpu_hours':sum(r['estimated_allocated_cpu_hours'] for r in rows),
            'estimated_additional_stage_and_count_bytes':sum(r['estimated_stage_allocated_bytes'] for r in rows)+sum(r['estimated_allocated_bytes'] for r in caches),
            'excludes':'raw/processed cohort baseline, figures and aggregate reports, filesystem replication/quotas',
            'estimate_is_measurement':False,'production_approved':False}


def assess_pilots(root,manifest,*,production_manifest=None):
    """Validate all selected stages and report measured, reviewable resources.

    Resource-blocked geometry and its descendants are explicit terminal scientific
    dispositions; OOM, timeout, corruption and other exceptions are failures.
    Scheduler terminal/live-job checks remain a separate root-controller audit.
    """
    root=Path(root);manifest=Path(manifest);dag=json.loads(manifest.read_text())
    if dag.get('pilot_design',{}).get('hidden_subsampling') is not False:
        raise ValueError('Assessment requires the prespecified full-observation pilot manifest')
    mapping={task['task_id']:task for task in dag['tasks']};rows=[];states={}
    unique_count_directories=set()
    for task in dag['tasks']:
        directory=_stage_path(root,task)
        row={'task_id':task['task_id'],'dataset':task['dataset'],'stage':task['stage'],
             'variant':task['variant'],'K':task['spec'].get('K'),'path':str(directory),
             'artifact_storage':_physical_bytes(directory) if directory.exists() else {'allocated_bytes':0,'apparent_bytes':0,'files':0}}
        marker=directory/'complete.json'
        if compatible_completed(directory,task['task_id'],verify_hashes=True):
            effective=json.loads((directory/'effective.json').read_text())
            if effective.get('task',{}).get('code_hash')!=dag['code_hash']:
                row.update(state='failed',error='Completed pilot effective source hash mismatch')
            else:
                row.update(state='complete',runtime_seconds=effective['runtime_seconds'],
                           peak_rss_bytes=effective['peak_rss_bytes'],slurm_job_id=effective.get('slurm_job_id'),
                           partition=effective.get('partition'),
                           n_training_observations=sum(x.get('n_observations',0) for x in effective.get('prepared',{}).get('training_sample_contributions',[])) or None,
                           process_memory_scope='resource.getrusage(RUSAGE_SELF); verify scheduler whole-job peak before production')
                key=effective.get('prepared',{}).get('cache_key')
                if key:unique_count_directories.add(root/f'data/interim/joint_v2/count_splits/{task["dataset"]}/{key}')
                fit=directory/'fit.json'
                if fit.exists():
                    metadata=json.loads(fit.read_text());row['scientific_converged']=metadata.get('converged')
                    row['scientific_status']=metadata.get('status',metadata.get('vertex_status'))
                spectral=directory/'spectral.json'
                if spectral.exists():
                    metadata=json.loads(spectral.read_text());row['cv_boundary_status']=metadata.get('metadata',{}).get('cv',{}).get('boundary_status')
        elif marker.exists():
            row.update(state='failed',error='Completed artifact failed checksum/identity validation')
        else:
            attempts=list((directory/'attempts').glob('*.json'))
            if attempts:
                path=max(attempts,key=lambda p:(p.stat().st_mtime_ns,p.name));details=json.loads(path.read_text())
                row.update(state='resource_blocked' if _blocked_resource(details) else 'failed',
                           error_type=details.get('type'),error=details.get('error'),attempt=str(path))
                if _blocked_resource(details):row['resource_evidence']=details['metadata']
            else:row['state']='pending'
        rows.append(row);states[task['task_id']]=row
    for row in rows:
        if row['state']=='complete':
            unavailable=[parent for parent in mapping[row['task_id']]['parents']
                         if states[parent]['state']!='complete']
            if unavailable:
                row.update(state='failed',error='Completed child has an unavailable immutable parent',
                           unavailable_parents=unavailable)
    # Repeated relaxation handles arbitrary valid DAG ordering; failures never
    # acquire a favorable resource-blocked classification from another parent.
    changed=True
    while changed:
        changed=False
        for row in rows:
            if row['state']!='pending':continue
            parent_states=[states[p]['state'] for p in mapping[row['task_id']]['parents']]
            if parent_states and any(s in ['resource_blocked','blocked_by_resource_parent'] for s in parent_states):
                if all(s in ['complete','resource_blocked','blocked_by_resource_parent'] for s in parent_states):
                    row.update(state='blocked_by_resource_parent',
                               blocked_parents=[p for p in mapping[row['task_id']]['parents']
                                                if states[p]['state']!='complete'])
                    changed=True
    counts=Counter(row['state'] for row in rows)
    terminal={'complete','resource_blocked','blocked_by_resource_parent'}
    all_terminal=bool(rows) and all(row['state'] in terminal for row in rows)
    correctness=_correctness_evidence(root,dag['code_hash'],dag['config_hash'])
    smoke=_smoke_evidence(root,dag['code_hash'],dag['config_hash'])
    requests,resource_review=_resource_requests(rows,dag['config'])
    stage_storage={name:sum(row['artifact_storage'][name] for row in rows)
                   for name in ['allocated_bytes','apparent_bytes','files']}
    count_directory_measurements=[{'dataset':path.parent.name,'path':str(path),**_physical_bytes(path)}
                                  for path in unique_count_directories]
    count_storage={name:sum(row[name] for row in count_directory_measurements)
                   for name in ['allocated_bytes','apparent_bytes','files']}
    readiness=correctness['passed'] and smoke['passed'] and all_terminal
    report={'schema':'joint_v2_pilot_assessment','manifest':str(manifest),'manifest_sha256':sha256_file(manifest),
            'code_hash':dag['code_hash'],'config_hash':dag['config_hash'],
            'status':'ready_for_resource_review' if readiness else 'blocked_by_failures' if counts['failed'] else 'incomplete',
            'correctness':correctness,'smoke':smoke,'full_size_pilots_terminal':all_terminal,
            'scientific_terminal_dispositions':sorted(terminal),'stage_state_counts':dict(counts),
            'expected_stages':len(rows),'stage_results':rows,'proposed_resource_requests':requests,
            'resource_review_required':resource_review+[
                'Check scheduler terminal states and whole-job MaxRSS, elapsed time, CPU usage and partition limits.',
                'Check free/quota storage against the complete production workload, not just pilot bytes.',
                'Review correctness XML against its source snapshot and scientific nonconvergence/boundary flags.',
                'Diagnostic stage requests are estimated, not measured; broad K/panel/outer scaling remains to review.'],
            'observed_storage':{'stages':stage_storage,'unique_count_caches':count_storage,
                                'per_count_cache':count_directory_measurements,
                                'scope':'actual allocated and apparent bytes of selected stage/cache files; source raw/processed cohort excluded'},
            'production_workload_extrapolation':_production_extrapolation(production_manifest,dag,requests,count_directory_measurements),
            'observed_stage_wall_seconds':sum(row.get('runtime_seconds',0) for row in rows),
            'allocated_cpu_hours_estimate':sum(row.get('runtime_seconds',0) for row in rows)*dag['config']['resources']['pilot_cpus']/3600,
            'cpu_estimate_basis':'pilot requested CPUs times stage wall time; not measured user+system CPU time',
            'production_approved':False,'PRODUCTION_GATE_written':False}
    output=root/f'reports/joint_v2/{manifest.parent.name}';output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'pilot_assessment.json',report)
    atomic_json(output/'proposed_resources.json',requests)
    report['assessment_path']=str(output/'pilot_assessment.json')
    report['proposed_resources_path']=str(output/'proposed_resources.json')
    return report
