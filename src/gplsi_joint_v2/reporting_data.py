"""Prespecified coordinate split masks and training-only frequency diagnostics."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from .data import _atomic_json,_file_records,_lock,_verify_files,frequency_statistics


def plot_frozen_split_masks(root,dataset='visium_dlpfc'):
    """Plot every omitted spatial half, with shared donor/section layout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);directory=root/f'data/manifests/joint_v2/splits/{dataset}'
    splits=json.loads((directory/'splits.json').read_text())
    output=root/f'figures/joint_v2/{dataset}/frozen_split_masks';output.mkdir(parents=True,exist_ok=True)
    files=[]
    for split in splits:
        if split['protocol']!='spatial_half':continue
        frame=pd.read_parquet(directory/f"{split['split_id']}.parquet")
        sections=frame[['bio_id','section_id']].drop_duplicates().sort_values(['bio_id','section_id'])
        fig,axes=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
        for ax,(_,row) in zip(axes.flat,sections.iterrows()):
            local=frame.loc[frame.section_id==row.section_id]
            for role,color in [('train','#2878B5'),('spatial_half','#E69F00')]:
                selected=local.loc[local.role==role]
                ax.scatter(selected.x,selected.y,c=color,s=2,rasterized=True,
                           label=f"{role}: {len(selected):,}")
            ax.set_title(f'{row.bio_id} / {row.section_id}',fontsize=9)
            ax.set_aspect('equal');ax.set_xlabel('array x');ax.set_ylabel('array y')
            ax.legend(fontsize=6,markerscale=2,loc='best')
        fig.suptitle(f"{split['split_id']}: coordinate-only omitted halves\n{split['tie_rule']}")
        path=output/f"{split['split_id']}.png";fig.savefig(path,dpi=150);plt.close(fig);files.append(str(path))
    return files


def frequency_report(root,dataset,prepared):
    """Native-panel diagnostics; full arrays and per-unit statistics are saved.

    Called once per immutable count split and panel. No K, fitted factors,
    scoring counts, biological labels, or spatial outcomes enter these plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);key=prepared.split_metadata['cache_key'];panel=prepared.split_metadata['panel_requested']
    directory=root/f'data/interim/joint_v2/frequency/{dataset}/{key}/panel_{panel}'
    directory.mkdir(parents=True,exist_ok=True)
    output=root/f'figures/joint_v2/{dataset}/frequency/{key}';output.mkdir(parents=True,exist_ok=True)
    image=output/f'panel_{panel}.png'
    with _lock(directory/'.report.lock'):
        if (directory/'complete.json').exists():
            complete=json.loads((directory/'complete.json').read_text());_verify_files(directory,complete['artifacts'])
            _verify_files(output,complete['figure_artifacts']);return complete
        stats=frequency_statistics(prepared.train_fit,prepared.train_bio_ids)
        rows=pd.DataFrame({'feature_id':prepared.feature_ids,
                          'mean_row_normalized_frequency':stats['mean_row_normalized_frequency'],
                          'pooled_molecule_frequency':stats['pooled_molecule_frequency'],
                          'detected_training_observations':stats['detected'],'prevalence':stats['prevalence']})
        rows.to_parquet(directory/'gene_frequencies.parquet',index=False)
        unit_tables=[];unit_summaries=[]
        for bio,values in stats['per_biological_unit'].items():
            unit_tables.append(pd.DataFrame({'bio_id':bio,'feature_id':prepared.feature_ids,
                  'mean_row_normalized_frequency':values['mean_row_normalized_frequency'],
                  'pooled_molecule_frequency':values['pooled_molecule_frequency'],
                  'detected_training_observations':values['detected'],'prevalence':values['prevalence']}))
            unit_summaries.append({'bio_id':bio,**{k:v for k,v in values.items() if np.isscalar(v)}})
        pd.concat(unit_tables,ignore_index=True).to_parquet(directory/'per_biological_unit_frequencies.parquet',index=False)
        summary={k:v for k,v in stats.items() if np.isscalar(v)}
        summary.update({'per_biological_unit':unit_summaries,'panel_requested':panel,
                        'panel_actual':len(prepared.feature_ids),'feature_order_sha256':prepared.split_metadata['feature_order_sha256'],
                        'counts_used':'outer training fitting molecules only','threshold_definition':'mean row-normalized frequency, strict >',
                        'ranking_uses':'raw training count variance/mean; stable original-column ties'})
        _atomic_json(directory/'summary.json',summary)
        fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
        colors=['#2878B5','#E69F00'];labels=['Mean row-normalized','Pooled molecule']
        for values,label,color in zip([stats['mean_row_normalized_frequency'],stats['pooled_molecule_frequency']],labels,colors):
            positive=values[values>0];ranked=np.sort(positive)[::-1]
            if len(positive):
                lo,hi=positive.min(),positive.max()
                if lo==hi:lo*=.5;hi*=2
                axes[0,0].hist(positive,bins=np.geomspace(lo,hi,40),alpha=.45,label=label,color=color)
                axes[0,1].plot(np.arange(1,len(ranked)+1),ranked,label=label,color=color)
                axes[1,1].plot(np.arange(1,len(ranked)+1),np.cumsum(ranked)/ranked.sum(),label=label,color=color)
        axes[0,0].set_xscale('log');axes[0,0].set_xlabel('Positive gene frequency');axes[0,0].set_ylabel('Genes')
        if stats['tran_threshold']>0:axes[0,0].axvline(stats['tran_threshold'],ls='--',color='black',label='Tran threshold (row mean)')
        axes[0,0].legend(fontsize=8)
        axes[0,1].set_xscale('log');axes[0,1].set_yscale('log');axes[0,1].set_xlabel('Frequency rank');axes[0,1].set_ylabel('Gene frequency')
        axes[0,1].legend(fontsize=8)
        axes[1,0].hist(stats['prevalence'],bins=np.linspace(0,1,41),color=colors[0]);axes[1,0].set_xlabel('Fraction of training observations detecting gene');axes[1,0].set_ylabel('Genes')
        axes[1,1].set_xscale('log');axes[1,1].set_ylim(0,1.01);axes[1,1].set_xlabel('Number of genes (rank order)');axes[1,1].set_ylabel('Cumulative frequency mass');axes[1,1].legend(fontsize=8)
        fig.suptitle(f"{dataset}, {prepared.split_metadata['outer_split']['split_id']}, panel {panel} (actual {len(prepared.feature_ids)})\n"
                     f"{stats['zero_frequency_count']} zero-frequency genes; Tran survivors {stats['tran_strict_survival_count']}; "
                     f"fallback={stats['tran_top_10pct_fallback_triggered']}")
        partial=image.with_name(image.stem+f'.partial.{os.getpid()}.png')
        fig.savefig(partial,dpi=150);plt.close(fig);os.replace(partial,image)
        complete={'artifacts':_file_records(directory,['gene_frequencies.parquet','per_biological_unit_frequencies.parquet','summary.json']),
                  'figure_artifacts':_file_records(output,[image.name]),'figure':str(image),'data_directory':str(directory)}
        _atomic_json(directory/'complete.json',complete)
        return complete
