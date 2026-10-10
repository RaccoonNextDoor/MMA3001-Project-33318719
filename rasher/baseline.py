"""Week-5 regression baselines for annotated defect count per image.

Validation-only development: this module never evaluates test image predictions.
Counts measure annotations, not rashers, physical severity or food safety.
"""
from pathlib import Path
import csv
import json
import platform
import time
import warnings
import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from .features import extract_features


def metrics(actual, predicted):
    """Return MAE/RMSE in defects per image and dimensionless R-squared."""
    return {'mae':float(mean_absolute_error(actual,predicted)),
            'rmse':float(np.sqrt(mean_squared_error(actual,predicted))),
            'r2':float(r2_score(actual,predicted))}


def candidates():
    """Predeclared small search, selected by validation MAE (seed 42)."""
    yield 'mean',DummyRegressor(strategy='mean')
    yield 'median',DummyRegressor(strategy='median')
    yield 'linear',make_pipeline(StandardScaler(),LinearRegression())
    for depth in (3,6,12):
        yield f'tree_d{depth}',DecisionTreeRegressor(max_depth=depth,min_samples_leaf=5,random_state=42)
    for c in (0.1,1.,10.):
        yield f'svr_c{c:g}',make_pipeline(StandardScaler(),SVR(C=c,epsilon=.1))
    yield 'mlp_32',make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,),alpha=.1,
                                max_iter=400,early_stopping=False,random_state=42))


def run_baselines(data_dir, output):
    """Fit 2 feature resolutions x 10 candidates on unique training sources.

    Args:
        data_dir: Directory prepared by rasher.data.prepare.
        output: New results directory.
    Returns:
        Summary dictionary. Writes metrics, predictions, model, plots and environment.
    """
    data_dir,output=Path(data_dir),Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    records={}
    for split in ('train','valid'):
        unique={}
        for r in sorted(manifest,key=lambda r:r['file_name']):
            if r['split']==split: unique.setdefault(r['source'],r)
        records[split]=list(unique.values())
    results=[];winner=None;best=np.inf;feature_seconds={}
    with threadpool_limits(limits=2):
        for grid in (2,4):
            xy={};start=time.perf_counter()
            for s,rs in records.items():
                x=np.array([extract_features(data_dir/'images'/s/r['file_name'],grid) for r in rs])
                y=np.array([r['count'] for r in rs],dtype=float)
                xy[s]=(x,y)
            feature_seconds[str(grid)]=time.perf_counter()-start
            xt,yt=xy['train'];xv,yv=xy['valid']
            for name,model in candidates():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    start=time.perf_counter();model.fit(xt,yt);fit_s=time.perf_counter()-start
                start=time.perf_counter()
                for _ in range(20): raw=model.predict(xv)
                predict_ms=(time.perf_counter()-start)*1000/(20*len(xv))
                prediction=np.maximum(0,raw) # Explicit nonnegative postprocessing, no rounding.
                row={'model':name,'grid':grid,'features':xt.shape[1],**metrics(yv,prediction),
                     'train_mae':metrics(yt,np.maximum(0,model.predict(xt)))['mae'],
                     'fit_seconds':fit_s,'batch_ms_per_image':predict_ms,
                     'warnings':' | '.join(sorted(set(str(w.message) for w in caught)))}
                results.append(row);print(name,grid,round(row['mae'],4),flush=True)
                if row['mae'] < best:
                    best=row['mae'];winner=(model,grid,name,prediction.copy(),yv.copy())
    with (output/'validation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    model,grid,name,prediction,actual=winner
    joblib.dump({'model':model,'grid':grid,'target':'annotated defect count'},output/'count_model.joblib')
    with (output/'validation_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file_name','true_count','predicted_count'])
        w.writerows((r['file_name'],int(y),float(p)) for r,y,p in zip(records['valid'],actual,prediction))
    summary={'status':'DEVELOPMENT VALIDATION ONLY - test not evaluated','target':'annotated defect count per image',
             'selected_model':name,'grid':grid,'selection_metric':'validation MAE',
             'train_unique_sources':len(records['train']),'validation_unique_sources':len(records['valid']),
             'metrics':metrics(actual,prediction),'feature_extraction_seconds':feature_seconds,
             'environment':{'python':platform.python_version(),'platform':platform.platform(),'sklearn':sklearn.__version__,
                            'numpy':np.__version__,'threads':2},
             'limitations':['Image counts are not object detection or verified pass/fail labels.',
                            'Validation used for selection; scores are optimistic for final generalisation.',
                            'Adjacent-frame/tray correlation remains unresolved.',
                            'Timing is batch prediction only; excludes image decoding and descriptors.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for g in (2,4):
        subset=[r for r in results if r['grid']==g]
        axes[0].plot([r['model'] for r in subset],[r['mae'] for r in subset],'-o',label=f'{g}x{g} grid')
    axes[0].tick_params(axis='x',rotation=55);axes[0].set_ylabel('Validation MAE (defects/image)');axes[0].legend()
    axes[1].scatter(actual,prediction,alpha=.4);lim=max(actual.max(),prediction.max())+.2
    axes[1].plot([0,lim],[0,lim],'k--');axes[1].set(xlabel='Annotated count',ylabel='Predicted count',title=f'Selected {name}, {grid}x{grid}')
    fig.suptitle('Development validation results - not final test performance')
    fig.savefig(output/'validation_results.png',dpi=170);plt.close(fig)
    return summary
