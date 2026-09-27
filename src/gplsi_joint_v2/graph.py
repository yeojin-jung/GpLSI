"""Coordinate-only blocked six-neighbor graph with bounded-memory queries."""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree


def neighbor_indices(coords, graph_ids, k=6):
    xy=np.asarray(coords,dtype=float); ids=np.asarray(graph_ids,dtype=str)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)!=len(ids) or not np.all(np.isfinite(xy)) or k<1:
        raise ValueError("Coordinates and graph IDs must be finite and aligned")
    neighbors=np.full((len(xy),k),-1,dtype=np.int64)
    distances=np.full((len(xy),k),np.nan,dtype=float)
    for group in np.unique(ids):
        rows=np.flatnonzero(ids==group); count=min(k,len(rows)-1)
        if not count: continue
        d,j=cKDTree(xy[rows]).query(xy[rows],k=count+1,workers=1)
        for q in range(len(rows)):
            # Explicit self exclusion also works when duplicate coordinates cause
            # a different zero-distance observation to be returned first.
            valid=j[q]!=q
            order=np.lexsort((rows[j[q][valid]],d[q][valid]))[:count]
            neighbors[rows[q],:count]=rows[j[q][valid][order]]
            distances[rows[q],:count]=d[q][valid][order]
    return neighbors,distances


def build_graph(coords, graph_ids, k=6):
    """Return symmetric CSR exp(-(distance/stratum median positive d)^2).

    Each stratum is independently constructed; symmetrization uses maximum,
    matching the historical benchmark. Zero-positive-distance strata use scale1.
    """
    xy=np.asarray(coords,dtype=float); ids=np.asarray(graph_ids,dtype=str)
    neighbors,distances=neighbor_indices(xy,ids,k)
    rows=np.repeat(np.arange(len(xy)),k); cols=neighbors.ravel(); d=distances.ravel()
    keep=cols>=0; rows=rows[keep];cols=cols[keep];d=d[keep]
    weights=np.empty(len(d),dtype=float); scales={}
    for group in np.unique(ids):
        mask=ids[rows]==group; positive=d[mask&(d>0)]
        scale=float(np.median(positive)) if len(positive) else 1.
        weights[mask]=np.exp(-np.square(d[mask]/scale));scales[str(group)]=scale
    graph=coo_matrix((weights,(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    graph=graph.maximum(graph.T);graph.setdiag(0);graph.eliminate_zeros();graph.sort_indices()
    r,c=graph.nonzero()
    if np.any(ids[r]!=ids[c]): raise AssertionError("Forbidden cross-stratum edge")
    return graph, {"n_observations":len(xy),"n_edges_undirected":graph.nnz//2,
                   "k":k,"symmetrization":"maximum","weight_scales":scales,
                   "weight_sum_directed":float(graph.sum()),
                   "penalty_convention":"historical weighted incidence sum; no n/edge rescaling"}
