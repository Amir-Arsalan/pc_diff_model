import torch
from chamferdist import ChamferDistance
from evaluation.emd_module import emdModule

def EMD_CD(sample_pcs, ref_pcs, reduced=True):
    batch_size = sample_pcs.size(0)
    chamfDist = ChamferDistance()
    emd = emdModule()
    cd = chamfDist(sample_pcs.clone().cpu(), ref_pcs.clone().cpu(), bidirectional=True, batch_reduction=None if not reduced else 'mean')
    emd_dist, _ = emd(sample_pcs, ref_pcs, 0.002, 10000)
    emd_dist = emd_dist if not reduced else emd_dist.mean()
    
    return cd, emd_dist