"""Inference-only Z1/Z4 correction-path backtrace for the Pre-FAST model."""
import argparse, json, math, random
from pathlib import Path
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_direct_state_correction import make_paths, load_role_components, next_role_prediction, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import error_state, residual_writeback_host_feature
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_decoupled_semantic_temporal_corrections

CHECKPOINT="results/kitti_step_convgru_decoupling_quick_0c83ad0/full.pt"
VARIANTS=("Legacy-None","Legacy-Z1","Legacy-Z4","Legacy-Full")
SEQ_DEV3=("0002","0010","0018")
SEQ_FULL9=("0002","0006","0007","0008","0010","0013","0014","0016","0018")

def load_corrections(path):
    corrections=build_decoupled_semantic_temporal_corrections()
    corrections.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    corrections.requires_grad_(False); corrections.eval(); return corrections

def posterior_variants(observation,posterior1,posterior4):
    return {
        "Legacy-None": UnifiedFeatures(observation.z1,observation.z2,observation.z3,observation.z4),
        "Legacy-Z1": UnifiedFeatures(posterior1,observation.z2,observation.z3,observation.z4),
        "Legacy-Z4": UnifiedFeatures(observation.z1,observation.z2,observation.z3,posterior4),
        "Legacy-Full": UnifiedFeatures(posterior1,observation.z2,observation.z3,posterior4),
    }

def decode_variant(model,raw,observation,posterior,size):
    delta=UnifiedFeatures(posterior.z1-observation.z1,torch.zeros_like(observation.z2),torch.zeros_like(observation.z3),posterior.z4-observation.z4)
    return model.decode_from_host_feature(residual_writeback_host_feature(model,raw,delta,size)).argmax(1)

def evaluate(model,predictor,corrections,groups,raft):
    confusion={v:torch.zeros((19,19),dtype=torch.int64) for v in VARIANTS}; mvc={v:VideoConsistency(VARIANTS) for v in VARIANTS}; mtc_sum={v:0. for v in VARIANTS}; mtc_count={v:0 for v in VARIANTS}; per={}
    with torch.inference_mode():
      for seq,samples in groups.items():
        seq_conf={v:torch.zeros((19,19),dtype=torch.int64) for v in VARIANTS}; seq_vc=VideoConsistency(VARIANTS); seq_tsum={v:0. for v in VARIANTS}; seq_n=0; prev_preds=None; prev_img=None
        h4d=h4s=h1d=h1s=None; pending_dyn=pending_sem=None
        for index,sample in enumerate(samples):
            image=load_image(sample); raw=model.extract_backbone_features(image); observation=model.encode_backbone_features(raw); size=tuple(image.shape[-2:]); mask=semantic_mask_from_panoptic_png(sample["mask_path"])
            host_pred=model.decode_from_host_feature(HostFeature(raw.c4,raw.c1,size)).argmax(1)
            if index==0:
                pending_dyn,pending_sem,h4d,h4s,h1d,h1s=predictor.step(observation,zero_state(observation),None,None,None,None)
                predictions={v:host_pred for v in VARIANTS}
            else:
                error=error_state(observation,pending_dyn)
                # Both correction modules are always evaluated and both hidden
                # states advance, even when one output delta is masked.
                posterior1,hidden1,values1=corrections[0](observation.z1,pending_dyn.z1,pending_sem.z1,None if index==1 else hidden1)
                posterior4,hidden4,values4=corrections[1](observation.z4,pending_dyn.z4,pending_sem.z4,None if index==1 else hidden4)
                posts=posterior_variants(observation,posterior1,posterior4)
                predictions={v:decode_variant(model,raw,observation,p,size) for v,p in posts.items()}
                pending_dyn,pending_sem,h4d,h4s,h1d,h1s=predictor.step(observation,error,h4d,h4s,h1d,h1s)
            for v,pred in predictions.items():
                p=pred.squeeze(0).cpu(); update_confusion_matrix(confusion[v],p,mask); update_confusion_matrix(seq_conf[v],p,mask)
            mvc_seq={v:predictions[v].squeeze(0).cpu() for v in VARIANTS}; seq_vc.append(mask,mvc_seq)
            if prev_preds is not None:
                flow=raft.current_to_previous(image,prev_img); grid,valid=flow_grid(flow,image.shape[-2],image.shape[-1])
                for v in VARIANTS:
                    warped=F.grid_sample(prev_preds[v].float()[None],grid,mode="nearest",padding_mode="zeros",align_corners=True)[0,0].long(); keep=valid[0]; a,b=warped[keep].cpu(),predictions[v][0][keep].cpu(); pc=torch.zeros((19,19),dtype=torch.int64)
                    if a.numel(): pc += torch.bincount(19*a+b,minlength=361).reshape(19,19)
                    score=float(torch.nanmean(compute_iou(pc)).item())
                    if math.isfinite(score): mtc_sum[v]+=score; mtc_count[v]+=1; seq_tsum[v]+=score
                seq_n+=1
            prev_preds={v:predictions[v].detach() for v in VARIANTS}; prev_img=image
        vals=seq_vc.means()
        for v in VARIANTS:
            per.setdefault(seq,{})[v]={"mTC":seq_tsum[v]/seq_n if seq_n else float("nan"),"mVC8":vals[8][v],"mVC16":vals[16][v],"mIoU":float(torch.nanmean(compute_iou(seq_conf[v])).item())}
        per[seq]["valid_frame_pairs"]=seq_n
    result={}
    for v in VARIANTS:
        iou=compute_iou(confusion[v]); result[v]={"mIoU":float(torch.nanmean(iou).item()),"wIoU":weighted_iou(confusion[v]),"mVC8":sum(per[s][v]["mVC8"] for s in per)/len(per),"mVC16":sum(per[s][v]["mVC16"] for s in per)/len(per),"mTC":mtc_sum[v]/mtc_count[v],"valid_frame_pairs":mtc_count[v]}
    result["per_sequence"]=per; return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/home/lin/predify/kitti_step"); ap.add_argument("--checkpoint",default=CHECKPOINT); ap.add_argument("--output",default="results/kitti_step_z1_z4_temporal_backtrace"); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    paths=make_paths(); model,predictor=load_role_components(paths["static"],paths["adapter"],paths["predictor"],paths["writeback"]); corrections=load_corrections(args.checkpoint); raft=FrozenRAFT()
    ds=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"val"); groups=sequence_groups(ds); dev={s:groups[s] for s in SEQ_DEV3}; full={s:groups[s] for s in SEQ_FULL9}
    dev_result=evaluate(model,predictor,corrections,dev,raft)
    # Reinitialize recurrent states by reloading the immutable components for
    # full9, preventing dev3 state from crossing the protocol boundary.
    model,predictor=load_role_components(paths["static"],paths["adapter"],paths["predictor"],paths["writeback"])
    full_result=evaluate(model,predictor,corrections,full,raft)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); payload={"checkpoint":args.checkpoint,"dev3":dev_result,"full9":full_result,"variants":VARIANTS}; (output/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); (output/"README.md").write_text("# Z1/Z4 temporal backtrace\nInference-only Legacy-None/Z1/Z4/Full evaluation.\n")
    print(json.dumps(payload,indent=2,sort_keys=True),flush=True)

if __name__=="__main__": main()
