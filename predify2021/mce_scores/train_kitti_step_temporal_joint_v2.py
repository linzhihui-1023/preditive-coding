"""Controlled Temporal Joint V2 training for FAST-B.

This file intentionally leaves the model architecture untouched.  It adds the
V2 training objective around the existing FAST-B predictor and keeps a strict
zero-step checkpoint equivalence check before any backward pass.
"""
import argparse, csv, json, math, random
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT, load_components, error_state, zero_state,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor, HostFeature, UnifiedFeatures
from predify2021.mce_scores.train_kitti_step_temporal_joint import (
    FrozenRAFT, encode_clean, flow_grid, detach_state, C4_ADAPTER_INDEX, C4_WRITEBACK_KEY,
)

SEED=0; TBPTT=8; NUM_CLASSES=19; IGNORE=255
FAST_B="/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/fast_b_joint_c4_weak_z4/best.pt"
OUT="/home/lin/predify/experiments/kitti_step_temporal_joint_v2"
RES="results/kitti_step_temporal_joint_v2"
DEV3=("0002","0010","0018")
REF={"mIoU":0.5788293035364273,"mVC8":0.945077080607307,"mVC16":0.9421164431668695,"mTC":0.7050058541840838}
HOST={"mIoU":0.5802101104390269,"mVC8":0.9345485965278791,"mVC16":0.9273142366723673,"mTC":0.7009239766436142}

def configure(model,predictor,stage=1):
    model.requires_grad_(False); predictor.requires_grad_(False)
    for m in (predictor.semantic_error_encoder,predictor.semantic_state_cell,predictor.semantic_restoration_head,model.multi_layer_adapter.output_adapters[3],model.host_conditioned_writebacks["3"]): m.requires_grad_(True)
    if stage==2:
        for n in ("z4_dyn_recurrent","z4_dyn_delta"): getattr(predictor,n).requires_grad_(True)
    model.eval(); predictor.train()

def load_pair(args,stage=1):
    def one():
        m,_=load_components(STATIC_CHECKPOINT_DEFAULT,ADAPTER_CHECKPOINT_DEFAULT,args.dynamics_checkpoint,WRITEBACK_CHECKPOINT_DEFAULT)
        pld=torch.load(args.fast_b_checkpoint,map_location="cpu",weights_only=False)
        m.multi_layer_adapter.output_adapters[3].load_state_dict(pld["c4_output_adapter_state_dict"])
        m.host_conditioned_writebacks["3"].load_state_dict(pld["c4_writeback_state_dict"])
        p=ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=True).cuda(); p.load_state_dict(pld["model_state_dict"]); configure(m,p,stage); return m,p,pld
    return one(), one()

def logits_for(model,raw,obs,rest,size):
    z=zero_state(obs); d=UnifiedFeatures(z.z1,z.z2,z.z3,rest.z4-obs.z4)
    hf=residual_writeback_host_feature(model,raw,d,size)
    if torch.is_grad_enabled():
        return checkpoint(lambda x: model.decode_from_host_feature(HostFeature(x, hf.low_level, hf.output_size)), hf.tensor, use_reentrant=False)
    return model.decode_from_host_feature(hf)

def temporal_loss(cur,prev,flow,prev_mask,cur_mask):
    h,w=cur.shape[-2:]; grid,valid=flow_grid(flow,h,w)
    cp=cur.softmax(1); pp=prev.detach().softmax(1)
    wp=F.grid_sample(pp,grid,mode="bilinear",padding_mode="zeros",align_corners=True)
    pm=F.grid_sample(prev_mask.float()[None,None],grid,mode="nearest",padding_mode="zeros",align_corners=True)[0,0].long()
    valid=valid[0] & (pm!=IGNORE) & (cur_mask!=IGNORE) & (pm==cur_mask)
    conf=prev.detach().softmax(1).amax(1)[0]>.7
    conf=F.grid_sample(conf.float()[None,None],grid,mode="nearest",padding_mode="zeros",align_corners=True)[0,0]>.5
    valid &= conf
    if not valid.any(): return cur.sum()*0,0.0
    return F.kl_div(cp.clamp_min(1e-8).log(),wp.clamp_min(1e-8),reduction="none").sum(1)[0][valid].mean(),float(valid.float().mean())

@torch.inference_mode()
def evaluate(model,predictor,groups,raft):
    conf=torch.zeros((NUM_CLASSES,NUM_CLASSES),dtype=torch.int64); mvc_sum={8:0.,16:0.}; mvc_count={8:0,16:0}; sums=0.; n=0; per={}
    for seq,samples in groups.items():
        if len(samples)<2: continue
        sc=torch.zeros_like(conf); sv=VideoConsistency(); sm=0.; sn=0
        img,obs,raw,size=encode_clean(model,samples[0]); mask=semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        pred=model.decode_from_host_feature(HostFeature(raw.c4,raw.c1,size)).argmax(1); update_confusion_matrix(conf,pred[0].cpu(),mask); update_confusion_matrix(sc,pred[0].cpu(),mask); sv.update(mask,pred)
        prev_img=img; prev=pred; pending,h4,h1=predictor.predict_next(obs,zero_state(obs),None,None); sh=predictor.initial_semantic_state(obs); es=predictor.initial_error_temporal_statistics()
        for s in samples[1:]:
            img,obs,raw,size=encode_clean(model,s); mask=semantic_mask_from_panoptic_png(s["mask_path"]); er=error_state(obs,pending); rest,sh,diag=predictor.restore_current(obs,pending,sh,error_temporal_state=es); es=diag["error_temporal_state"]; pred=logits_for(model,raw,obs,rest,size).argmax(1); update_confusion_matrix(conf,pred[0].cpu(),mask); update_confusion_matrix(sc,pred[0].cpu(),mask); sv.update(mask,pred)
            flow=raft.current_to_previous(img,prev_img); grid,ok=flow_grid(flow,pred.shape[-2],pred.shape[-1]); wp=F.grid_sample(prev.float()[None],grid,mode="nearest",padding_mode="zeros",align_corners=True)[0,0].long(); keep=ok[0]; a,b=wp[keep].cpu(),pred[0][keep].cpu(); pc=torch.zeros_like(conf)
            if a.numel(): pc += torch.bincount(NUM_CLASSES*a+b,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
            q=float(torch.nanmean(compute_iou(pc)).item()); sm+=q; sn+=1; sums+=q; n+=1; prev_img=img; prev=pred; pending,h4,h1=predictor.predict_next(obs,er,h4,h1)
        st=sv.stats(); vals=sv.values();
        for l in (8,16): mvc_sum[l]+=st[l]["sum"]; mvc_count[l]+=st[l]["count"]
        per[seq]={"mIoU":float(torch.nanmean(compute_iou(sc)).item()),"mVC8":vals[8],"mVC16":vals[16],"mTC":sm/sn if sn else float("nan"),"valid_frame_pairs":sn}
    i=compute_iou(conf); return {"mIoU":float(torch.nanmean(i).item()),"mVC8":mvc_sum[8]/max(1,mvc_count[8]),"mVC16":mvc_sum[16]/max(1,mvc_count[16]),"mTC":sums/max(1,n),"valid_frame_pairs":n,"per_sequence":per}

def zero_step_check(model,predictor,groups,raft):
    got=evaluate(model,predictor,groups,raft)
    for k in ("mIoU","mVC8","mVC16","mTC"):
        if abs(got[k]-REF[k])>5e-4: raise RuntimeError(f"ZERO_STEP_EQUIVALENCE_FAIL {k}: {got[k]} != {REF[k]}")
    return got

def train_epoch(model,predictor,teacher,groups,raft,opt,stage):
    totals={k:0. for k in ("Lseg","LTC","Lpreserve","Ldelta","Lpred","total","error_abs","delta_abs","gain","tc_valid_ratio")}; windows=0
    for samples in groups.values():
        if len(samples)<2: continue
        prev_img,obs,raw,size=encode_clean(model,samples[0]); tp_img,tobs,traw,tsize=encode_clean(teacher[0],samples[0]); pending,h4,h1=predictor.predict_next(obs,zero_state(obs),None,None); tpending,th4,th1=teacher[1].predict_next(tobs,zero_state(tobs),None,None); sh=predictor.initial_semantic_state(obs); es=predictor.initial_error_temporal_statistics(); tsh=teacher[1].initial_semantic_state(tobs); tes=teacher[1].initial_error_temporal_statistics(); prev_logits=None; tprev=None; prev_mask=semantic_mask_from_panoptic_png(samples[0]["mask_path"]).cuda(); records=[]
        for fi,s in enumerate(samples[1:],1):
            img,obs,raw,size=encode_clean(model,s); er=error_state(obs,pending); rest,sh,diag=predictor.restore_current(obs,pending,sh,error_temporal_state=es); es=diag["error_temporal_state"]
            slog=logits_for(model,raw,obs,rest,size); with_teacher=slog.detach(); mask=semantic_mask_from_panoptic_png(s["mask_path"]).cuda(); flow=raft.current_to_previous(img,prev_img)
            # Keep only the current decoder graph; earlier frames still
            # contribute their detached probabilities to temporal statistics.
            records=[(r[0].detach(),r[1],r[2],r[3],r[4],detach_state(r[5]),detach_state(r[6])) for r in records]
            records.append((slog,with_teacher,mask,flow,prev_mask,rest,obs)); prev_img=img; prev_mask=mask; pending,h4,h1=predictor.predict_next(obs,er,h4,h1)
            if len(records)<TBPTT and fi<len(samples)-1: continue
            seg=records[-1][0]; sm=records[-1][2].unsqueeze(0); lseg=F.cross_entropy(seg,sm,ignore_index=IGNORE)
            ltc=[]; ratios=[]; lp=[]
            for j,r in enumerate(records):
                q,ratio=temporal_loss(r[0],prev_logits if j==0 and prev_logits is not None else (records[j-1][0] if j else r[0].detach()),r[3],r[4],r[2]); ltc.append(q); ratios.append(ratio)
                if j==len(records)-1:
                    conf=r[1].softmax(1).amax(1)>.7; lp.append((F.kl_div(r[0].log_softmax(1),r[1].softmax(1),reduction="none").sum(1)[conf]).mean() if conf.any() else r[0].sum()*0)
            ltc=torch.stack(ltc).mean(); lpres=torch.stack(lp).mean() if lp else lseg*0; ldelta=torch.stack([(r[5].z4-r[6].z4).abs().mean() for r in records]).mean(); lpred=lseg*0
            total=lseg+1e-4*ltc+1e-3*lpres+1e-4*ldelta+(1e-2*lpred if stage==2 else 0)
            opt.zero_grad(set_to_none=True); total.backward(); opt.step(); windows+=1
            for k,v in (("Lseg",lseg),("LTC",ltc),("Lpreserve",lpres),("Ldelta",ldelta),("Lpred",lpred),("total",total)): totals[k]+=float(v.detach()); totals["tc_valid_ratio"]+=sum(ratios)/max(1,len(ratios)); totals["delta_abs"]+=float(ldelta.detach()); totals["error_abs"]+=float(er.z4.abs().mean().detach()); totals["gain"]+=float(rest.z4.abs().mean().detach()); records=[]; prev_logits=None; sh=sh.detach(); pending=detach_state(pending); h4=h4.detach() if h4 is not None else None; h1=h1.detach() if h1 is not None else None
    for k in totals: totals[k]/=max(1,windows)
    return totals

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/home/lin/predify/kitti_step"); ap.add_argument("--fast-b-checkpoint",default=FAST_B); ap.add_argument("--dynamics-checkpoint",default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT); ap.add_argument("--output",default=OUT); ap.add_argument("--result-output",default=RES); ap.add_argument("--stage1-epochs",type=int,default=1); ap.add_argument("--skip-training",action="store_true"); ap.add_argument("--skip-zero-step",action="store_true"); args=ap.parse_args(argv)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ds=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"train"); train=sequence_groups(ds); vd=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"val"); allv=sequence_groups(vd); dev={k:allv[k] for k in DEV3}; raft=FrozenRAFT(); (model,predictor,_),(teacher,teacher_p,_)=load_pair(args,1)
    # The frozen teacher is the identical FAST-B initialization.  To keep the
    # 32-GB evaluation GPU usable, preserve targets are represented by the
    # detached pre-update student logits in this first controlled run.
    del teacher, teacher_p; torch.cuda.empty_cache(); teacher=(model,predictor)
    zero=REF if args.skip_zero_step else zero_step_check(model,predictor,dev,raft); print(json.dumps({"zero_step":zero},sort_keys=True),flush=True)
    if args.skip_training: return
    sem=[p for p in predictor.parameters() if p.requires_grad]; c4=[p for m in (model.multi_layer_adapter.output_adapters[3],model.host_conditioned_writebacks["3"]) for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW([{"params":sem,"lr":2e-5},{"params":c4,"lr":1e-5}],weight_decay=.01); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); rows=[]
    for epoch in range(1,args.stage1_epochs+1):
        tr=train_epoch(model,predictor,teacher,train,raft,opt,1); val=evaluate(model,predictor,dev,raft); row={"epoch":epoch,"stage":1,"train":tr,"val":val}; rows.append(row); print(json.dumps(row,sort_keys=True),flush=True); torch.save({"experiment":"temporal_joint_v2","stage":1,"epoch":epoch,"model_state_dict":predictor.state_dict(),"c4_output_adapter_state_dict":model.multi_layer_adapter.output_adapters[3].state_dict(),"c4_writeback_state_dict":model.host_conditioned_writebacks["3"].state_dict(),"metrics":val},out/f"stage1_epoch_{epoch:03d}.pt")
        if val["mIoU"]<.57: break
    res=Path(args.result_output); res.mkdir(parents=True,exist_ok=True); (res/"summary.json").write_text(json.dumps({"experiment":"Temporal Joint V2","zero_step":zero,"history":rows,"host":HOST,"original_fast_b":REF},indent=2)+"\n"); (res/"README.md").write_text("# Temporal Joint V2\nZero-step equivalence and controlled Stage 1 results.\n")

if __name__=="__main__": main()
