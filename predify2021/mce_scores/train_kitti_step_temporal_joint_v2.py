"""Controlled Temporal Joint V2 training for FAST-B.

This file intentionally leaves the model architecture untouched.  It adds the
V2 training objective around the existing FAST-B predictor and keeps a strict
zero-step checkpoint equivalence check before any backward pass.
"""
import argparse, copy, csv, json, math, random
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
# Dense task/safety supervision: every frame in each TBPTT window contributes
# to the segmentation objective.  The decoder is already evaluated per frame.
SEGMENTATION_SUPERVISION="every_frame"
FAST_B="/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/fast_b_joint_c4_weak_z4/best.pt"
OUT="/home/lin/predify/experiments/kitti_step_temporal_joint_v2"
RES="results/kitti_step_temporal_joint_v2"
DEV3=("0002","0010","0018")
REF={"mIoU":0.5788293035364273,"mVC8":0.945077080607307,"mVC16":0.9421164431668695,"mTC":0.7050058541840838}
HOST={"mIoU":0.5802101104390269,"mVC8":0.9345485965278791,"mVC16":0.9273142366723673,"mTC":0.7009239766436142}
LAMBDA_SEG=3e-4; LAMBDA_TC=1e-4; LAMBDA_PRESERVE=1e-2
MIN_MTC_DELTA=0.002

def stage_c_gate(delta):
    """Return the fixed dev3 Stage C decision in metric (not pp) units."""
    return {
        "mIoU_nonnegative": delta["mIoU"] >= 0.0,
        "mTC_min_delta": MIN_MTC_DELTA,
        "mTC_threshold_passed": delta["mTC"] >= MIN_MTC_DELTA,
        "mVC16_floor": -0.002,
        "mVC16_threshold_passed": delta["mVC16"] >= -0.002,
        "passed": (
            delta["mIoU"] >= 0.0
            and delta["mTC"] >= MIN_MTC_DELTA
            and delta["mVC16"] >= -0.002
        ),
    }

def configure(model,predictor,stage=1):
    model.requires_grad_(False); predictor.requires_grad_(False)
    for m in (predictor.semantic_error_encoder,predictor.semantic_state_cell): m.requires_grad_(True)
    if stage==2:
        for n in ("z4_dyn_recurrent","z4_dyn_delta"): getattr(predictor,n).requires_grad_(True)
    model.eval(); predictor.train()

def load_student(args,stage=1):
    m,_=load_components(STATIC_CHECKPOINT_DEFAULT,ADAPTER_CHECKPOINT_DEFAULT,args.dynamics_checkpoint,WRITEBACK_CHECKPOINT_DEFAULT)
    pld=torch.load(args.fast_b_checkpoint,map_location="cpu",weights_only=False)
    m.multi_layer_adapter.output_adapters[3].load_state_dict(pld["c4_output_adapter_state_dict"])
    m.host_conditioned_writebacks["3"].load_state_dict(pld["c4_writeback_state_dict"])
    p=ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=True).cuda()
    p.load_state_dict(pld["model_state_dict"])
    configure(m,p,stage)
    return m,p,pld

def build_frozen_teacher(model,predictor):
    """Freeze the original FAST-B path without duplicating the frozen host."""
    teacher_predictor=copy.deepcopy(predictor).eval()
    teacher_predictor.requires_grad_(False)
    teacher_c4_adapter=copy.deepcopy(model.multi_layer_adapter.output_adapters[3]).eval()
    teacher_c4_adapter.requires_grad_(False)
    teacher_c4_writeback=copy.deepcopy(model.host_conditioned_writebacks["3"]).eval()
    teacher_c4_writeback.requires_grad_(False)
    return teacher_predictor,teacher_c4_adapter,teacher_c4_writeback

def logits_for(model,raw,obs,rest,size):
    z=zero_state(obs); d=UnifiedFeatures(z.z1,z.z2,z.z3,rest.z4-obs.z4)
    hf=residual_writeback_host_feature(model,raw,d,size)
    if torch.is_grad_enabled():
        return checkpoint(lambda x: model.decode_from_host_feature(HostFeature(x, hf.low_level, hf.output_size)), hf.tensor, use_reentrant=False)
    return model.decode_from_host_feature(hf)

def teacher_logits_for(model,raw,obs,rest,size,teacher_c4_adapter,teacher_c4_writeback):
    """Decode with frozen FAST-B C4 adapter/writeback and shared frozen decoder."""
    delta_z4=rest.z4-obs.z4
    high=raw.c4+teacher_c4_adapter(delta_z4)
    if model.host_conditioned_writeback_enabled:
        high=high+teacher_c4_writeback(raw.c4,delta_z4)
    return model.decode_from_host_feature(HostFeature(high,raw.c1,size))

def safe_loss(student_logits,teacher_logits):
    teacher_prob=teacher_logits.detach().softmax(1)
    confident=teacher_prob.amax(1)>.7
    if not confident.any():
        return student_logits.sum()*0
    per_pixel=F.kl_div(
        student_logits.log_softmax(1),
        teacher_prob,
        reduction="none",
    ).sum(1)
    return per_pixel[confident].mean()

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

def gradient_boundary_test(model,predictor,teacher,samples,raft):
    """Run real isolated backward probes on one ordered eight-frame clip."""
    sem_modules=(predictor.semantic_error_encoder,predictor.semantic_state_cell)
    sem_params=tuple(p for m in sem_modules for p in m.parameters())
    dyn_params=tuple(p for n in ("z4_dyn_recurrent","z4_dyn_delta") for p in getattr(predictor,n).parameters())
    old_flags=[p.requires_grad for p in dyn_params]
    for p in dyn_params: p.requires_grad_(True)
    original_sem=next(iter(predictor.semantic_error_encoder.parameters())).detach().clone()

    def norm(params):
        values=[p.grad.detach().norm() for p in params if p.grad is not None]
        return float(torch.stack(values).norm().item()) if values else 0.0

    def probe(kind):
        model.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True)
        clip=list(samples[:9])
        prev_img,obs,raw,size=encode_clean(model,clip[0])
        prev_mask=semantic_mask_from_panoptic_png(clip[0]["mask_path"]).cuda()
        with torch.no_grad():
            prev_logits=model.decode_from_host_feature(HostFeature(raw.c4,raw.c1,size)).detach()
            pending,h4,h1=predictor.predict_next(obs,zero_state(obs),None,None)
            tpending,th4,th1=teacher[0].predict_next(obs,zero_state(obs),None,None)
        sh=predictor.initial_semantic_state(obs); es=predictor.initial_error_temporal_statistics()
        tsh=teacher[0].initial_semantic_state(obs); tes=teacher[0].initial_error_temporal_statistics()
        segs=[]; tcs=[]; preserves=[]
        for local,sample in enumerate(clip[1:],1):
            image,observation,raw,output_size=encode_clean(model,sample)
            error=error_state(observation,pending)
            detached_error=UnifiedFeatures(*(x.detach() for x in error.as_tuple()))
            restored,sh,diag=predictor.restore_current(observation,pending,sh,prediction_error_override=detached_error,error_temporal_state=es)
            es=diag["error_temporal_state"]
            logits=logits_for(model,raw,observation,restored,output_size)
            mask=semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
            flow=raft.current_to_previous(image,prev_img)
            # Lseg and Lsafe/Lpreserve are dense; every frame is supervised.
            segs.append(F.cross_entropy(logits,mask.unsqueeze(0),ignore_index=IGNORE))
            tc,_=temporal_loss(logits,prev_logits,flow,prev_mask,mask); tcs.append(tc)
            with torch.no_grad():
                teacher_error=error_state(observation,tpending)
                teacher_restored,tsh,teacher_diag=teacher[0].restore_current(observation,tpending,tsh,error_temporal_state=tes)
                tes=teacher_diag["error_temporal_state"]
                teacher_logits=teacher_logits_for(model,raw,observation,teacher_restored,output_size,teacher[1],teacher[2])
                tpending,th4,th1=teacher[0].predict_next(observation,teacher_error,th4,th1)
            preserves.append(safe_loss(logits,teacher_logits))
            prev_img=image; prev_mask=mask; prev_logits=logits.detach()
            pending,h4,h1=predictor.predict_next(observation,error,h4,h1)
        losses={"Lseg":torch.stack(segs).mean(),"LTC":torch.stack(tcs).mean(),"Lpreserve":torch.stack(preserves).mean()}
        if kind=="Lpreserve" and float(losses[kind].detach())==0.0:
            with torch.no_grad():
                next(iter(predictor.semantic_error_encoder.parameters())).add_(1e-4)
            # Re-run the preserve probe with a non-identical student; restore
            # immediately after measuring the real backward path.
            model.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True)
            return probe(kind)
        losses[kind].backward()
        result={"semantic_error_encoder":norm(tuple(predictor.semantic_error_encoder.parameters())),"semantic_state":norm(tuple(predictor.semantic_state_cell.parameters())),"dynamics_z4":norm(dyn_params),"loss":float(losses[kind].detach())}
        model.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True)
        return result

    results={k:probe(k) for k in ("Lseg","LTC","Lpreserve")}
    with torch.no_grad(): next(iter(predictor.semantic_error_encoder.parameters())).copy_(original_sem)
    for p,flag in zip(dyn_params,old_flags): p.requires_grad_(flag)
    if any(results[k]["semantic_error_encoder"]<=0 or results[k]["semantic_state"]<=0 or results[k]["dynamics_z4"]!=0.0 for k in results):
        raise RuntimeError(f"GRADIENT_ROLE_SEPARATION_FAIL: {results}")
    return results

def train_epoch(model,predictor,teacher,groups,raft,opt,stage):
    teacher_predictor,teacher_c4_adapter,teacher_c4_writeback=teacher
    totals={k:0. for k in ("Lseg","LTC","Lpreserve","Ldelta","Lpred","total","error_abs","delta_abs","gain","tc_valid_ratio")}
    windows=0
    for samples in groups.values():
        if len(samples)<2:
            continue
        prev_img,obs,raw,size=encode_clean(model,samples[0])
        prev_mask=semantic_mask_from_panoptic_png(samples[0]["mask_path"]).cuda()
        with torch.no_grad():
            prev_student_logits=model.decode_from_host_feature(HostFeature(raw.c4,raw.c1,size)).detach()
            pending,h4,h1=predictor.predict_next(obs,zero_state(obs),None,None)
            tpending,th4,th1=teacher_predictor.predict_next(obs,zero_state(obs),None,None)
        sh=predictor.initial_semantic_state(obs)
        es=predictor.initial_error_temporal_statistics()
        tsh=teacher_predictor.initial_semantic_state(obs)
        tes=teacher_predictor.initial_error_temporal_statistics()

        seg_losses=[]; tc_losses=[]; preserve_losses=[]; delta_losses=[]; tc_ratios=[]; error_values=[]
        for fi,s in enumerate(samples[1:],1):
            img,obs,raw,size=encode_clean(model,s)
            er=error_state(obs,pending)
            semantic_error=UnifiedFeatures(*(x.detach() for x in er.as_tuple()))
            rest,sh,diag=predictor.restore_current(obs,pending,sh,prediction_error_override=semantic_error,error_temporal_state=es)
            es=diag["error_temporal_state"]
            slog=logits_for(model,raw,obs,rest,size)
            mask=semantic_mask_from_panoptic_png(s["mask_path"]).cuda()
            flow=raft.current_to_previous(img,prev_img)

            # True frozen FAST-B teacher: independent recurrent state and frozen
            # semantic/C4 correction weights, while sharing only the frozen host.
            with torch.no_grad():
                ter=error_state(obs,tpending)
                trest,tsh,tdiag=teacher_predictor.restore_current(
                    obs,tpending,tsh,error_temporal_state=tes
                )
                tes=tdiag["error_temporal_state"]
                tlog=teacher_logits_for(
                    model,raw,obs,trest,size,teacher_c4_adapter,teacher_c4_writeback
                )
                tpending,th4,th1=teacher_predictor.predict_next(obs,ter,th4,th1)

            local_position=((fi-1)%TBPTT)+1
            # Protect every frame against changing a Host-correct prediction;
            # this is intentionally dense rather than sparse 4/16 supervision.
            seg_losses.append(F.cross_entropy(slog,mask.unsqueeze(0),ignore_index=IGNORE))

            # Always pair the current frame with the actual immediately previous
            # student prediction.  Across TBPTT boundaries it is detached, not reset.
            q,ratio=temporal_loss(slog,prev_student_logits,flow,prev_mask,mask)
            tc_losses.append(q)
            tc_ratios.append(ratio)
            preserve_losses.append(safe_loss(slog,tlog))
            delta_losses.append((rest.z4-obs.z4).abs().mean())
            error_values.append(er.z4.abs().mean())

            prev_student_logits=slog.detach()
            prev_img=img
            prev_mask=mask
            pending,h4,h1=predictor.predict_next(obs,er,h4,h1)

            window_end=(local_position==TBPTT or fi==len(samples)-1)
            if not window_end:
                continue

            lseg=torch.stack(seg_losses).mean() if seg_losses else slog.sum()*0
            ltc=torch.stack(tc_losses).mean()
            lpres=torch.stack(preserve_losses).mean()
            ldelta=torch.stack(delta_losses).mean()
            lpred=lseg*0
            total=LAMBDA_SEG*lseg+LAMBDA_TC*ltc+LAMBDA_PRESERVE*lpres+(1e-2*lpred if stage==2 else 0)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            windows+=1

            for k,v in (("Lseg",lseg),("LTC",ltc),("Lpreserve",lpres),("Ldelta",ldelta),("Lpred",lpred),("total",total)):
                totals[k]+=float(v.detach())
            totals["tc_valid_ratio"]+=sum(tc_ratios)/max(1,len(tc_ratios))
            totals["delta_abs"]+=float(ldelta.detach())
            totals["error_abs"]+=float(torch.stack(error_values).mean().detach())
            totals["gain"]+=float(rest.z4.abs().mean().detach())

            seg_losses=[]; tc_losses=[]; preserve_losses=[]; delta_losses=[]; tc_ratios=[]; error_values=[]
            sh=sh.detach()
            es=es.detach() if es is not None else None
            pending=detach_state(pending)
            h4=h4.detach() if h4 is not None else None
            h1=h1.detach() if h1 is not None else None

    for k in totals:
        totals[k]/=max(1,windows)
    return totals

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/home/lin/predify/kitti_step"); ap.add_argument("--fast-b-checkpoint",default=FAST_B); ap.add_argument("--dynamics-checkpoint",default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT); ap.add_argument("--output",default=OUT); ap.add_argument("--result-output",default=RES); ap.add_argument("--stage1-epochs",type=int,default=1); ap.add_argument("--skip-training",action="store_true"); ap.add_argument("--skip-zero-step",action="store_true"); args=ap.parse_args(argv)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ds=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"train"); train=sequence_groups(ds); vd=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"val"); allv=sequence_groups(vd); dev={k:allv[k] for k in DEV3}; raft=FrozenRAFT(); model,predictor,_=load_student(args,1)
    args.lambda_tc = LAMBDA_TC
    teacher=build_frozen_teacher(model,predictor)
    boundary=gradient_boundary_test(model,predictor,teacher,next(iter(train.values())),raft)
    zero=REF if args.skip_zero_step else zero_step_check(model,predictor,dev,raft); print(json.dumps({"gradient_boundary":boundary,"zero_step":zero},sort_keys=True),flush=True)
    if args.skip_training: return
    sem=[p for m in (predictor.semantic_error_encoder,predictor.semantic_state_cell) for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW([{"params":sem,"lr":1e-5}],weight_decay=.01); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); rows=[]; best=None
    for epoch in range(1,args.stage1_epochs+1):
        tr=train_epoch(model,predictor,teacher,train,raft,opt,1); val=evaluate(model,predictor,dev,raft)
        delta={key: val[key]-REF[key] for key in ("mIoU","mTC","mVC16")}; gate=stage_c_gate(delta)
        row={"epoch":epoch,"stage":1,"train":tr,"val":val,"stage_c_delta":delta,"stage_c_gate":gate}; rows.append(row); print(json.dumps(row,sort_keys=True),flush=True)
        payload={"experiment":"temporal_joint_v2","stage":1,"epoch":epoch,"model_state_dict":predictor.state_dict(),"c4_output_adapter_state_dict":model.multi_layer_adapter.output_adapters[3].state_dict(),"c4_writeback_state_dict":model.host_conditioned_writebacks["3"].state_dict(),"metrics":val}
        torch.save(payload,out/f"stage1_epoch_{epoch:03d}.pt")
        if gate["passed"] and (best is None or val["mIoU"] > best["val"]["mIoU"]):
            best={"epoch":epoch,"val":val,"delta":delta,"gate":gate}; torch.save(payload,out/"best.pt")
        if val["mIoU"]<.57: break
    full9=None
    if best is not None:
        best_payload=torch.load(out/"best.pt",map_location="cuda",weights_only=False)
        predictor.load_state_dict(best_payload["model_state_dict"],strict=True)
        model.multi_layer_adapter.output_adapters[3].load_state_dict(best_payload["c4_output_adapter_state_dict"],strict=True)
        model.host_conditioned_writebacks["3"].load_state_dict(best_payload["c4_writeback_state_dict"],strict=True)
        full9=evaluate(model,predictor,allv,raft)
        best["full9"]=full9
    res=Path(args.result_output); res.mkdir(parents=True,exist_ok=True); summary={"experiment":"Temporal Joint V2","zero_step":zero,"history":rows,"host":HOST,"original_fast_b":REF,"segmentation_supervision":SEGMENTATION_SUPERVISION,"safe_loss_supervision":"every_frame","min_mtc_delta":MIN_MTC_DELTA,"best":best,"full9_evaluation_after_gate":True}; (res/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (res/"README.md").write_text("# Temporal Joint V2\nDense Lseg/Lsafe supervision, strict Stage C mTC gate, and Full9 evaluation after a passing gate.\n")

if __name__=="__main__": main()
