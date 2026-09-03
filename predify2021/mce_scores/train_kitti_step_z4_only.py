"""Z4-only predictive-coding mainline.

Stage P is deliberately isolated: only the existing Z4 recurrent predictor
and Z4 delta head are optimized.  Z1 modules remain in old checkpoints for
compatibility but are never called by this entry point.
"""
import argparse, json, random
from pathlib import Path
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT, load_components,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor

FAST_B_DEFAULT="/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/fast_b_joint_c4_weak_z4/best.pt"
OUTPUT_DEFAULT="/home/lin/predify/experiments/kitti_step_z4_only_stage_p"
RESULT_DEFAULT="results/kitti_step_z4_only_stage_p"
TBPTT=16; LR=1e-5; WEIGHT_DECAY=.01; Z1_WEIGHT=.0

def load_fast_b(args):
    model,_=load_components(STATIC_CHECKPOINT_DEFAULT,ADAPTER_CHECKPOINT_DEFAULT,args.dynamics_checkpoint,WRITEBACK_CHECKPOINT_DEFAULT)
    payload=torch.load(args.fast_b_checkpoint,map_location="cpu",weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(payload["c4_output_adapter_state_dict"],strict=True)
    model.host_conditioned_writebacks["3"].load_state_dict(payload["c4_writeback_state_dict"],strict=True)
    predictor=ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=True).cuda()
    predictor.load_state_dict(payload["model_state_dict"],strict=True)
    model.requires_grad_(False); predictor.requires_grad_(False)
    for name in ("z4_dyn_recurrent","z4_dyn_delta"): getattr(predictor,name).requires_grad_(True)
    model.eval(); predictor.train()
    return model,predictor,payload

def z4_predict_next(predictor,observation_z4,error_z4,hidden=None):
    hidden=predictor.z4_dyn_recurrent(torch.cat((observation_z4,error_z4),dim=1),hidden)
    return observation_z4+predictor.z4_dyn_delta(hidden),hidden

def assert_z4_only_contract(model,predictor):
    z4=tuple(p for p in predictor.z4_dyn_recurrent.parameters())+tuple(predictor.z4_dyn_delta.parameters())
    if not z4 or not all(p.requires_grad for p in z4): raise RuntimeError("Z4 predictor is not trainable")
    for name in ("z1_dyn_recurrent","z1_dyn_delta"):
        if any(p.requires_grad for p in getattr(predictor,name).parameters()): raise RuntimeError("Z1 predictor must be frozen")
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Host must remain frozen")

def train_sequence(model,predictor,samples,optimizer):
    if len(samples)<2: return {"windows":0,"pred_loss":0.,"pred_mse":0.,"copy_mse":0.,"ratio":float("nan")}
    hidden=None; pending=None; losses=[]; pred_mse=[]; copy_mse=[]; totals={"windows":0,"pred_loss":0.,"pred_mse":0.,"copy_mse":0.,"ratio":float("nan")}
    with torch.no_grad():
        image=load_image(samples[0]); raw=model.extract_backbone_features(image); obs=model.encode_backbone_features(raw); previous=obs.z4.detach()
    pending=previous
    for step,sample in enumerate(samples[1:],1):
        with torch.no_grad():
            image=load_image(sample); raw=model.extract_backbone_features(image); obs=model.encode_backbone_features(raw)
        error=obs.z4-pending
        prediction,hidden=z4_predict_next(predictor,previous,error,hidden)
        losses.append(F.smooth_l1_loss(prediction,obs.z4.detach()))
        pred_mse.append(F.mse_loss(prediction,obs.z4.detach())); copy_mse.append(F.mse_loss(previous,obs.z4.detach()))
        pending=prediction.detach(); previous=obs.z4.detach()
        if len(losses)<TBPTT and step < len(samples)-1: continue
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite Z4 prediction loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); totals["windows"]+=1
        totals["pred_loss"]+=float(loss.detach()); totals["pred_mse"]+=float(torch.stack(pred_mse).mean().detach()); totals["copy_mse"]+=float(torch.stack(copy_mse).mean().detach())
        hidden=hidden.detach(); losses=[]; pred_mse=[]; copy_mse=[]
    if totals["windows"]:
        for key in ("pred_loss","pred_mse","copy_mse"): totals[key]/=totals["windows"]
        totals["ratio"]=totals["pred_mse"]/max(totals["copy_mse"],1e-12)
    return totals

@torch.no_grad()
def measure_sequence(model,predictor,samples):
    if len(samples)<2: return {"pred_mse":float("nan"),"copy_mse":float("nan"),"ratio":float("nan"),"frames":0}
    hidden=None; previous=None; pending=None; pm=[]; cm=[]
    for index,sample in enumerate(samples):
        image=load_image(sample); raw=model.extract_backbone_features(image); obs=model.encode_backbone_features(raw)
        if index==0: previous=obs.z4; pending=previous; continue
        error=obs.z4-pending; prediction,hidden=z4_predict_next(predictor,previous,error,hidden)
        pm.append(F.mse_loss(prediction,obs.z4).item()); cm.append(F.mse_loss(previous,obs.z4).item()); previous=obs.z4; pending=prediction
    p=sum(pm)/len(pm); c=sum(cm)/len(cm); return {"pred_mse":p,"copy_mse":c,"ratio":p/max(c,1e-12),"frames":len(pm)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/home/lin/predify/kitti_step"); ap.add_argument("--fast-b-checkpoint",default=FAST_B_DEFAULT); ap.add_argument("--dynamics-checkpoint",default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT); ap.add_argument("--output",default=OUTPUT_DEFAULT); ap.add_argument("--result-output",default=RESULT_DEFAULT); ap.add_argument("--epochs",type=int,default=1); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model,predictor,source=load_fast_b(args); assert_z4_only_contract(model,predictor)
    dataset=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"train"); groups=sequence_groups(dataset)
    val_dataset=KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root),"val"); val_groups=sequence_groups(val_dataset)
    optimizer=torch.optim.AdamW([p for n in ("z4_dyn_recurrent","z4_dyn_delta") for p in getattr(predictor,n).parameters()],lr=LR,weight_decay=WEIGHT_DECAY)
    history=[]
    for epoch in range(1,args.epochs+1):
        train={"pred_loss":0.,"pred_mse":0.,"copy_mse":0.,"windows":0}
        for samples in groups.values():
            row=train_sequence(model,predictor,samples,optimizer)
            for key in train: train[key]+=row[key]
        eval_rows={seq:measure_sequence(model,predictor,samples) for seq,samples in val_groups.items()}
        better=sum(row["ratio"]<1.0 for row in eval_rows.values()); mean_pred=sum(row["pred_mse"] for row in eval_rows.values())/len(eval_rows); mean_copy=sum(row["copy_mse"] for row in eval_rows.values())/len(eval_rows)
        record={"epoch":epoch,"stage":"P","train":train,"full9_prediction":{"mean_pred_mse":mean_pred,"mean_copy_mse":mean_copy,"ratio":mean_pred/max(mean_copy,1e-12),"sequences_better_than_persistence":better,"sequence_count":len(eval_rows),"per_sequence":eval_rows}}; history.append(record); print(json.dumps(record,sort_keys=True),flush=True)
        out=Path(args.output); out.mkdir(parents=True,exist_ok=True); torch.save({"experiment":"z4_only_stage_p","epoch":epoch,"model_state_dict":predictor.state_dict(),"metrics":record},out/f"epoch_{epoch:03d}.pt")
    result={"experiment":"Predify Z4-only Stage P","source_fast_b_checkpoint":args.fast_b_checkpoint,"tbptt":TBPTT,"lr":LR,"trainable_modules":["z4_dyn_recurrent","z4_dyn_delta"],"unused_formal_modules":["all Z1 paths","Dynamics Error"],"history":history,"gate":{"ratio_lt_1":history[-1]["full9_prediction"]["ratio"]<1.0,"sequences_better_than_persistence":history[-1]["full9_prediction"]["sequences_better_than_persistence"],"required_sequences":6}}
    result_dir=Path(args.result_output); result_dir.mkdir(parents=True,exist_ok=True); (result_dir/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); (result_dir/"README.md").write_text("# Z4-only Stage P\nStage P trains only the existing Z4 predictor; Stage C/W/J are not automatic.\n")

if __name__=="__main__": main()
