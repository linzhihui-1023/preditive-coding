import json, os, random
from pathlib import Path
import torch
from torch.nn import functional as F
from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import encode_image, load_image, predict_current, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_error_correction import corrected_host_feature
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import MultiLayerPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou

ROOT=Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT","/home/lin/predify/kitti_step")); OUT=Path(os.environ.get("PREDIFY_LINE_A_OUTPUT_DIR","results/kitti_step_line_a")); SEED=0
PREDICTOR="/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
WRITEBACK="/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"
def metrics(conf, vc):
    return {"miou":float(torch.nanmean(compute_iou(conf)).item()),"wiou":weighted_iou(conf),"mvc":vc.means()}
def main():
    if not torch.cuda.is_available(): raise RuntimeError("Line A requires CUDA")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model=build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model,Path(os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT",STATIC_CHECKPOINT_DEFAULT)))
    adapter=torch.load(os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT",ADAPTER_CHECKPOINT_DEFAULT),map_location="cpu",weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"]); load_writeback_checkpoint(model,Path(WRITEBACK))
    predictor=MultiLayerPredictor().cuda(); payload=torch.load(PREDICTOR,map_location="cpu",weights_only=False); predictor.load_state_dict(payload["predictor_state_dict"]); model.eval(); predictor.eval()
    ds=KITTISTEPSegmentationDataset.from_kitti_step_root(ROOT,"val"); groups=sequence_groups(ds); names=("static","persistence","predictor"); conf={n:torch.zeros((19,19),dtype=torch.int64) for n in names}; vc=VideoConsistency(); state={"persistence":[],"predictor":[]}; cos=[]; n=0
    with torch.inference_mode():
      for samples in groups.values():
        pp=p=None
        for sample in samples:
          image=load_image(sample); raw=model.extract_backbone_features(image); obs=model.encode_backbone_features(raw)
          if p is None: p=obs; continue
          if pp is None: pp,p=p,obs; continue
          pred,_=predict_current(predictor,pp,p,obs); targets={"persistence":p,"predictor":pred}; delta=UnifiedFeatures(*(a-b for a,b in zip(pred.as_tuple(),p.as_tuple()))); actual=UnifiedFeatures(*(a-b for a,b in zip(obs.as_tuple(),p.as_tuple())))
          cos.append(F.cosine_similarity(delta.z1.flatten(1),actual.z1.flatten(1)).mean().item()); mask=semantic_mask_from_panoptic_png(sample["mask_path"]); predictions={"static":model(image).argmax(1).squeeze(0).cpu()};
          for key,target in targets.items(): predictions[key]=model.decode_from_host_feature(corrected_host_feature(model,raw,obs,target,tuple(image.shape[-2:]))).argmax(1).squeeze(0).cpu(); state[key].append(sum(F.mse_loss(target.z1,obs.z1).item() if key=="persistence" else F.mse_loss(target.z1,obs.z1).item() for _ in [0]))
          for key,value in predictions.items(): update_confusion_matrix(conf[key],value,mask)
          vc.append(mask,predictions); pp,p=p,obs; n+=1
    result={"protocol":{"split":"val","sequence_count":len(groups),"total_frame_count":len(ds.samples),"evaluated_frame_count":n},"metrics":{k:metrics(v,vc) for k,v in conf.items()},"state_mse":{"persistence":sum(state["persistence"])/n,"predictor":sum(state["predictor"])/n},"predictor_delta_cosine":sum(cos)/len(cos),"historical_clean_static_miou":0.6552125562}
    result["PREDICTOR_TEMPORAL_SIGNAL"]="GO" if result["state_mse"]["predictor"]<result["state_mse"]["persistence"] and result["metrics"]["predictor"]["miou"]>result["metrics"]["persistence"]["miou"] else "NO-GO"; OUT.mkdir(parents=True,exist_ok=True); (OUT/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
