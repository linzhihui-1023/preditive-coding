import json, os
from pathlib import Path
import torch
from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, predict_current, sequence_groups, update_dynamic_error
from predify2021.mce_scores.evaluate_kitti_step_error_correction import corrected_host_feature
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_direct_state_correction import direct_posterior
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_persistent_blur_direct_correction import persistent_blur
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import DirectStateCorrection, HostFeature, MultiLayerPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host

PREDICTOR="/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"; WRITEBACK="/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"; NAMES=("static","observation_only","full_error"); PHASES=("clean","transition","persistent")
def load_corrections(path):
    modules=torch.nn.ModuleList([DirectStateCorrection(),DirectStateCorrection()]).cuda(); payload=torch.load(path,map_location="cpu",weights_only=False)
    for position,index in enumerate((0,3)): modules[position].load_state_dict(payload["corrections"][str(index)],strict=True)
    modules.eval(); return modules
def phase(index,total):
    start=total//3
    return "clean" if index<start else ("transition" if index<start+5 else "persistent")
def miou(conf): return float(torch.nanmean(compute_iou(conf)).item())
def main():
    if not torch.cuda.is_available(): raise RuntimeError("Persistent blur evaluation requires CUDA")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); root=Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT","/home/lin/predify/kitti_step")); output=Path(os.environ.get("PREDIFY_PERSISTENT_BLUR_EVAL_OUTPUT_DIR","results/kitti_step_persistent_blur"))
    model=build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model,Path(STATIC_CHECKPOINT_DEFAULT)); adapter=torch.load(ADAPTER_CHECKPOINT_DEFAULT,map_location="cpu",weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"],strict=True); load_writeback_checkpoint(model,Path(WRITEBACK)); model.eval()
    predictor=MultiLayerPredictor().cuda(); predictor.load_state_dict(torch.load(PREDICTOR,map_location="cpu",weights_only=False)["predictor_state_dict"],strict=True); predictor.eval()
    obs=load_corrections(Path(os.environ["PREDIFY_PERSISTENT_BLUR_OBSERVATION_CHECKPOINT"])); full=load_corrections(Path(os.environ["PREDIFY_PERSISTENT_BLUR_FULL_CHECKPOINT"])); zeros=None
    dataset=KITTISTEPSegmentationDataset.from_kitti_step_root(root,"val"); groups=sequence_groups(dataset); overall={n:torch.zeros((19,19),dtype=torch.int64) for n in NAMES}; by_phase={p:{n:torch.zeros((19,19),dtype=torch.int64) for n in NAMES} for p in PHASES}; vc=VideoConsistency(NAMES); persistent_vc=VideoConsistency(NAMES); evaluated=0
    with torch.inference_mode():
      for samples in groups.values():
        vc.reset_sequence(); persistent_vc.reset_sequence(); pp=previous=dynamic=None
        for index,sample in enumerate(samples):
          clean=load_image(sample); degraded=persistent_blur(clean,index,len(samples)); raw=model.extract_backbone_features(degraded); observation=model.encode_backbone_features(raw)
          if previous is None: previous=observation; continue
          if pp is None: pp,previous=previous,observation; continue
          predicted,error=predict_current(predictor,pp,previous,observation); dynamic=update_dynamic_error(error,dynamic); zeros=UnifiedFeatures(*(torch.zeros_like(value) for value in error.as_tuple())); obs_state=direct_posterior(observation,zeros,zeros,obs); full_state=direct_posterior(observation,error,dynamic,full); size=tuple(clean.shape[-2:]); static_host=HostFeature(raw.c4,raw.c1,size)
          hosts={"static":static_host,"observation_only":corrected_host_feature(model,raw,observation,obs_state,size),"full_error":corrected_host_feature(model,raw,observation,full_state,size)}; predictions={name:model.decode_from_host_feature(host).argmax(1).squeeze(0).cpu() for name,host in hosts.items()}; mask=semantic_mask_from_panoptic_png(sample["mask_path"]); current_phase=phase(index,len(samples))
          for name in NAMES: update_confusion_matrix(overall[name],predictions[name],mask); update_confusion_matrix(by_phase[current_phase][name],predictions[name],mask)
          vc.append(mask,predictions)
          if current_phase=="persistent": persistent_vc.append(mask,predictions)
          else: persistent_vc.reset_sequence()
          pp,previous=previous,observation; dynamic=UnifiedFeatures(*(value.detach() for value in dynamic.as_tuple())); evaluated+=1
    overall_mvc=vc.means(); persistent_mvc=persistent_vc.means(); metrics={name:{"overall_miou":miou(overall[name]),"overall_wiou":weighted_iou(overall[name]),"overall_mvc8":overall_mvc[8][name],"overall_mvc16":overall_mvc[16][name],"persistent_mvc8":persistent_mvc[8][name],"persistent_mvc16":persistent_mvc[16][name],"phase_miou":{p:miou(by_phase[p][name]) for p in PHASES}} for name in NAMES}
    clean_static=metrics["static"]["phase_miou"]["clean"]; static_persistent=metrics["static"]["phase_miou"]["persistent"]
    recovery={name:(metrics[name]["phase_miou"]["persistent"]-static_persistent)/(clean_static-static_persistent) for name in ("observation_only","full_error")}; delta=metrics["full_error"]["phase_miou"]["persistent"]-metrics["observation_only"]["phase_miou"]["persistent"]
    decision="GO" if delta>=.005 and metrics["full_error"]["persistent_mvc8"]>=metrics["observation_only"]["persistent_mvc8"]-.005 and metrics["full_error"]["persistent_mvc16"]>=metrics["observation_only"]["persistent_mvc16"]-.005 else ("PARTIAL-GO" if delta>0 else "NO-GO")
    summary={"protocol":{"sequence_count":len(groups),"total_frame_count":len(dataset.samples),"evaluated_frame_count":evaluated},"metrics":metrics,"recovery":recovery,"full_minus_observation_persistent_miou":delta,"PERSISTENT_TEMPORAL_ERROR":decision,"CLEAN_PERFORMANCE_REGRESSION":metrics["full_error"]["phase_miou"]["clean"]<clean_static-.005,"finite":True}; output.mkdir(parents=True,exist_ok=True); (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True),flush=True)
if __name__=="__main__": main()
