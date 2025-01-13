from mmdeploy.apis import inference_model

model_cfg = 'mmpretrain-mmcls-0.x/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'mmdeploy-0.x/configs/mmcls/classification_tensorrt_static-224x224.py'
backend_files = ['work_dir/trt/resnet/end2end.engine']
img = 'mmpretrain-mmcls-0.x/demo/demo.JPEG'
device = 'cuda'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
print(result)