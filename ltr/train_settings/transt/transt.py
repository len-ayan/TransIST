import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt as transt_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import numpy as np

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 8
    settings.num_workers = 16
    settings.multi_gpu = False
    settings.print_interval = 500
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    # Train datasets
    lasot_train = Lasot("/root/autodl-tmp/GOT-10k",split='train')
    # lasot_train = Lasot(settings.env.lasot_dir)
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    #got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.TransTSampler([lasot_train], [1],
                                samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)
    # dataset_train = sampler.TransTSampler([lasot_train, got10k_train, coco_train, trackingnet_train], [1,1,1,1],
    #                             samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)


    # Create network and actor
    model = transt_models.transt_resnet50(settings)
    # #预训练
    # model_path='/root/autodl-tmp/TransT/pytracking/networks/transt1.pth'
    #
    # checkpoint = torch.load(model_path)
    # if 'net' in checkpoint:
    #     state_dict = checkpoint['net']
    #     # 现在您可以尝试将state_dict加载到您的模型中
    #     model.load_state_dict(state_dict)
    #   #  model.load_state_dict(state_dict,strict=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # unfreeze_layers(model)
    # # 第一步：读取当前模型参数
    # model_dict = model.state_dict()
    # # 第二步：读取预训练模型
    # pretrained_dict = torch.load(model_path, map_location='cuda')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # # 第三步：使用预训练的模型更新当前模型参数
    # model_dict.update(pretrained_dict)
    # # 第四步：加载模型参数
    # model.load_state_dict(model_dict)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(180, load_latest=True,fail_safe=True)
def unfreeze_layers(model):
    for name, child in model.named_children():
        if name in ['bbox_embed','class_embed']:  # 指定要冻结的层
            for param in child.parameters():
                param.requires_grad = True
                # print(name)
        else:
            #break  # 一旦到达不冻结的层，停止循环
            continue #如果你想冻结 layer1 之后的某些层，这里应该用 continue