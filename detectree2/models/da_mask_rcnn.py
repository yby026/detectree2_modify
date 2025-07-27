import torch, torch.nn as nn, torch.nn.functional as F
from detectron2.modeling import build_model, META_ARCH_REG
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable

# 1) Gradient Reversal
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

# 2) DA Model
@META_ARCH_REG.register()
class DA_MaskRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *, domain_loss_weight, gradrev_lambda, **kwargs):
        super().__init__(**kwargs)
        self.do_da = kwargs["cfg"].MODEL.DOMAIN_ADAPT
        if self.do_da:
            in_ch = self.backbone.output_shape().channels
            self.domain_classifier = nn.Sequential(
                nn.Linear(in_ch, 256), nn.ReLU(),
                nn.Linear(256, kwargs["cfg"].MODEL.DOMAIN_NUM_CLASSES)
            )
            self.domain_loss_weight = domain_loss_weight
            self.gradrev_lambda = gradrev_lambda

    @classmethod
    def from_config(cls, cfg):
        base_args = super().from_config(cfg)
        da_args = {
            "domain_loss_weight": cfg.SOLVER.DOMAIN_LOSS_WEIGHT,
            "gradrev_lambda": cfg.SOLVER.GRADREV_LAMBDA,
        }
        return {**base_args, **da_args, "cfg": cfg}

    def forward(self, batched_inputs):
        # 1) run standard RCNN
        outputs, losses = super().forward(batched_inputs)

        # 2) if DA turned on, add domain loss
        if self.training and self.do_da:
            # Reuse the captured features instead of re-running backbone:
            feat_map = self._last_features["res5"]   # shape [N, C, H, W]
            feat = feat_map.mean([2, 3])            # [N, C]

            # Adversarial GET (GRL)
            feat_rev = GradReverse.apply(feat, self.gradrev_lambda)

            # Domain prediction
            dom_logits = self.domain_classifier(feat_rev)
            dom_labels = torch.tensor(
                [1 if x["dataset_name"] == "TARGET" else 0
                 for x in batched_inputs],
                device=feat.device,
            )
            loss_dom = F.cross_entropy(dom_logits, dom_labels)

            # scale & add
            losses["loss_domain"] = loss_dom * self.domain_loss_weight

        return outputs, losses