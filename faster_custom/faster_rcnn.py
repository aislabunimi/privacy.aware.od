from typing import Any, Callable, List, Optional, Tuple #, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
#from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
#from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor, _validate_trainable_layers
from .generalized_rcnn import GeneralizedRCNN
from .roi_heads import RoIHeads
from .rpn import RegionProposalNetwork, RPNHead
from .transform_faster_removed import GeneralizedRCNNTransformRemoved #custom transform class to avoid transforms and leave them on the dataset

#####
# CODE COPYPASTED FROM TORCHVISION DETECTION, ADDED MY COMMENTS AND PARAMETERS; REMOVED USELESS PARTS FOR ME LIKE KEYPOINTS AND MASKS
#####

__all__ = [
    "FasterRCNN",
    "FasterRCNN_ResNet50_FPN_Weights",
    "FasterRCNN_ResNet50_FPN_V2_Weights",
    "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
]

#default anchor generation function
def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class FasterRCNN(GeneralizedRCNN):
    #input: tensors with shape [C, H, W]
    #during training:  boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``. labels (Int64Tensor[N])
    def __init__(
        self,
        backbone, #backbone used for feature extraction
        num_classes=None, #number of output classes including background
        # transform parameters
        min_size=256, #changed this parameter but I don't use them at all as I skip resize code in Faster RCNN.
        max_size=800,
        image_mean = [0.0, 0.0, 0.0], #Forcing to not applying normalization as default; it's done in the dataset
        image_std = [1.0, 1.0, 1.0],
        #image_mean=None,
        #image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None, 
        rpn_pre_nms_top_n_train=2000, #int, number of proposals to keep before applying nms during training
        rpn_pre_nms_top_n_test=1000, #int, number of proposals to keep before applying nms during testing
        rpn_post_nms_top_n_train=2000, #int, number of proposals to keep after applying nms during training
        rpn_post_nms_top_n_test=1000, #int, number of proposals to keep after applying nms during testing
        rpn_nms_thresh=0.7, #float, threshold NMS used above; number of max remaining proposals is the parameter above
        rpn_fg_iou_thresh=0.7, #float, minimum IoU between anchor and GT box to consider anchor positive
        rpn_bg_iou_thresh=0.3, #float, maximum IoU between anchor and GT box to consider anchor negative
        rpn_batch_size_per_image=256, #int, number of sampled anchors per image during rpn training for loss
        rpn_positive_fraction=0.5, #float, ratio of positive anchors to satisfy by sampler
        rpn_score_thresh=0.0, #float, used in inference; return only proposals above certain thresh
        # Box parameters
        box_roi_pool=None, #MultiScaleRoIAlign
        box_head=None, #TwoMLPHead
        box_predictor=None, #FastRCNNPredictor
        box_score_thresh=0.05, #float, used in inference; return only proposals above certain thresh
        box_nms_thresh=0.5, #float,  used in inference; nms thresh
        box_detections_per_img=100, #int, maximum number of detection for each image.
        box_fg_iou_thresh=0.5, #float, minimum IoU between proposals and GT to consider them as positive.
        box_bg_iou_thresh=0.5, #float, maximum IoU between proposals and GT to consider them as background.
        box_batch_size_per_image=512, #int, number of sampled proposals per image during roi training for loss
        box_positive_fraction=0.25, #float, ratio of positive proposals to satisfy by sampler
        bbox_reg_weights=None, #(Tuple[float, float, float, float])
        #My parameters
        rpn_use_custom_filter_anchors=False, #To activate filter anchors custom
        rpn_n_top_pos_to_keep=1, #How many top positive anchors to keep for each gt
        rpn_n_top_neg_to_keep=5, #How many top negative anchors to keep for each gt
        rpn_n_top_bg_to_keep=1, #How many top background anchors to keep for each gt
        rpn_objectness_bg_thresh=0.0, #Threshold to pick only background anchors with high obj score
	box_use_custom_filter_proposals_objectness=False, #To activate filter proposal custom based on objectness
	box_use_custom_filter_proposals_scores=False, #To activate filter proposal custom based on class score
	box_n_top_pos_to_keep=1, #How many top positive proposals to keep for each gt
	box_n_top_neg_to_keep=5, #How many top negative proposals to keep for each gt
	box_n_top_bg_to_keep=0, #How many top background proposals to keep for each gt
	box_obj_bg_score_thresh=0.9, #Threshold to pick only background anchors with above score
        **kwargs,
    ):

        #SOME CHECKS
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        #additional rpn settings
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        #rpn definition
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
            use_custom_filter_anchors=rpn_use_custom_filter_anchors, #parametri miei
            n_top_pos_to_keep=rpn_n_top_pos_to_keep,
            n_top_neg_to_keep=rpn_n_top_neg_to_keep,
            n_top_bg_to_keep=rpn_n_top_bg_to_keep,
            objectness_bg_thresh=rpn_objectness_bg_thresh,
        )

        #SETTING for faster head
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        #roi heads definition
        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_use_custom_filter_proposals_objectness,
            box_use_custom_filter_proposals_scores, 
            box_n_top_pos_to_keep, 
            box_n_top_neg_to_keep,
            box_n_top_bg_to_keep,
            box_obj_bg_score_thresh,
        )

        #Removing Useless Default transformation. I'll normalize in the dataset as it should be done.
        #if image_mean is None:
        #    image_mean = [0.485, 0.456, 0.406]
        #if image_std is None:
        #    image_std = [0.229, 0.224, 0.225]
        #My custom RCNN transform class that skips resize forcefully
        transform = GeneralizedRCNNTransformRemoved(min_size, max_size, image_mean, image_std, _skip_resize=True)

        super().__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Args:
            input_size (Tuple[int, int, int]): the input size in CHW format.
            conv_layers (list): feature dimensions of each Convolution layer
            fc_layers (list): feature dimensions of each FCN layer
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        in_channels, in_height, in_width = input_size

        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.append(misc_nn_ops.Conv2dNormActivation(previous_channels, current_channels, norm_layer=norm_layer))
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width
        for current_channels in fc_layers:
            blocks.append(nn.Linear(previous_channels, current_channels))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "min_size": (1, 1),
}


class FasterRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 41755286,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 37.0,
                }
            },
            "_ops": 134.38,
            "_file_size": 159.743,
            "_docs": """These weights were produced by following a similar training recipe as on the paper.""",
        },
    )
    DEFAULT = COCO_V1

@register_model()
@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fasterrcnn_resnet50_fpn_custom(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    #Faster CONFIG
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model
