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
from torchvision.models.detection.transform import GeneralizedRCNNTransform

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

#Funzione che si occupa di generare le ancore di diversa dimensione e ratio
def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class FasterRCNN(GeneralizedRCNN):
    """
    Input del modello: lista di tensori con shape [C, H, W] uno per ogni img nella range 0-1.
    Durante il training, il modello si aspetta i tensori e i targets (lista di dizionari) che contiene:
     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
     - labels (Int64Tensor[N]): la class label per ogni bbox
    Il modello ritorna un Dict[Tensor] che contiene le loss di classificazione e regressione della RPN e R-CNN.
    Durante il testing, il modello si aspetta solo i tensori e ritorna una List[Dict[Tensor]] di predizioni, una per ogni img di input. I campi di Dict sono:
     - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
     - labels (Int64Tensor[N]): le labels predette per ogni img
     - scores (Tensor[N]): gli score di ogni predizione
    """
    #Costruttore della Faster
    def __init__(
        self,
        backbone, #la backbone usata per estrarre le feature
        num_classes=None, #numero di classi in output dal modello, incluso il background. Usando il box predictor rimarrà a none
        # transform parameters # NON interessanti per noi
        min_size=800, #int, dimensione minima dell'immagine da fare il rescaling prima di darlo alla backbone
        max_size=1333, #int, come sopra ma dimensione massima
        image_mean=None, #Tuple[float, float, float], mean per normalizzare l'input
        image_std=None, #Tuple[float, float, float], std per normalizzare l'input
        # RPN parameters #Quelli più importanti
        rpn_anchor_generator=None, #AnchorGenerator, modulo per generare le anchors da insieme di feature maps.
        rpn_head=None, #nn.Module, la head del RPN network per calcolare l'objectness score e il regression bbox score
        rpn_pre_nms_top_n_train=2000, #int, numero di proposals da tenere PRIMA di applicare NMS durante il training
        rpn_pre_nms_top_n_test=1000, #int, come sopra ma per il testing
        rpn_post_nms_top_n_train=2000, #int, numero di proposals da tenere DOPO aver applicato NMS durante il training
        rpn_post_nms_top_n_test=1000, #int, come sopra ma per il testing
        rpn_nms_thresh=0.7, #float, threshold NMS usata per il postprocessing delle proposals delle RPN
        rpn_fg_iou_thresh=0.7, #float, minima IoU tra l'anchor e la GT box per poterle considerare come positive (ovvero contengono un oggetto) durante il training. In questo modo durante il training vado a scartare tutte quelle anchor che ho generato che non soddisfano questo overlap. Il modello imparerà a modificare l'anchor in modo che assomigli di ratio, dimensione e offset alla GT box. Otterrò così bbox raffinate. 
        rpn_bg_iou_thresh=0.3, #float, come sopra, solo che qui è la massima IoU tra l'anchor e la GT box da considerare negative (ovvero non contenenti un oggetto ma solo lo sfondo) durante il training.
        rpn_batch_size_per_image=256, #int, numero di anchors boxes che sono campionate, scelte (sampled) per computare la loss durante il training del RPN. Queste sono le anchors boxes che vengono tenute
        rpn_positive_fraction=0.5, #float, rapporto di anchors positive in un mini-batch durante training of the RPN
        rpn_score_thresh=0.0, #float, usato in inferenze; ritorna solo le proposals con uno score sopra la thresh definita da questo parametro
        # Box parameters
        box_roi_pool=None, #MultiScaleRoIAlign, modulo che fa il crop e resize delle feature maps nelle posizioni indicate dalle bbox
        box_head=None, #nn.Module, modulo che prende le cropped feature maps come input
        box_predictor=None, #nn.Module, modulo che prende l'output di box_head e ritorna le classification logits e box regression deltas.
        box_score_thresh=0.05, #float, durante inferenza ritorna solo le proposals sopra a tale classification score
        box_nms_thresh=0.5, #float, NMS threshold per le predizioni della testa usata in inferenza.
        box_detections_per_img=100, #int, numero massimo di detections per immagine per tutte le classi.
        box_fg_iou_thresh=0.5, #float, minima IoU tra le proposals e la GT box per poterle considerare come positive (ovvero contengono un oggetto) durante il training della classification head.
        box_bg_iou_thresh=0.5, #float, come sopra solo che qui è la massima IoU tra le proposals e la GT box da considerare negative durante il training della classification head.
        box_batch_size_per_image=512, #int, numero di proposals che sono campionate (sampled) durante il training della classification head.
        box_positive_fraction=0.25, #float, rapporto di proposals positive in un mini-batch durante training of the RPN
        bbox_reg_weights=None, #(Tuple[float, float, float, float]), pesi per l'encoding/decoding delle bbox
        use_custom_filter_proposals=False, #per usare il filter proposal custom. I parametri sotto vengono usati solo se questa variabile è a true
        rpn_n_top_iou_to_keep=1, #quante proposal con top iou da tenere
        rpn_iou_neg_thresh=0.5, #thresh per considerare negative delle prop
        rpn_n_top_neg_to_keep=100, #quante proposal negative con iou al di sotto del thresh imposto sopra da tenere
        **kwargs,
    ):

        #CONTROLLI VARI DI INIZIALIZZAZIONE DELLA FASTER
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

        #SETTING vari dell'rpn
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        #Creazione oggetto RPN con i parametri passati
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
            use_custom_filter_proposals=use_custom_filter_proposals, #parametri miei
            n_top_iou_to_keep=rpn_n_top_iou_to_keep,
            iou_neg_thresh=rpn_iou_neg_thresh,
            n_top_neg_to_keep=rpn_n_top_neg_to_keep,
        )

        #SETTING vari per l'head della faster
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        #Creazione dell'head con i parametri passati
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
        )

        #Trasformazioni di default se non applicate prima
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        #init del padre Generalized RCNN
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
def fasterrcnn_resnet50_fpn_modificata(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """ FUNZIONE per creare un'istanza della Faster.
    Args: weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights`, optional). I weights pretraining.
    progress (bool, optional): serve solo per mostrare la progress bar del download dei pesi.
    num_classes (int, optional): numero di classi di output del modello incluso background
    weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional), pesi per la backbone
    trainable_backbone_layers (int, optional): numero di layers trainabili. I valori vanno da 0 a 5 (tutti i layers). Default è 3
    **kwargs: i parametri della classe FasterRCNN all'inizio del file
    """
    #CONFIGURAZIONE VARIA della faster
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
