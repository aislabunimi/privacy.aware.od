from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation

from torchvision.models.detection import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList

#####
# CODE COPYPASTED FROM TORCHVISION DETECTION, ADDED MY COMMENTS AND PARAMETERS; REMOVED USELESS PARTS FOR ME LIKE KEYPOINTS AND MASKS
# ADDED CUSTOM METHOD FOR PROPOSALS: filter_proposals_custom
#####

class RPNHead(nn.Module):
    """ Head dell'RPN per la classificazione e regressione delle bbox
    Args: in_channels (int): delle input feature, num_anchors (int): qui è 3. credo faccia riferimento al numero di livelli delle feature
    conv_depth (int, optional): numero di convoluzioni
    """
    _version = 2

    #Costruttore della classe RPNHead
    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    #FUNZIONE per Caricare lo stato del Modello da un file 
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    #FORWARD. Dò la probabilità di classificazione e le bbox regressors
    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

# Funzione per manipolare i tensori
def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

# Funzione per calcolare le bbox e class score
# Per farlo è necessario concatenare tutti i feature level, che è fatto qui sotto
def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """ Classe del Region Proposal Network (RPN). Gli argomenti sono gli stessi RPN parameters
    del file faster_rcnn.py, fare riferimento a lì per la spiegazione. Args:
    anchor_generator (AnchorGenerator), head (nn.Module), fg_iou_thresh (float),
    bg_iou_thresh (float), batch_size_per_image (int), positive_fraction (float),
    pre_nms_top_n (Dict[str, int]): dizionario con i due pre_nms_top_n della faster, il primo campo per training e il secondo per test
    post_nms_top_n (Dict[str, int]): come sopra ma per il post nms; 
    nms_thresh (float)
    """
    
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    #Costruttore dell'RPN. Usa i parametri passati in fase di costruzione della faster
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
        use_custom_filter_anchors: bool=False,
        n_top_pos_to_keep: int=1,
        #iou_neg_thresh: float=0.5,
        n_top_neg_to_keep: int=8,
        n_top_bg_to_keep: int=0,
        absolute_bg_score_thresh: float = 0.75,
        objectness_bg_thresh: float = 0.00,
        use_not_overlapping_proposals: bool=False,
        overlapping_prop_thresh: float=0.6,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        
        #se voglio usare il mio filter proposals sarà a true
        
        self.use_custom_filter_anchors = use_custom_filter_anchors
        self.n_top_pos_to_keep = n_top_pos_to_keep
        #self.iou_neg_thresh = iou_neg_thresh
        self.n_top_neg_to_keep = n_top_neg_to_keep
        self.n_top_bg_to_keep = n_top_bg_to_keep           
        self.absolute_bg_score_thresh = absolute_bg_score_thresh
        self.use_not_overlapping_proposals = use_not_overlapping_proposals
        self.overlapping_prop_thresh = overlapping_prop_thresh
        self.objectness_bg_thresh = objectness_bg_thresh

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction) # This class samples batches, ensuring that they contain a fixed proportion of positives
        # estrae casualmente un numero di sample positivi (con oggetto) e negativi (senza) che rispettano la proporzione
        
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3
        

    #Due funzioni solo per passare il parametro giusto
    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    #Funzione che per ogni anchor boxes, assegna -1, 0 o 1 per sapere se è una anchor da ignorare, negativa o positiva
    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        #2 liste di tensori. Lista perché ogni tensore che contiene la label intesa come bg, fg o niente.
        labels = []
        matched_gt_boxes = [] #questo è un tensore contenente le gt boxes che le anchors matchano
        #in particolare i tensori saranno lunghi quante sono le anchors delle img, es 185460             
        #COME FUNZIONA: a ciascuna ancora viene associato un gt. Rispetto a tale gt, verrà poi calcolato il box regression. Quindi se ho 100000 ancore, per ognuna di esse avrò il gt
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            #Se non c'è nessuna GT box nel targets, allora l'immagine è "vuota", non contiene alcun oggetto
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                #allora faccio un tensore di zero sia per gt boxes che labels, non ci sarà alcun match a prescindere
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                #ottengo IoU ancore e gt. In base a questo assegno ogni ancora a una Gt.
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                #qui ottengo un tensore lungo quanto tutte le anchors, es 185460
                #ad ogni id delle anchors c'è False o True se rispetta la condizione
                labels_per_image = matched_idxs >= 0          
                
                #nel convertirlo a float32, i False diventano 0; i True diventano 1.
                #questo 1 lo uso come valore di inizializzazione delle label
                #qui mi interessa solo distinguere fra anchors positive e negative, non che classe c'è dentro
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                # Qui etichetto le anchors associate al background, ovvero con threshold che non supera il minimo per poterle considerare associate alle GT box
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                #tutte quelle anchors che non superano la thresh per considerarle positive, cioè che overlappano con le GT box ma solo parzialmente e non abbastanza per considerarle positive
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            #gli indici che non ho toccato nelle 2 operazioni precedenti saranno quelli delle anchors positive
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)       
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        
        #qui levels è un tensor ([0, ..., 0], [1, ..., 1],  [2, ..., 2],  [3, ..., 3],  [4, ..., 4])
        levels = torch.cat(levels, 0) #concateno i levels su prima dimensione
        #ottengo tensor([[0, 0, 0,  ..., 4, 4, 4]])
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        #es output:
        #Se prenms = 2 --> 
        #tensor([[ 35069,  81989,  65828,  65825, 157592, 157589, 157292, 156992, 178550, 178400])
        #	     0  ,    1  ,    2  ,    3  ,    4  ,    0  ,    1  ,    2  ,    3  ,    4
        #queste sono le top bbox per ogni ancora di varia dimensione e ratio
        #es (32,), (64,), (128,), (256,), (512,) -> [139200, 34800, 8700, 2175, 585]

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        # l'id dell'img nel batch, quindi se batch=3, id sarà tensor([[0], [1], [2]])

        #Seleziono objectness, levels e proposals delle migliori bbox delle img di tutto il batch
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]     
        proposals = proposals[batch_idx, top_n_idx]
        
        objectness_prob = torch.sigmoid(objectness) #calcolo le prob

        # i risultati delle proposals
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Args:
        objectness (Tensor)
        pred_bbox_deltas (Tensor)
        labels (List[Tensor])
        regression_targets (List[Tensor])

        Returns:
        objectness_loss (Tensor)
        box_loss (Tensor)
        """
	#la loss del RPN è calcolato in base a quanto sono buone le anchors boxes generate rispetto al GT
        #come prima cosa estrae un numero di anchors positivi e negativi. 
        #la selezione delle anchors è fatta da fg_bg_sampler così:
        #seleziono le indici delle anchors; se il tensore delle labels corrispondente contiene:
        # -1 -> ignora l'anchors
        # 0 -> considero la anchor come negativa (background)
        # 1+ -> considero la anchor come positiva (è di una classe)
        #questo lo posso fare grazie alla funzione assign_targets_to_anchors usata prima
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        #Le anchors negative di background saranno usate solo per il calcolo della loss di classificazione
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        #Qui si calcola la bbox regression loss. Si sommano tutte le loss per le anchors di foreground;
        #non lo fanno per le background perché non vi è associata alcuna GT
        #ogni regression loss per un anchor positive è calcolata come differenza fra i coefficienti
        #di regressione della bbox predetta e i coefficienti del target della GT
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        #loss di classificazione così calcolata: loss(predicted_class, actual_class) -> con class intendo 0 (background) o 1 (foreground), non le classi dell'oggetto
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss
    
    def filter_proposals_custom(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        
        #qui levels è un tensor ([0, ..., 0], [1, ..., 1],  [2, ..., 2],  [3, ..., 3],  [4, ..., 4])
        levels = torch.cat(levels, 0) #concateno i levels su prima dimensione
        #ottengo tensor([[0, 0, 0,  ..., 4, 4, 4]])
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        # l'id dell'img nel batch, quindi se batch=3, id sarà tensor([[0], [1], [2]])

        #Seleziono objectness, levels e proposals delle migliori bbox delle img di tutto il batch
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]     
        proposals = proposals[batch_idx, top_n_idx]
        
        objectness_prob = torch.sigmoid(objectness) #calcolo le prob

        # i risultati delle proposals
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape, gt in zip(proposals, objectness_prob, levels, image_shapes, targets):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            
            #QUI in keep ho gli indici delle proposals sopravvissute alle bbox, ordinate in modo decrescente, quindi i primi risultati nel tensore saranno già le best prediction
            prop = boxes[keep]
            scor = scores[keep]
            #best_det_for_each_gt, taken = self.assign_targets_to_proposals_gt(prop, gt['boxes'], gt['labels'], scor)
            taken = self.select_proposals_custom(prop, gt['boxes'], gt['labels'], scor)
            
            #keep = keep[: self.post_nms_top_n()]
            boxes = prop[taken]
            scores = scor[taken]
            
            if(len(boxes)==0): #protezione contro eventuale assenza di pred
            	device = prop.device
            	boxes=torch.empty((0,)).to(device)
            	scores=torch.empty((0,)).to(device)
            
            final_boxes.append(boxes)
            final_scores.append(scores)
        #print(final_scores)  
        return final_boxes, final_scores
    
    def select_proposals_custom(self, proposals_in_image, gt_boxes_in_image, gt_labels_in_image, scores):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        #2 liste vuote. La label qui conterrà le label dell'oggetto preso dalla GT.
        #matched_idxs = []
        #labels = []
        if gt_boxes_in_image.numel() == 0:
           # Background image
           #img vuota sensa oggetti se dal GT vedo che non ci sono box. Allora tensore tutto a 0
           device = proposals_in_image.device
           clamped_matched_idxs_in_image = torch.zeros(
           	(proposals_in_image.shape[0],), dtype=torch.int64, device=device
           )
           labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
           best_det_for_each_gt=torch.empty((0,)).to(device)
           tensor_det = best_det_for_each_gt
        else:
           match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
           device = proposals_in_image.device
           n_gt_in_image=len(gt_boxes_in_image)
           #n_gt_now=0 #cosi sono sicuro di controllare per ogni gt. In realtà non lo posso fare perché l'assegnamento proposal a gt è basato solo sulla thresh di iou
           tensor_taken=torch.empty(0, dtype=torch.int64).to(device) #per non prendere 2 volte le stesse proposals. Può capitare in teoria, ma è raro. Con le negative se se ne tengono tante questo controllo rallenta molto le performance. Allora rimuovo le negative duplicate e basta
           if self.use_not_overlapping_proposals:
           	tensor_neg_taken=torch.empty(0, dtype=torch.int64).to(device)
           for match_matrix_each_gt in match_quality_matrix: #itero per GT			
              iou_values , iou_indices = torch.sort(match_matrix_each_gt, 0, descending=True)
              #l'indice qui è rispetto alla pos originale in match_matrix_each_gt! es: iou_indices[:0] da 145, il top iou è la prop con indice 145 in match_matrix_each_gt
              """
              Premessa 1 (attivare proposals): ordinerei tutte le proposals per ogni target in base a IOU con esso (tralasciando condifence e label), seleziono le prime n e le passo alla loss
              """
              #numero di proposal per ogni GT da tenere indipendentemente dal loro score, label ecc..., ma solo ordinate per loro IoU. In teoria sono necessariamente positive
              top_prop=iou_indices[:self.n_top_iou_to_keep]
              for index_prop in top_prop:
                 if index_prop not in tensor_taken:
                    tensor_taken = torch.cat([tensor_taken, index_prop.unsqueeze(0)])
                 else:   #se è già preso, allora cerco il prossimo index
                    next_index = next((i for i in iou_indices if i not in tensor_taken), None)
                    if next_index is None: #se è None vuol dire che non ho trovato un indice
                       break #posso uscire subito dal for perché vuol dire che li ho passati tutti e non ho trovato nulla che soddisfi la condizione
                    tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
              """
              Premessa 2 (sopprimere predicitions FPiou): considero solo le bbox con label 1 (persona), le ordino per confidence  e seleziono le prime t con iou < di 0.5
              """
              """
              Commento mio: in realtà il modello lavora con objectness score e filtra solo in base a quello se è persona o meno. Questo vuol dire che non so mai se una persona è persona, ma solo se soddisfa la soglia di threshold messa dal matcher. Siccome la label non mi interessa, a sto punto ordino per confidence tutte le bbox che hanno iou<0.5 e propago quelle, che so già che sono background
              """
              neg_iou_indices = iou_indices[match_matrix_each_gt[iou_indices] < self.iou_neg_thresh] #gli indici che soddisfano IoU. L'ordine IoU è conservato, ma tanto poi faccio il sort per score
              neg_iou_indices, _ = torch.sort(neg_iou_indices, 0, descending=False) #riordino per score
              #print(neg_iou_indices)
              #print(match_matrix_each_gt[neg_iou_indices])
              #print(scores[neg_iou_indices])
              top_fpiou=neg_iou_indices[:self.n_top_neg_to_keep]
              for index_fpiou in top_fpiou:       			
                 if index_fpiou not in tensor_taken:
                    if self.use_not_overlapping_proposals:
                       if tensor_neg_taken.numel() == 0: #tensore negativo vuoto, aggiungo la prop subito
                          tensor_taken = torch.cat([tensor_taken, index_fpiou.unsqueeze(0)])
                          tensor_neg_taken = torch.cat([tensor_neg_taken, index_fpiou.unsqueeze(0)])
                       else: #devo verificare che non vado a prendere una negativa che overlappi tanto con un'altra negativa che ho già preso
                          match = box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[index_fpiou].unsqueeze(0))
                          if torch.any(match > self.overlapping_prop_thresh):
                             next_index = next((i for i in neg_iou_indices if torch.all(box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[i].unsqueeze(0)) <= self.overlapping_prop_thresh)), None)
                             if next_index is None: 
                                break
                             tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                             tensor_neg_taken = torch.cat([tensor_neg_taken, next_index.unsqueeze(0)])
                          else:
                             tensor_taken = torch.cat([tensor_taken, index_fpiou.unsqueeze(0)])
                             tensor_neg_taken = torch.cat([tensor_neg_taken, index_fpiou.unsqueeze(0)])
                    else: #metodo non overlapping
                       tensor_taken = torch.cat([tensor_taken, index_fpiou.unsqueeze(0)])
                 else:
                    if self.use_not_overlapping_proposals:
                       if tensor_neg_taken.numel() == 0: #tensore negativo vuoto, aggiungo la prop subito
                          next_index = next((i for i in neg_iou_indices if i not in tensor_taken), None)
                          if next_index is None: #se è None vuol dire che non ho trovato un indice
                             break
                          tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                          tensor_neg_taken = torch.cat([tensor_neg_taken, next_index.unsqueeze(0)])
                       else:
                          next_index = next((i for i in neg_iou_indices if i not in tensor_taken and (torch.all(box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[i].unsqueeze(0)) <= self.overlapping_prop_thresh))), None)
                          if next_index is None: #se è None vuol dire che non ho trovato un indice
                             break
                          tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                          tensor_neg_taken = torch.cat([tensor_neg_taken, next_index.unsqueeze(0)])
                    else: #metodo non overlapping, se è già preso, allora cerco il prossimo index
                       next_index = next((i for i in neg_iou_indices if i not in tensor_taken), None) 
                       if next_index is None: #se è None vuol dire che non ho trovato un indice
                          break #posso uscire subito dal for perché vuol dire che li ho passati tutti e non ho trovato nulla che soddisfi la condizione
                       tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
           """
           Mia premessa: porto dietro tutto ciò che è sicuramente background. Prop molto negative
           """
           if self.n_top_absolute_bg_to_keep>0: #se è = 0 vuol dire che non uso il metodo per le bg, allora sta roba sotto non la faccio nememno e risparmio performance-
              absolute_bg_mask = torch.arange(len(proposals_in_image)).to(device) #rappresenta gli indici di elementi che verranno settate a -1 se non appartengono al bg.
              for match_matrix_each_gt in match_quality_matrix:       		
                 bg_thresh=0.00 #hard coded ma va bene visto che devo tenere quelle sicure nel bg      		
                 absolute_bg_mask[match_matrix_each_gt > bg_thresh] = -1
        	# Idea: tengo tutte le negative che cadono nel background al 100% e che non cadono in nessun GT.
              absolute_bg_indices = absolute_bg_mask[absolute_bg_mask != -1]
              #Problema: le bg contribuisco tantissimo alla ricostruzione dell'immagine. Idea: tieni solo quelle che hanno alta confidence
              absolute_bg_indices = absolute_bg_indices[scores[absolute_bg_indices] >= self.absolute_bg_score_thresh]
              #sono già ordinati per score, essendo indici delle prop come l'originale. Mi basta eliminare le -1 che sono quelle che non soddisfano quella thresh
              top_absolute_bg = absolute_bg_indices[:self.n_top_absolute_bg_to_keep]
              for index_vn in top_absolute_bg:
                 if index_vn not in tensor_taken:
                    if self.use_not_overlapping_proposals:
                       #posso evitare di verificare che il tensore negativo sia vuoto
                       match = box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[index_vn].unsqueeze(0))
                       if torch.any(match > self.overlapping_prop_thresh):
                          next_index = next((i for i in absolute_bg_indices if torch.all(box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[i].unsqueeze(0)) <= self.overlapping_prop_thresh)), None)
                          if next_index is None: 
                             break
                          tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                          tensor_neg_taken = torch.cat([tensor_neg_taken, next_index.unsqueeze(0)])
                       else:
                          tensor_taken = torch.cat([tensor_taken, index_vn.unsqueeze(0)])
                          tensor_neg_taken = torch.cat([tensor_neg_taken, index_vn.unsqueeze(0)])
                    else: #metodo non overlapping
                       tensor_taken = torch.cat([tensor_taken, index_vn.unsqueeze(0)])
                 else:
                    if self.use_not_overlapping_proposals:
                       next_index = next((i for i in absolute_bg_indices if i not in tensor_taken and (torch.all(box_ops.box_iou(proposals_in_image[tensor_neg_taken], proposals_in_image[i].unsqueeze(0)) <= self.overlapping_prop_thresh))), None)
                       if next_index is None: #se è None vuol dire che non ho trovato un indice
                          break
                       tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                       tensor_neg_taken = torch.cat([tensor_neg_taken, next_index.unsqueeze(0)])
                    else: #metodo non overlapping
                       next_index = next((i for i in absolute_bg_indices if i not in tensor_taken), None) 
                       if next_index is None: #se è None vuol dire che non ho trovato un indice
                          break
                       tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
           tensor_taken = torch.unique(tensor_taken, dim=0) #rimuovo indici doppi per non far contare 2 volte stessa proposal
           tensor_taken, _ = torch.sort(tensor_taken, 0, descending=False) #per sicurezza riordino
        #print(len(gt_boxes_in_image))
        return tensor_taken

    def assign_targets_to_anchors_custom(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        #2 liste di tensori. Lista perché ogni tensore che contiene la label intesa come bg, fg o niente.
        labels = []
        matched_gt_boxes = [] #questo è un tensore contenente le gt boxes che le anchors matchano
        indexes = []
        indexes_offset = []
        #in particolare i tensori saranno lunghi quante sono le anchors delle img, es 185460             
        #COME FUNZIONA: a ciascuna ancora viene associato un gt. Rispetto a tale gt, verrà poi calcolato il box regression. Quindi se ho 100000 ancore, per ognuna di esse avrò il gt
        n_offset=0 #in base al batch size alla fine, quante len(anchors) devo sommare
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            #DEFINISCO INDEX OFFSET DA USARE POI PER RECUPERE SOLO L'OBJECTNESS E PREDBBOX DELTA CHE MI SERVONO
            tot_offset=0
            if n_offset>0:
               for a in range(0, n_offset):
                  tot_offset+=len(anchors[n_offset])
            #print(tot_offset)
            #Se non c'è nessuna GT box nel targets, allora l'immagine è "vuota", non contiene alcun oggetto
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                #allora faccio un tensore di zero sia per gt boxes che labels, non ci sarà alcun match a prescindere
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                #ottengo IoU ancore e gt. In base a questo assegno ogni ancora a una Gt.
                device = anchors_per_image.device
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device) #gli indici
                #matched_gt_taken=torch.empty(0, dtype=torch.float32).to(device)
                #labels_taken=torch.empty(0, dtype=torch.float32).to(device)
                for i, match_matrix_each_gt in enumerate(match_quality_matrix):
                   """
                   Premessa 1: tengo quella con IoU più alta. 
                   """
                   iou_values , iou_indices = torch.sort(match_matrix_each_gt, 0, descending=True)
                   pos_iou_indices = iou_indices[match_matrix_each_gt[iou_indices] >= self.proposal_matcher.high_threshold]
                   top_prop=iou_indices[:2]
                   for index_prop in top_prop:
                      if index_prop not in tensor_taken:
                         tensor_taken = torch.cat([tensor_taken, index_prop.unsqueeze(0)])
                         #matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[i].unsqueeze(0)])
                         #labels_taken= torch.cat([labels_taken, torch.ones((), dtype=torch.float32).to(device).unsqueeze(0)])
                      else:   #se è già preso, allora cerco il prossimo index
                         next_index = next((i for i in iou_indices if i not in tensor_taken), None)
                         if next_index is None: #se è None vuol dire che non ho trovato un indice
                            break
                         tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])
                   """
                   Premessa 2: tengo quelle con iou più alto sotto soglia di 0.3
                   """
                   neg_iou_indices = iou_indices[match_matrix_each_gt[iou_indices] < self.proposal_matcher.low_threshold]
                   #riordinarle non mi serve, voglio quelle negative con iou maggiore
                   top_fpiou=neg_iou_indices[:5]
                   for index_fpiou in top_fpiou:       			
                      if index_fpiou not in tensor_taken:
                         tensor_taken = torch.cat([tensor_taken, index_fpiou.unsqueeze(0)])
                         #matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[i].unsqueeze(0)])
                         #labels_taken= torch.cat([labels_taken, torch.zeros((), dtype=torch.float32).to(device).unsqueeze(0)])
                      else:
                         next_index = next((i for i in neg_iou_indices if i not in tensor_taken), None) 
                         if next_index is None: #se è None vuol dire che non ho trovato un indice
                            break 
                         tensor_taken = torch.cat([tensor_taken, next_index.unsqueeze(0)])  
                   #fine selezione matrix. Ora devo tenere nella matrice solo le prop selezionate, rifarle riclassificare
                
                #RIMAPPARE INDICI RISPETTO A OBJECTNESS
                #tensor_taken = tensor_taken + tot_offset
                tensor_taken_offset = tensor_taken + tot_offset
                #FARE SORTING CON INDICI DECRESCENTI OTTENUTI QUA, ottengo indices che USO PER SORTARE GLI ALTRI DUE TENSORI. NEL MIO CASO MI BASTA RIORDINARE ORDINE CRESCENTE PER AVERE ORDINAMENTO SIMILE ALL'ORIGINALE
                tensor_taken = torch.unique(tensor_taken, dim=0) #rimuovo indici doppi
                tensor_taken_offset = torch.unique(tensor_taken_offset, dim=0)
                tensor_taken, _ = torch.sort(tensor_taken, 0, descending=False) #per sicurezza riordino
                tensor_taken_offset, _ = torch.sort(tensor_taken_offset, 0, descending=False)
                
                #RIOTTENGO LE LABEL GIUSTE FACENDO NUOVA MATRIX
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image[tensor_taken])
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                #print(matched_idxs)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                #print(matched_gt_boxes_per_image)

                #qui ottengo un tensore lungo quanto tutte le anchors, es 185460
                #ad ogni id delle anchors c'è False o True se rispetta la condizione
                labels_per_image = matched_idxs >= 0          
                
                #nel convertirlo a float32, i False diventano 0; i True diventano 1.
                #questo 1 lo uso come valore di inizializzazione delle label
                #qui mi interessa solo distinguere fra anchors positive e negative, non che classe c'è dentro
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                # Qui etichetto le anchors associate al background, ovvero con threshold che non supera il minimo per poterle considerare associate alle GT box
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                #tutte quelle anchors che non superano la thresh per considerarle positive, cioè che overlappano con le GT box ma solo parzialmente e non abbastanza per considerarle positive
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            #gli indici che non ho toccato nelle 2 operazioni precedenti saranno quelli delle anchors positive
            n_offset=n_offset+1
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            #labels.append(labels_taken)
            #matched_gt_boxes.append(matched_gt_taken)
            indexes.append(tensor_taken)
            indexes_offset.append(tensor_taken_offset)
        #print(labels)
        #print(matched_gt_boxes)
        return labels, matched_gt_boxes, indexes, indexes_offset

    def assign_targets_to_anchors_custom_v2(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]], objectness: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:

        #2 liste di tensori. Lista perché ogni tensore che contiene la label intesa come bg, fg o niente.
        labels = []
        matched_gt_boxes = [] #questo è un tensore contenente le gt boxes che le anchors matchano
        indexes = [] #servono per essere retrocompatibile con il codice dell'rpn.py
        indexes_offset = [] #serve per batch_size>1, perchè l'objectness sono indici tutti insieme delle img
        n_offset=0 #in base al batch size alla fine, quante len(anchors) devo sommare
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            
            tot_offset=0
            if n_offset>0:
               for a in range(0, n_offset):
                  tot_offset+=len(anchors[n_offset])
            
            #Se non c'è nessuna GT box nel targets, allora l'immagine è "vuota", non contiene alcun oggetto
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                #allora faccio un tensore di zero sia per gt boxes che labels, non ci sarà alcun match a prescindere
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
                my_labels = labels_per_image
                my_matched_gt_boxes = matched_gt_boxes_per_image
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device)
                tensor_taken_offset=torch.empty(0, dtype=torch.int64).to(device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                #ottengo IoU ancore e gt. In base a questo assegno ogni ancora a una Gt.
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs >= 0          
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
                #IDEA: prendo indici con label 1 che sono positivi, con label 0 che sono negativi
                my_pos = torch.where(labels_per_image == 1.0)[0] #[0] per evitare che venga considerato tupla
                my_neg = torch.where(labels_per_image == 0.0)[0]
                
                n_gt = len(gt_boxes)
                device = anchors_per_image.device
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device) #tensori di inizializzazione
                matched_gt_taken=torch.empty(0, dtype=torch.float32).to(device)
                labels_taken=torch.empty(0, dtype=torch.float32).to(device)
                "Premessa 1: anchors positive (con IoU sopra al matcher)"
                only_pos = match_quality_matrix[:, my_pos] #recupero solo i valori di IoU positivi e neg
                only_neg = match_quality_matrix[:, my_neg] #questi sono in matrice NxM dove N numero gt
                # Per ogni anchor, trovo la best match gt, ovvero quella con IoU maggiore fra le gt. Prima lo faccio con i pos, poi con i neg. Non ci possono essere duplicati da rimuovere qui
                matched_vals, matches_idx = only_pos.max(dim=0) #matches_idx rappresenta il gt a cui associare l'anchor; se è 0 significa che l'anchor ha max IoU con il gt 0.
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True) #Ora li ordino in modo che così sono ordinate per IoU max          
                my_pos_sort = my_pos[sort_ma_val_idx] #riordino i positivi rispetto a tale ordine
                matches_idx_sort = matches_idx[sort_ma_val_idx] #riordino anche gli id
                """
                n_gt -> 2
                only_pos -> tensor([[0.7189, 0.1421, 0.0642, 0.7189, 0.7069], [0.1407, 0.6593, 0.6593, 0.1407, 0.1358]], device='cuda:0')
                matched_vals -> tensor([0.7189, 0.6593, 0.6593, 0.7189, 0.7069], device='cuda:0')
                matches_idx -> tensor([0, 1, 1, 0, 0], device='cuda:0') #questo indica che di tutte le anchors positive, 2 (quelle con 1) matchano meglio con la gt 1 
                sort_ma_val_idx -> tensor([0, 3, 4, 1, 2], device='cuda:0') #rappresenta come gli indici originali (0,1, 2... 4) si trovano ora nel nuovo ordine (0, 3 ... 2). Lo uso poi per riordinare 
                my_pos_sort -> tensor([217123, 217174, 217225, 217139, 217142], device='cuda:0')
                matches_idx_sort -> tensor([0, 0, 0, 1, 1], device='cuda:0')
                """
                for val in range(0, n_gt): #per ogni gt
                   index = (matches_idx_sort == val) #cerco nel matched id quelli corrispondenti alla gt attuale; è una maschera booleana
                   true_idx = torch.where(index)[0] #prendo solo i valori a true
                   true_idx = true_idx[:self.n_top_pos_to_keep]  #ne tengo i primi n (avranno iou più alta)      
                   tensor_taken = torch.cat([tensor_taken, my_pos_sort[true_idx]]) #salvo indicid da tenere
                   for i in range(0, self.n_top_pos_to_keep): #ora aggiungo n matched gt e labels quante n anchors tengo. Questo lo faccio per fare la label e definire la matched gt box dell'anchors. Nelle positive potrei anche non farlo, ma nelle negative sono obbligato perché rpn.py di default considera le negative come tutte associate alla prima matched_gt_box
                      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[val].unsqueeze(0)])
                      labels_taken= torch.cat([labels_taken, torch.tensor(1.0, dtype=torch.float32, device=device).unsqueeze(0)])
     
                "Premessa 2: Anchors negative (IoU sotto al matcher). Ordinate per IoU, poi riordinate per Objectness score maggiore"                 
                matched_vals, matches_idx = only_neg.max(dim=0)
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True) 
                #estraggo objectness. Non ha senso normalizzarla tra 0 e 1 e non è possibile usare una thresh di score sopra il quale prendere un objectness (l'objectness ha valori molto random e imprevedibili.
                #min_value = my_objectness.min()
                #max_value = my_objectness.max()
                #norm_obj = (my_objectness - min_value) / (max_value - min_value)  
                objectness = objectness.flatten()
                my_obj = objectness[sort_ma_val_idx+tot_offset]
                my_obj_sort, my_obj_sort_idx = torch.sort(my_obj, descending=True) #sort per score, diverso dall'originale
                #my_neg_sort = my_neg[sort_ma_val_idx]
                #matches_idx_sort = matches_idx[sort_ma_val_idx]
                my_neg_sort = my_neg[my_obj_sort_idx]
                matches_idx_sort = matches_idx[my_obj_sort_idx]
                neg_already_taken=torch.empty(0, dtype=torch.int64).to(device)
                for val in range(0, n_gt):
                   index = (matches_idx_sort == val)
                   true_idx = torch.where(index)[0]
                   true_idx = true_idx[:self.n_top_neg_to_keep]   
                   tensor_taken = torch.cat([tensor_taken, my_neg_sort[true_idx]])
                   neg_already_taken = torch.cat([neg_already_taken, my_neg_sort[true_idx]])
                   for i in range(0, self.n_top_neg_to_keep):
                      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[val].unsqueeze(0)])
                      labels_taken= torch.cat([labels_taken, torch.tensor(0.0, dtype=torch.float32, device=device).unsqueeze(0)])
                
                "Premessa 3: Anchors nel background completo (IoU=0.0). Riordinate secondo ordine originale, poi riordinate secondo Objectness score maggiore"
                if self.n_top_bg_to_keep>0:
                   
                   bg_mask = (sort_ma_val == 0.0)
                   bg_ma_val_idx = sort_ma_val_idx[bg_mask]
                   #bg_ma_val_idx, _ = torch.sort(bg_ma_val_idx, descending=False) #per ordine originale
                   #in teoria sono già ordinati come ordine originale, perché avendo tutti IoU zero quando vengono riordinati per IoU rimarranno prima le prima occorrenze (ovvero quelli con score più alto). Li ho riordinati  per sicurezza
                   
                   #elimino le neg già prese
                   neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
                   bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
                   
                   my_obj = objectness[bg_ma_val_idx+tot_offset]
                   my_obj_sort, my_obj_sort_idx = torch.sort(my_obj, descending=True) #sort per score, diverso dall'originale
                   #my_bg_sort = my_neg[bg_ma_val_idx]
                   #bg_matches_idx = matches_idx[bg_ma_val_idx]
                   my_bg_sort = my_neg[my_obj_sort_idx]
                   bg_matches_idx = matches_idx[my_obj_sort_idx]
                   #tengo tutti i valori dell'array sopra a una soglia di objectness
                   thresh = torch.nonzero(my_obj_sort>=self.objectness_bg_thresh).flatten()
                   #i true idx li prendo da quelli che soddisfano la soglia             
                   #le background proposal sono tutte associate a un gt a caso. Quindi ne recupero n*bg_to_keep
                   #true_idx = torch.arange(len(bg_matches_idx)) #sono già ordinati
                   #true_idx = true_idx[:self.n_top_bg_to_keep*n_gt]
                   true_idx = thresh[:self.n_top_bg_to_keep*n_gt]
                   tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
                   for i in range(0, self.n_top_bg_to_keep*n_gt): #prendo la prima come fa di default l'rpn in questo caso
                      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[0].unsqueeze(0)])
                      labels_taken= torch.cat([labels_taken, torch.tensor(0.0, dtype=torch.float32, device=device).unsqueeze(0)])
                   #for i in true_idx:
                   #   if my_bg_sort[i] not in tensor_taken: #guardia contro i doppi che potrebbero capitare (se già presi fra i negativi sopra)
                   #      tensor_taken = torch.cat([tensor_taken, my_bg_sort[i].unsqueeze(0)])
                   #      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[0].unsqueeze(0)])
                   #      labels_taken= torch.cat([labels_taken, torch.tensor(0.0, dtype=torch.float32, device=device).unsqueeze(0)])
                   #   else: #per assicura di prendere n bg se ci sono; se trovo duplicato passo a prossima
                   #      next_index = next((i for i in thresh if my_bg_sort[i] not in tensor_taken), None)
                   #      if next_index is None: #se è None vuol dire che non ho trovato un indice
                   #         break
                   #      tensor_taken = torch.cat([tensor_taken, my_bg_sort[next_index].unsqueeze(0)])
                   #      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[0].unsqueeze(0)])
                   #      labels_taken= torch.cat([labels_taken, torch.tensor(0.0, dtype=torch.float32, device=device).unsqueeze(0)])
                
                #riordino gli indici delle anchors secondo ordine originale, poi riordino anche le matched gt boxes e labels rispettive allo stesso modo
                tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
                my_matched_gt_boxes = matched_gt_taken[orig_order]
                my_labels = labels_taken[orig_order]
                
                tensor_taken_offset = tensor_taken + tot_offset
                
                #Ora tutti ordinati per IoU. Alla fine di tutto riordino per score
            #gli indici che non ho toccato nelle 2 operazioni precedenti saranno quelli delle anchors positive
            #incremento offset per prossima img nel batch
            n_offset=n_offset+1
            #labels.append(labels_per_image)
            #matched_gt_boxes.append(matched_gt_boxes_per_image)
            #mie append
            labels.append(my_labels)
            matched_gt_boxes.append(my_matched_gt_boxes)
            indexes.append(tensor_taken)
            indexes_offset.append(tensor_taken_offset)       
        return labels, matched_gt_boxes, indexes, indexes_offset

    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """ Args:
        images (ImageList): le img su cui fare le predizioni
        features (Dict[str, Tensor]): features computate dalla backbone a partire dalle immagini. Ogni feature list corrisponde a diversi livelli di feature.
        targets (List[Dict[str, Tensor]]): le GT boxes delle img (optional).
        
        Returns:
        boxes (List[Tensor]): le boxes predette dalla rpn, una per img.
        losses (Dict[str, Tensor]): le loss del modello durante il training; nel testing è vuoto.
        """
        # RPN uses all feature maps that are available
        # PASSO 0: ottengo le feature dalla backbone
        features = list(features.values())
        
        # PASSO 1: calcolo lo score di classificazione e le bbox regressor.
        # lo stride della backbone è lo stesso usato per generare le anchors (3x3)
        # quindi c'è una corrispondenza 1:1 tra le anchor boxes e le class score e regressor
        # es 20000 anchor boxes che ricoprono l'immagine -> 20 000 class score e bbox regressor
        #All'inizio le anchors usate sono messe a caso a ricoprire l'img
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        #Prima vengono generate le anchors sulle feature maps e poi trasformate sull'img
        #la loss quindi delle anchors di objectness e pred_bbox_deltas guarda quanto sono buone la generazione delle anchor di default
        
        num_images = len(anchors)    
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        #es valore: [torch.Size([3, 232, 200]), torch.Size([3, 116, 100]), torch.Size([3, 58, 50]), torch.Size([3, 29, 25]), torch.Size([3, 15, 13])]
        
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        #es valore: [139200, 34800, 8700, 2175, 585]
        
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through the proposals
        # Quella sotto è una classe che si occupa di codificare un set di bbox nella rappresentazione
        # usata per trainare i regressors.
        #Dal codice: From a set of original boxes (il secondo arg qui sotto) and encoded relative box offsets (il primo arg), get the decoded boxes.
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)        
        proposals = proposals.view(num_images, -1, 4)
        #Qui proposals è ancora uguale al numero di anchors
        
        #boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        
        #PASSO 2: qui vado a prendere tutte le anchor boxes del generatore per livello (es 185460) e vado
        #a eliminare quelle che non soddisfano certe condizioni: seleziono solo le migliori box per livello
        #rimuovo le bbox piccole e con score basso; applico nms e infine tengo solo le predizioni migliori
        #boxes, scores, _ = self.filter_proposals_mia(proposals, objectness, images.image_sizes, num_anchors_per_level, targets)
        """
        if self.use_custom_filter_proposals:
        	if self.training:
        		boxes, scores = self.filter_proposals_custom(proposals, objectness, images.image_sizes, num_anchors_per_level, targets)
        	else:
        		#temp = self.score_thresh #intanto che testo lo score thresh non ci deve essere perchè la faster si deve comportare come una faster normale
        		#self.score_thresh = 0.0
        		boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        		#self.score_thresh = temp
        """
        #else:
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        #	boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,10))
        import torchvision.transforms as transforms
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        
        for box, score, img in zip(boxes, scores, images.tensors):
        	ax = plt.gca()
        	img = unnormalize(img)
        	img = unnormalize(img)
        	plt.imshow(img.cpu().permute(1, 2, 0).detach().numpy())
        	
        	for (xmin, ymin, xmax, ymax), prob in zip(box, score):
        		ax.add_patch(plt.Rectangle((xmin.cpu(), ymin.cpu()), xmax.cpu() - xmin.cpu(), ymax.cpu() - ymin.cpu(), fill=False, color='red', linewidth=3))
        		prob = f'{prob.item():0.3f}'
        		ax.text(xmin, ymin, prob, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        	plt.show()
        	plt.clf()
        """
        
        losses = {}
        #PASSO 3: solo in training, alleno il modello a migliorare le anchors.
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")           
            if self.use_custom_filter_anchors:
               #labels, matched_gt_boxes, indexes, indexes_offset = self.assign_targets_to_anchors_custom(anchors, targets)
               labels, matched_gt_boxes, indexes, indexes_offset = self.assign_targets_to_anchors_custom_v2(anchors, targets, objectness)
               device = anchors[0].device
               #my_anchors=torch.empty(0, dtype=torch.int64).to(device)
               my_anchors=[]
               for anchors_in_image, indexes_in_image in zip(anchors, indexes):
                  #my_anchors = torch.cat([my_anchors, anchors_in_image[indexes_in_image].unsqueeze(0)])
                  my_anchors.append(anchors_in_image[indexes_in_image])
               my_objectness=torch.empty(0, dtype=torch.float32).to(device)
               my_pred_bbox_deltas = torch.empty(0, dtype=torch.float32).to(device)
               for indexes in indexes_offset:
                  my_objectness = torch.cat((my_objectness, objectness[indexes]), dim=0)
                  my_pred_bbox_deltas = torch.cat((my_pred_bbox_deltas, pred_bbox_deltas[indexes]), dim=0)
               regression_targets = self.box_coder.encode(matched_gt_boxes, my_anchors)          
               loss_objectness, loss_rpn_box_reg = self.compute_loss(
                my_objectness, my_pred_bbox_deltas, labels, regression_targets)           
            else:
               #distinguo fra le anchors positive e le negative. Per ogni anchors segno la matched box corrispondente
               labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
               #Encode a set of proposals with respect to some reference boxes Args:
               #reference_boxes (Tensor): reference boxes
               #proposals (Tensor): boxes to be encoded
               regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
               #calcolo della loss effettiva. Quella di objectness rappresenta quanto le mie anchors sono
               #"buone", ovvero ad es con quanto score la mia anchors positiva è effettivamente giusto che sia positiva secondo il GT
               #quella di box_reg rappresenta quanto sono buone le mie anchors boxes positive come bbox di oggetti
               loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            #print(losses)
        #return boxes, losses
        return boxes, scores, losses #scores aggiunto io
