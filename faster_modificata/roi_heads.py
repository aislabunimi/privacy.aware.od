from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from torchvision.models.detection import _utils as det_utils

#####
# CODE COPYPASTED FROM TORCHVISION DETECTION, ADDED MY COMMENTS AND PARAMETERS; REMOVED USELESS PARTS FOR ME LIKE KEYPOINTS AND MASKS FUNCTIONS.
#####

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN. Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    #la classification loss è calcolata su tutte le pred, sia le pos che negative, in base alla probabilità di classe assegnata e rispettiva label.
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    #la box loss è calcolata solo sulle positive ovvimanete.
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        use_custom_filter_proposals=True,
        n_top_pos_to_keep=1, 
        n_top_neg_to_keep=5,
        n_top_bg_to_keep=0,
        obj_bg_score_thresh=0.9,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        #miei parametri
        self.use_custom_filter_proposals = use_custom_filter_proposals
        self.n_top_pos_to_keep = n_top_pos_to_keep
        self.n_top_neg_to_keep = n_top_neg_to_keep
        self.n_top_bg_to_keep = n_top_bg_to_keep
        self.obj_bg_score_thresh = obj_bg_score_thresh

        #per me inutili, tengo per backward compatibility
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    #questa e has_keypoint sono usate nel check target; non mi interessano, le tengo per backward compatibility
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        #2 liste vuote. La label qui conterrà le label dell'oggetto preso dalla GT.
        matched_idxs = []
        labels = []
        match_q_matrixes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                #img vuota sensa oggetti se dal GT vedo che non ci sono box. Allora tensore tutto a 0
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
               
                #calcolo IoU di ogni proposal rispetto al GT. Ottengo matrice di M gt x N prop.
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                #ad ogni proposal assegno la GT corrispondendente in base all'IoU se soddisfa thresh.
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                #otterrò un tensor([ 0, -1, -1,  0,  1,  2]) ad esempio
                #il -1 indica che la prop non è associata a nessuna GT; 0 indica che la prop è associata alla prima GT; 1 alla seconda; 2 alla terza .... ci saranno n valori in base alle n gt possibili.

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                #il -1 da quello che c'è scritto nel codice originale dà un errore con la parte sotto, quindi preparo questa variabile settando quei valori a 0

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                #creo un tensore dove nel mio caso metterò tutti 1, perché ho solo classe persona. è come se partissi immaginando che tutte le proposal hanno label 1, e poi vado a filtrare i valori sotto
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold) #setto label di quegli indici a 0
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds) #-1 sono indici ignorati
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            match_q_matrixes.append(match_quality_matrix) #le ritorno per usarle dopo nel mio filtro
        return matched_idxs, labels, match_q_matrixes


    def assign_targets_to_proposals_custom_v2(self, proposals, gt_boxes, gt_labels, obj_score):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        #2 liste vuote. La label qui conterrà le label dell'oggetto preso dalla GT.
        labels = []
        matched_idxs = [] #questo è un tensore contenente le gt boxes che le anchors matchano
        indexes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, obj_score_in_image in zip(proposals, gt_boxes, gt_labels, obj_score):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                #img vuota sensa oggetti se dal GT vedo che non ci sono box. Allora tensore tutto a 0
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                my_clamped = clamped_matched_idxs_in_image
                my_labels = labels_in_image
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
               
                #calcolo IoU di ogni proposal rispetto al GT. Ottengo matrice di M gt x N prop.
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                #otterrò un tensor([ 0, -1, -1,  0,  1,  2]) ad esempio
                #il -1 indica che la prop non è associata a nessuna GT; 0 indica che la prop è associata alla prima GT; 1 alla seconda; 2 alla terza .... ci saranno n valori in base alle n gt possibili.

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                #il -1 da quello che c'è scritto nel codice originale dà un errore con la parte sotto, quindi preparo questa variabile settando quei valori a 0
                #è un numero 0, 1, 2 che rappresenta per ogni proposal a quale gt deve essere associata, sia le prop positive che le neg
                
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                #creo un tensore dove nel mio caso metterò tutti 1, perché ho solo classe persona. è come se partissi immaginando che tutte le proposal hanno label 1, e poi vado a filtrare i valori sotto
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold) #setto label di quegli indici a 0
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds) #-1 sono indici ignorati
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                
                #original_index = torch.arange(len(proposals_in_image)) #rappresenta gli indici originali per perscare i corrispondenti clamped_matched e labels
                #IDEA: RECUPERO INDICI CON LABEL=1 che sono negativi o del primo gt o dell'altro
                my_pos = torch.where(labels_in_image >= 1)[0]
                #questi sono negativi o primo gt o degli altri
                my_neg = torch.where(labels_in_image == 0)[0]

                n_gt = len(gt_boxes_in_image)
                device = proposals_in_image.device
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device)
                only_pos = match_quality_matrix[:, my_pos]
                only_neg = match_quality_matrix[:, my_neg]
                
                """ Premessa 1 (attivare proposals): ordinerei tutte le proposals per ogni target in base a IOU con esso (tralasciando condifence e label), seleziono le prime n e le passo alla loss """
                #estraggo i valori positivi
                matched_vals, matches_idx = only_pos.max(dim=0)
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)      
                my_pos_sort = my_pos[sort_ma_val_idx]
                matches_idx_sort = matches_idx[sort_ma_val_idx]            
                #le prime n sono il gt. I gt li devo tenere comunque perché così funziona l'originale, e poi prenderne altri n. Allora li seleziono rimuovendo le prime n_gt occorrenze, e recuperando tali n_gt occorrenze subito dopo il for
                my_pos_sort_without_gt = my_pos_sort[n_gt:]
                matches_idx_without_gt = matches_idx_sort[n_gt:]
                for val in range(0, n_gt):
                   index = (matches_idx_without_gt == val)
                   true_idx = torch.where(index)[0]
                   true_idx = true_idx[:self.n_top_pos_to_keep]
                   tensor_taken = torch.cat([tensor_taken, my_pos_sort_without_gt[true_idx]])

                tensor_taken = torch.cat([tensor_taken, my_pos_sort[:n_gt]])
                """ Premessa 2 (sopprimere predicitions FPiou): considero solo le bbox con label 1 (persona), le ordino per confidence  e seleziono le prime t con iou < di 0.5 (qui è lo standard) """
                #Ora negativi
                matched_vals, matches_idx = only_neg.max(dim=0) #prima sono ordinati per score
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True) #ordina per iou          
                sort_ma_val_idx_score, _ = torch.sort(sort_ma_val_idx, descending=False) #più facile poi da gestire, ordinati per score adesso
                my_neg_sort = my_neg[sort_ma_val_idx_score]
                matches_idx_sort = matches_idx[sort_ma_val_idx_score]
                neg_already_taken=torch.empty(0, dtype=torch.int64).to(device)
                for val in range(0, n_gt):
                   index = (matches_idx_sort == val)
                   true_idx = torch.where(index)[0]
                   true_idx = true_idx[:self.n_top_neg_to_keep]        
                   tensor_taken = torch.cat([tensor_taken, my_neg_sort[true_idx]])
                   neg_already_taken = torch.cat([neg_already_taken, my_neg_sort[true_idx]])
                
                """ Premessa 3 Full background (IoU zero con tutti gli oggetti). Ordinate per score """
                if self.n_top_bg_to_keep>0:
                   
                   bg_mask = (sort_ma_val == 0.0) #IoU pari a 0
                   bg_ma_val_idx = sort_ma_val_idx[bg_mask]

                   #elimino le neg già prese
                   neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
                   bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
                   
                   my_obj = obj_score_in_image[bg_ma_val_idx]
                   
                   bg_ma_val_idx, _ = torch.sort(bg_ma_val_idx, descending=False)
                   my_bg_sort = my_neg[bg_ma_val_idx]
                   bg_matches_idx = matches_idx[bg_ma_val_idx]
                   #in teoria sono già ordinati per score, perché avendo tutti IoU zero quando vengono riordinati per IoU rimarranno prima le prima occorrenze (ovvero quelli con score più alto). Li ho riordinati  per sicurezza
                   #le background proposal sono tutte associate al primo gt. Quindi ne recupero n*bg_to_keep
                   #index = (bg_matches_idx == 0) #giusto di sicurezza
                   #true_idx = torch.where(index)[0]
                   #true_idx = true_idx[:self.n_top_bg_to_keep*n_gt]        
                   #tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
                   thresh = torch.nonzero(my_obj>=self.obj_bg_score_thresh).flatten()
                   #true_idx = torch.arange(len(bg_matches_idx)) #sono già ordinati
                   #true_idx = true_idx[:self.n_top_bg_to_keep*n_gt]
                   true_idx = thresh[:self.n_top_bg_to_keep*n_gt]
                   tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
                   #for i in true_idx:
                   #   if my_bg_sort[i] not in tensor_taken: #guardia contro i doppi che potrebbero capitare (se già presi fra i negativi sopra)
                   #      tensor_taken = torch.cat([tensor_taken, my_bg_sort[i].unsqueeze(0)])
                   #   else: #per assicura di prendere n bg se ci sono; se trovo duplicato passo a prossima
                   #      next_index = next((i for i in thresh if my_bg_sort[i] not in tensor_taken), None)
                   #      if next_index is None: #se è None vuol dire che non ho trovato un indice
                   #         break
                   #      tensor_taken = torch.cat([tensor_taken, my_bg_sort[next_index].unsqueeze(0)])

                #risorto le anchors, le matched gt boxes rispettive e labels
                
                tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
                my_clamped = clamped_matched_idxs_in_image[tensor_taken]
                my_labels = labels_in_image[tensor_taken]
                
                #mask = torch.ones_like(labels_in_image, dtype=torch.bool)
                #mask[tensor_taken] = False
                # Set values to -1 where indexes are NOT present in 'tensor_taken'
                #labels_in_image[mask] = -1
                #print(labels_in_image)

            #matched_idxs.append(clamped_matched_idxs_in_image)
            #labels.append(labels_in_image)
            matched_idxs.append(my_clamped)
            labels.append(my_labels)
            indexes.append(tensor_taken)
        return matched_idxs, labels, indexes

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels) #2 maschere binarie dove è 1 se è indice pos ed è stato scelto oppure 0 se non lo è. Le labels originale con -1 verranno ignorate.
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0] #si fa un or fra le due maschere e si scelgono solo quegli indici che sono stati presi dal sampler o come pos o come neg o come entrambi.
            sampled_inds.append(img_sampled_inds)
        return sampled_inds
    """
	commento su sampler degli indici delle proposal. Valore di default dei campi: 512, 0.25. Il primo ti dice quante proposal tenere al massimo, scelte a caso. Il secondo ti dice: di quelle 512 proposal quante al massimo sono positive? Il sampler nel codice fa una roba del genere. Immaginiamo di avere 1 positiva (con IoU >0.5, il Roi head le discrimina così) e tenere 1000 negative (IoU<0.5).
num_pos = int(self.batch_size_per_image * self.positive_fraction)   #num_pos = 512*0.25 -> 128
# protect against not enough positive examples
num_pos = min(positive.numel(), num_pos)   # num_pos = min(1, 128) -> 1
num_neg = self.batch_size_per_image - num_pos    # num_neg = 512 - 1 -> 511
# protect against not enough negative examples
num_neg = min(negative.numel(), num_neg)    # num_neg = min(1000, 511) -> 511

# randomly select positive and negative examples
perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos] #1 a caso fra le 1 pos che ho
perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg] #511 a caso fra le 1000 neg che ho

Quindi:se voglio tenere tutte le negative, devo aumentare il 512 a un valore maggiore, ad esempio 2000, visto che di negative con basso IoU ne ho molte
    """

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
        obj_score, #aggiunto da me
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        #Ora proposals conterrà le proposals dell'rpn originale + le GT boxes
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        #
        #if self.use_custom_filter_proposals:
        if not self.use_custom_filter_proposals:
           matched_idxs, labels, indexes = self.assign_targets_to_proposals_custom_v2(proposals, gt_boxes, gt_labels, obj_score)
           #ora estraggo solo le prop che userò
           my_proposals=[]
           for proposals_in_image, indexes_in_image in zip(proposals, indexes):
              my_proposals.append(proposals_in_image[indexes_in_image])
           proposals = my_proposals
        else:
           matched_idxs, labels, match_q_matrixes = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels) #le match_q_matrixes mi servono poi per il compare
 
        # sample a fixed proportion of positive-negative proposals
        #queste rispettano i due parametri di batch_size e pos_fraction. Di default sono a 512 e 0.25, quindi verranno prese 128 pos e il resto delle 512 a neg. Questo a patto che ce ne siano abbastanza di pos e neg; altrimenti, ne vengono tenute le n pos e le restanti 512 - npos sono negative.
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
      
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, match_q_matrixes #le ritorno mi serve poi per il compare

    #Funzione qui sotto usata solo in testing
    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
        obj_score=None,
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        #PASSO 0: preparo il necessario per il training. Recupero le proposals positive e negative
        if self.training:
            #proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            proposals, matched_idxs, labels, regression_targets, match_q_matrixes = self.select_training_samples(proposals, targets, obj_score) #aggiungo io obj score; proviene da rpn ed è quanto è probabile che ogni box contenga oggetto
            #le match q matrixes mi servono poi per il compare
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        
        #PASSO 1: uso le proposals della RPN e seleziono quelle che sono interessanti, facendo ROI pooling e cropping (selezionando gli oggetti). è MultiScaleRoIAlign
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        #PASSO 2: uso le cropped feature maps di sopra che sono bbox e da li ottengo le bbox vere. è TwoMLPHead
        box_features = self.box_head(box_features)
        #PASSO 3: calcolo la bbox regression e la probabilità della classe usando l'output sopra. è FastRCNNPredictor
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training:
           #attenzione, ora proposals contiene anche le gt
           dtype = proposals[0].dtype
           device = proposals[0].device
           gt_boxes = [t["boxes"].to(dtype) for t in targets]
           gt_labels = [t["labels"] for t in targets]
           index, index_offset = self.select_proposals_custom(proposals, gt_boxes, gt_labels, labels, match_q_matrixes, class_logits)
           my_labels=[]
           my_regression_targets=[]
           for labels_in_image, regression_targets_in_image, indexes_in_image in zip(labels, regression_targets, index):
              my_labels.append(labels_in_image[indexes_in_image])
              my_regression_targets.append(regression_targets_in_image[indexes_in_image])
           my_class_logits = torch.empty(0, dtype=torch.float32).to(device)
           my_box_regression = torch.empty(0, dtype=torch.float32).to(device)
           for indexes in index_offset:
              my_class_logits = torch.cat((my_class_logits, class_logits[indexes]), dim=0)
              my_box_regression = torch.cat((my_box_regression, box_regression[indexes]), dim=0)
           
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        #PASSO 4: calcolo la loss. Questa è di classificazione e quanto buone sono le bbox.
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            if self.use_custom_filter_proposals:
               loss_classifier, loss_box_reg = fastrcnn_loss(my_class_logits, my_box_regression, my_labels, my_regression_targets)
            else:
               loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        #THIS FINAL PARTS CONTAINED OMITTED CODE OF MASKS, KEYPOINTS...
        #[...]
        return result, losses
        
    def select_proposals_custom(self, proposals, gt_boxes, gt_labels, labels, match_q_matrixes, class_logits):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        #2 liste vuote. La label qui conterrà le label dell'oggetto preso dalla GT.
        indexes = []
        indexes_offset = [] #serve per batch_size>1, perchè l'objectness sono indici tutti insieme delle img
        n_offset=0 #in base al batch size alla fine, quante len(anchors) devo sommare
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, labels_in_image, match_quality_matrix in zip(proposals, gt_boxes, gt_labels, labels, match_q_matrixes):
           tot_offset=0
           if n_offset>0:
              for a in range(0, n_offset):
                 tot_offset+=len(proposals[a])
           #recupero class_logits relative a questa proposal
           max_index = len(proposals_in_image) + tot_offset
           min_index = tot_offset
           class_logits_image = class_logits[min_index:max_index]
           #IDEA: RECUPERO INDICI CON LABEL=1 che sono negativi o del primo gt o dell'altro
           my_pos = torch.where(labels_in_image >= 1)[0]
           #questi sono negativi o primo gt o degli altri
           my_neg = torch.where(labels_in_image == 0)[0]
           #recupero class_logits dei negativi
           class_logits_neg = class_logits_image[my_neg]

           n_gt = len(gt_boxes_in_image)
           device = proposals_in_image.device
           tensor_taken=torch.empty(0, dtype=torch.int64).to(device)
           only_pos = match_quality_matrix[:, my_pos]
           only_neg = match_quality_matrix[:, my_neg]
                
           """ Premessa 1 (attivare proposals): ordinerei tutte le proposals per ogni target in base a IOU con esso (tralasciando condifence e label), seleziono le prime n e le passo alla loss """
           #estraggo i valori positivi
           matched_vals, matches_idx = only_pos.max(dim=0)
           sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)     
           my_pos_sort = my_pos[sort_ma_val_idx]
           matches_idx_sort = matches_idx[sort_ma_val_idx]            
           #le prime n sono il gt (ordinate per IoU). I gt li devo tenere comunque perché così funziona l'originale, e poi prenderne altri n. Allora li seleziono rimuovendo le prime n_gt occorrenze, e recuperando tali n_gt occorrenze subito dopo il for
           my_pos_sort_without_gt = my_pos_sort[n_gt:]
           matches_idx_without_gt = matches_idx_sort[n_gt:]
           for val in range(0, n_gt):
              index = (matches_idx_without_gt == val)
              true_idx = torch.where(index)[0]
              true_idx = true_idx[:self.n_top_pos_to_keep]
              tensor_taken = torch.cat([tensor_taken, my_pos_sort_without_gt[true_idx]])
              
           tensor_taken = torch.cat([tensor_taken, my_pos_sort[:n_gt]])
           """ Premessa 2 (sopprimere predicitions FPiou): considero solo le bbox con label 1 (persona), le ordino per confidence  e seleziono le prime t con iou < di 0.5 (qui è lo standard) """
           #Ora negativi
           matched_vals, matches_idx = only_neg.max(dim=0) #prima sono ordinati per objectness score  
           class_val, class_idx = torch.sort(class_logits_neg[:, 1], descending=True) #ora ordinati per class logits
           my_neg_sort = my_neg[class_idx]
           matches_idx_sort = matches_idx[class_idx]
           neg_already_taken=torch.empty(0, dtype=torch.int64).to(device)
           for val in range(0, n_gt):
              index = (matches_idx_sort == val)
              true_idx = torch.where(index)[0]
              true_idx = true_idx[:self.n_top_neg_to_keep]        
              tensor_taken = torch.cat([tensor_taken, my_neg_sort[true_idx]])
              neg_already_taken = torch.cat([neg_already_taken, my_neg_sort[true_idx]])
           
           """ Premessa 3 Full background (IoU zero con tutti gli oggetti). Ordinate per score """
           if self.n_top_bg_to_keep>0:
              bg_mask = (sort_ma_val == 0.0) #IoU pari a 0
              bg_ma_val_idx = sort_ma_val_idx[bg_mask]
              #elimino le neg già prese
              neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
              bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
              my_obj = obj_score_in_image[bg_ma_val_idx]
              bg_ma_val_idx, _ = torch.sort(bg_ma_val_idx, descending=False)
              my_bg_sort = my_neg[bg_ma_val_idx]
              bg_matches_idx = matches_idx[bg_ma_val_idx]
              thresh = torch.nonzero(my_obj>=self.obj_bg_score_thresh).flatten()
              true_idx = thresh[:self.n_top_bg_to_keep*n_gt]
              tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
           
           #riordino anchors nell'ordine originale
           tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
           tensor_taken_offset = tensor_taken + tot_offset
           #incremento offset per prossima img nel batch
           n_offset=n_offset+1
           indexes.append(tensor_taken)
           indexes_offset.append(tensor_taken_offset)
        return indexes, indexes_offset
