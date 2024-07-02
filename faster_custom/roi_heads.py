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

    #torch.set_printoptions(threshold=10_000)
    #print(labels)
    labels = torch.cat(labels, dim=0)
    #print(class_logits, labels)
    #x=input()
    regression_targets = torch.cat(regression_targets, dim=0)

    #classification loss computed on all preds, positive and negative; based on confidence and assigned label
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    #bbox loss only on positive
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
        #My parameters
        use_custom_filter_proposals_objectness=False,
        use_custom_filter_proposals_scores=False, 
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
        
        #My parameters
        self.use_custom_filter_proposals_objectness=use_custom_filter_proposals_objectness
        self.use_custom_filter_proposals_scores=use_custom_filter_proposals_scores
        self.n_top_pos_to_keep = n_top_pos_to_keep
        self.n_top_neg_to_keep = n_top_neg_to_keep
        self.n_top_bg_to_keep = n_top_bg_to_keep
        self.obj_bg_score_thresh = obj_bg_score_thresh

        #Useless for me, keep only for compatibility
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    #Useless for me, keep only for compatibility; they're used when checking target
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
        matched_idxs = []
        labels = []
        match_q_matrixes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image. If image is without objects in GT, I create tensor of zeros
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                match_quality_matrix = None
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
               
                #compute IoU for every proposal w.r.t. GT. Output: matrix M gt x N prop.
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                #to each proposal assign corresponding GT based on IoU thresh
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                #I'll obtain a tensor([ 0, -1, -1,  0,  1,  2]) where
                #-1 prop associated to any GT; 1 to first GT; 2 to second GT; and so on.

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                #-1 gives error, so clamping to 0

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold) #set label to 0
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds) #-1 ignored index
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            match_q_matrixes.append(match_quality_matrix) #use them for my filter
        return matched_idxs, labels, match_q_matrixes

    def assign_targets_to_proposals_custom_v2(self, proposals, gt_boxes, gt_labels, obj_score):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_idxs = []
        indexes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, obj_score_in_image in zip(proposals, gt_boxes, gt_labels, obj_score):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                #I keep highest scoring self.n_top_bg_to_keep if i have a background image
                my_clamped = torch.zeros(
                    (self.n_top_bg_to_keep), dtype=torch.int64, device=device
                )
                my_labels = torch.zeros((self.n_top_bg_to_keep), dtype=torch.int64, device=device)
                #By default proposals are ordered by objectness score, I can just keep directly first n proposals
                tensor_taken = torch.arange((self.n_top_bg_to_keep), dtype=torch.int64, device=device)
            else:
                #compute IoU for every proposal w.r.t. GT. Output: matrix M gt x N prop.
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                #I'll obtain a tensor([ 0, -1, -1,  0,  1,  2]) where
                #-1 prop associated to any GT; 1 to first GT; 2 to second GT; and so on.

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                #-1 gives error, so clamping to 0; number from 0-2 that represents to which gt a proposal must be associated
                
               
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                
                # Label background (below the low threshold) 
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                # Label ignore proposals (between low and high thresholds) 
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

                #FROM HERE MY CODE. I work on all proposals (using indexes)
                n_gt = len(gt_boxes_in_image)
                device = proposals_in_image.device
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device)
                #IDEA 1. Activating positive proposals. Order all of them based on their IoU with GT; keep first n.            
                labels_indexes = torch.arange(len(labels_in_image), device=device) #length of all proposals
                matched_vals, matches_idx = match_quality_matrix.max(dim=0) #max of this returns the gt with highest IoU for each proposal
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)
                my_pos_sort = labels_indexes[sort_ma_val_idx]
                matches_idx_sort = matches_idx[sort_ma_val_idx]
                #Original Faster RCNN implementation contains gt as first n proposals (put gt inside proposals list). 
                #I firstly remove the first n gt; then select best n proposals; and add back the n gt after the for loop
                my_pos_sort_without_gt = my_pos_sort[n_gt:]
                matches_idx_without_gt = matches_idx_sort[n_gt:]
                #sort_ma_val = sort_ma_val[n_gt:] #Debug only, to see the IoU of prop with gt present also in for loop
                for val in range(0, n_gt):
                   index = (matches_idx_without_gt == val) #avoid duplicates; every prop can be assigned to only one gt at a time
                   true_idx = torch.where(index)[0]
                   true_idx = true_idx[:self.n_top_pos_to_keep]
                   #print(torch.mean(sort_ma_val[true_idx])) #to see for each gt the IoU of the selected proposal with the gt, debug only
                   tensor_taken = torch.cat([tensor_taken, my_pos_sort_without_gt[true_idx]])
                   #I must overwrite old labels to be sure that the proposals will be considered as positive
                   #Generalized approach indipendent of number of classes
                   labels_in_image[my_pos_sort_without_gt[true_idx]] = gt_labels_in_image[val]
                   

                #Debug only, to see the IoU of positive proposals between each other. Doesn't include the gt "fake proposal" as we add them right before IDEA 2
                #pos_iou_set = proposals_in_image[tensor_taken]
                #iou_pos = box_ops.box_iou(pos_iou_set, pos_iou_set)
                #iou_pos = iou_pos[iou_pos != 1] #Remove IoU of proposals with themselves, that is 1
                #mean_iou_pos = torch.mean(iou_pos)
                
                tensor_taken = torch.cat([tensor_taken, my_pos_sort[:n_gt]])
                #IDEA 2. Suppressing wrong predictions. Pick negative proposals (under IoU thresh), sort them by score; select top n
                #For selecting which are negative, I exploit the parameter of negative Matcher of bg_iou_thresh.
                #It's impossible to take again the proposals picked up at step before
                #When I do this line of code, I won't pick previously taken proposals as I have updated their label at step before!
                #By default proposals are ordered by objectness score. So my_neg is already ordered for score, but I need to grab the corresponding GT
                my_neg = torch.where(labels_in_image == 0)[0]
                only_neg = match_quality_matrix[:, my_neg]
                matched_vals, matches_idx = only_neg.max(dim=0)
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)
                sort_ma_val_idx_score, _ = torch.sort(sort_ma_val_idx, descending=False)
                my_neg_sort = my_neg[sort_ma_val_idx_score]
                matches_idx_sort = matches_idx[sort_ma_val_idx_score]
                neg_already_taken=torch.empty(0, dtype=torch.int64).to(device)
                for val in range(0, n_gt):
                   index = (matches_idx_sort == val)
                   true_idx = torch.where(index)[0]
                   true_idx = true_idx[:self.n_top_neg_to_keep]     
                   tensor_taken = torch.cat([tensor_taken, my_neg_sort[true_idx]])
                   neg_already_taken = torch.cat([neg_already_taken, my_neg_sort[true_idx]])
                
                #Debug only, to see the IoU of negative proposals between each other.
                #neg_iou_set = proposals_in_image[neg_already_taken]
                #iou_neg = box_ops.box_iou(neg_iou_set, neg_iou_set)
                #iou_neg = iou_neg[iou_neg != 1] #Remove IoU of proposals with themselves, that is 1
                #mean_iou_neg = torch.mean(iou_neg)
                #print(mean_iou_pos, mean_iou_neg)
                              
                #IDEA 3. Remove totally wrong predictions on background (with IoU=0 with all GTs). Ordered by score, above a certain thresh
                if self.n_top_bg_to_keep>0:
                   bg_mask = (sort_ma_val == 0.0) #IoU = 0
                   bg_ma_val_idx = sort_ma_val_idx[bg_mask]
                   #Remove previously taken proposals
                   neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
                   bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
                   my_obj = obj_score_in_image[bg_ma_val_idx]
                   bg_ma_val_idx, _ = torch.sort(bg_ma_val_idx, descending=False)
                   my_bg_sort = my_neg[bg_ma_val_idx]
                   bg_matches_idx = matches_idx[bg_ma_val_idx]
                   #pick only proposals above a thresh
                   thresh = torch.nonzero(my_obj>=self.obj_bg_score_thresh).flatten()
                   true_idx = thresh[:self.n_top_bg_to_keep*n_gt]
                   #being with IoU=0, these proposals are all assigned by convention to first GT. I don't care to which GT are assigned still.
                   tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])   
                #Sorting back proposals, matched gt boxes and labels. Code expect it to be in original order
                tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
                my_clamped = clamped_matched_idxs_in_image[tensor_taken]
                my_labels = labels_in_image[tensor_taken]

            matched_idxs.append(my_clamped)
            labels.append(my_labels)
            indexes.append(tensor_taken)
            #torch.set_printoptions(threshold=10_000)
            #print(my_clamped)
        return matched_idxs, labels, indexes

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels) #2 binary mask; 1 index pos and it has been chosen; 0 if not; -1 ignored.
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0] #OR between mask and pick only proposals took by positive sampler, negative sampler or both
            sampled_inds.append(img_sampled_inds)
        return sampled_inds
    #Example of how sampler works
    #num_pos = int(self.batch_size_per_image * self.positive_fraction)   #num_pos = 512*0.25 -> 128
    # protect against not enough positive examples
    #num_pos = min(positive.numel(), num_pos)   # num_pos = min(1, 128) -> 1
    #num_neg = self.batch_size_per_image - num_pos    # num_neg = 512 - 1 -> 511
    # protect against not enough negative examples
    #num_neg = min(negative.numel(), num_neg)    # num_neg = min(1000, 511) -> 511
    # randomly select positive and negative examples
    #perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos] #1 random from the 1 that i have
    #perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg] #511 random from the 1000 neg that i have
    #So if you want to take all negatives, you need to increment the batch_size from 512 to 10000 (way higher than the possible number of prop

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
        obj_score, #added by me
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
        #now proposals will contain original rpn proposals + GT boxes
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        if self.use_custom_filter_proposals_objectness: #If I use my method
           matched_idxs, labels, indexes = self.assign_targets_to_proposals_custom_v2(proposals, gt_boxes, gt_labels, obj_score)
           #Extract now only proposals that I need
           my_proposals=[]
           for proposals_in_image, indexes_in_image in zip(proposals, indexes):
              my_proposals.append(proposals_in_image[indexes_in_image])
           proposals = my_proposals
           match_q_matrixes=None
        else:
           matched_idxs, labels, match_q_matrixes = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels) #Here I need match_q_matrixes for the compare
 
        # sample a fixed proportion of positive-negative proposals
        #Subsample follows parameter batch_size and pos_fraction. By default they are 512 and 0.25, 
        #So we'll get 128 pos and the rest are neg. This if there are enough positive; otherwise, we take n pos and the rest 512 - npos will be negative.
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
        return proposals, matched_idxs, labels, regression_targets, match_q_matrixes #match_q_matrixes needed for compare

    #Function used only in testing
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

        #STEP 0: grab necessary for training. Get positive and negative proposals.
        if self.training:
            #proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            proposals, matched_idxs, labels, regression_targets, match_q_matrixes = self.select_training_samples(proposals, targets, obj_score) #Added obj score; comes from rpn. # match q matrixes needed for compare
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        
        #STEP 1: ROI pooling and cropping. It's MultiScaleRoIAlign from faster_rcnn.py
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        #STEP 2: using cropped feature maps of above and grab real bbox. It's TwoMLPHead
        box_features = self.box_head(box_features)
        #STEP 3: compute bbox regression and class logits using output above. It's FastRCNNPredictor
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training and self.use_custom_filter_proposals_scores:
           #!now proposals contains also GT. #This is used if i filter proposals based on score
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
        #STEP 4: compute loss. Classification and bbox regression
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            if self.use_custom_filter_proposals_scores:
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
        indexes = []
        indexes_offset = [] #needed for batch_size>1, because objectness is flattened and contains all indexes all together
        n_offset=0 #based on batch size, how many len(anchors) I need to sum
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, labels_in_image, match_quality_matrix in zip(proposals, gt_boxes, gt_labels, labels, match_q_matrixes):
           tot_offset=0
           if n_offset>0:
              for a in range(0, n_offset):
                 tot_offset+=len(proposals[a])
           #grab class logits related to this proposal
           max_index = len(proposals_in_image) + tot_offset
           min_index = tot_offset
           class_logits_image = class_logits[min_index:max_index]
           device = proposals_in_image.device
           
           if match_quality_matrix is None: #means it's an image without gt
              #I keep highest scored self.n_top_bg_to_keep if i have a background image
              max_neg_noclass, _ = torch.max(class_logits_image[:, 1:], dim=1) #grab the prediction with highest score indipendent from associated class. The 1: is for skipping the background associated score
              class_val, class_idx = torch.sort(max_neg_noclass, descending=True)
              
              tensor_taken = class_idx[:self.n_top_bg_to_keep]
              tensor_taken_offset = tensor_taken + tot_offset
              indexes.append(tensor_taken)
              indexes_offset.append(tensor_taken_offset)
              n_offset=n_offset+1
              continue
           
           n_gt = len(gt_boxes_in_image)
           tensor_taken=torch.empty(0, dtype=torch.int64).to(device)
                
           #IDEA 1.
           labels_indexes = torch.arange(len(labels_in_image), device=device)
           matched_vals, matches_idx = match_quality_matrix.max(dim=0)
           sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)     
           my_pos_sort = labels_indexes[sort_ma_val_idx]
           matches_idx_sort = matches_idx[sort_ma_val_idx]            
           my_pos_sort_without_gt = my_pos_sort[n_gt:]
           matches_idx_without_gt = matches_idx_sort[n_gt:]
           for val in range(0, n_gt):
              index = (matches_idx_without_gt == val)
              true_idx = torch.where(index)[0]
              true_idx = true_idx[:self.n_top_pos_to_keep]
              tensor_taken = torch.cat([tensor_taken, my_pos_sort_without_gt[true_idx]])
              labels_in_image[my_pos_sort_without_gt[true_idx]] = gt_labels_in_image[val]
              
           tensor_taken = torch.cat([tensor_taken, my_pos_sort[:n_gt]])         
           #IDEA 2.
           my_neg = torch.where(labels_in_image == 0)[0]
           #grab negative class logits
           class_logits_neg = class_logits_image[my_neg]
           only_neg = match_quality_matrix[:, my_neg]
 
           matched_vals, matches_idx = only_neg.max(dim=0) #ordered by objectness score  
           max_neg_noclass, _ = torch.max(class_logits_neg[:, 1:], dim=1) #grab the prediction with highest score indipendent from associated class. The 1: is for skipping the background associated score
           class_val, class_idx = torch.sort(max_neg_noclass, descending=True)
           
           #class_val, class_idx = torch.sort(class_logits_neg[:, 1], descending=True) #now order by class logits
           my_neg_sort = my_neg[class_idx]
           matches_idx_sort = matches_idx[class_idx]
           neg_already_taken=torch.empty(0, dtype=torch.int64).to(device)
           for val in range(0, n_gt):
              index = (matches_idx_sort == val)
              true_idx = torch.where(index)[0]
              true_idx = true_idx[:self.n_top_neg_to_keep]        
              tensor_taken = torch.cat([tensor_taken, my_neg_sort[true_idx]])
              neg_already_taken = torch.cat([neg_already_taken, my_neg_sort[true_idx]])           

           #IDEA 3.
           if self.n_top_bg_to_keep>0:
              bg_mask = (sort_ma_val == 0.0)
              bg_ma_val_idx = sort_ma_val_idx[bg_mask]
              neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
              bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
              class_bg = class_logits_image[bg_ma_val_idx]
              
              max_neg_noclass, _ = torch.max(class_bg[:, 1:], dim=1) #grab the prediction with highest score indipendent from associated class. The 1: is for skipping the background associated score
              class_val, class_idx = torch.sort(max_neg_noclass, descending=True)
              my_bg_sort = my_neg[class_idx]
              thresh = torch.nonzero(class_val>=self.obj_bg_score_thresh).flatten()
              true_idx = thresh[:self.n_top_bg_to_keep*n_gt]
              tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
           
           #order back to original order
           tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
           tensor_taken_offset = tensor_taken + tot_offset
           #increment offset for next image in the batch
           n_offset=n_offset+1
           indexes.append(tensor_taken)
           indexes_offset.append(tensor_taken_offset)
        return indexes, indexes_offset
