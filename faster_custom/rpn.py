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
#####

class RPNHead(nn.Module):
    _version = 2

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

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

#To manipulate tensors
def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

#To compute bbox and class score, it's necessary to concatenate all feature levels; this is what this function does
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
    #RPN class. The arguments are the same from faster rcnn file.   
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }
    
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
        # My parameters
        use_custom_filter_anchors: bool=False,
        n_top_pos_to_keep: int=1,
        n_top_neg_to_keep: int=5,
        n_top_bg_to_keep: int=0,
        objectness_bg_thresh: float = 0.00,
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
        
        self.use_custom_filter_anchors = use_custom_filter_anchors
        self.n_top_pos_to_keep = n_top_pos_to_keep
        self.n_top_neg_to_keep = n_top_neg_to_keep
        self.n_top_bg_to_keep = n_top_bg_to_keep           
        self.objectness_bg_thresh = objectness_bg_thresh

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction) # This class samples batches, ensuring that they contain a fixed proportion of positives
        # Pick a random number of positive and negative samples that satisfy the ratio
        
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3
        
    #two functions used just to pick the right parameter
    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    #Default assign target to anchors; this function for every anchor assign -1 (ignore anchor), 0 (background), 1 (positive)
    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            #If there isn't any gt in targets; the image is background and doesn't contain object
            if gt_boxes.numel() == 0:
                # Background image (negative example). I create a tensor of zeroes
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image) #grab IoU between anchor and each gt
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs >= 0          
                #this code creates tensor of labels full of 1; then if needed updates accordingly to 0 or -1
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples). Update value with 0
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds. Anchors not positive but not too bad to consider them negative. Set them to -1
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            #the labels index not updated are the anchors that will be considered as positive
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

        levels = torch.cat(levels, 0) #from tensor ([0, ..., 0], [1, ..., 1],  [2, ..., 2],  [3, ..., 3],  [4, ..., 4]) to tensor([[0, 0, 0,  ..., 4, 4, 4]])
        levels = levels.reshape(1, -1).expand_as(objectness)
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None] #id of image in batch; if batch=3 -> id= tensor([[0], [1], [2]])
        #Select objectness, levels and proposals of best bbox of images on all batch
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]     
        proposals = proposals[batch_idx, top_n_idx]       
        objectness_prob = torch.sigmoid(objectness) #compute objectness prob

        #final proposals to send to roi heads
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
	#anchors for computing the rpn loss are selected by sampler in this way:
        # -1 -> ignore anchor; 0 -> consider anchors as background; 1 -> consider anchor as positive
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        #Negative background anchors are used in obj score loss; not for bbox regression
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        #loss(predicted_class, actual_class) -> where class is 0 for background or 1 for object
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss
    
    #My custom method
    def assign_targets_to_anchors_custom_v2(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]], objectness: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        labels = []
        matched_gt_boxes = []
        indexes = [] #needed for being compatible with previous rpn.py code
        indexes_offset = [] #needed for batch_size>1, because objectness is flattened and contains all obj score for all images all together
        n_offset=0 #based on batch size, how much len(anchors) I need to sum
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            
            tot_offset=0
            if n_offset>0:
               for a in range(0, n_offset):
                  tot_offset+=len(anchors[a])

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                #I keep highest scored self.n_top_bg_to_keep if i have a background image
                #4 is for the feature levels
                my_matched_gt_boxes = torch.zeros(([self.n_top_bg_to_keep, 4]), dtype=torch.float32, device=device)
                my_labels = torch.zeros((self.n_top_bg_to_keep), dtype=torch.float32, device=device)
                objectness = objectness.flatten() #extract objectness
                my_obj = objectness[sort_ma_val_idx+tot_offset]
                my_obj_sort, my_obj_sort_idx = torch.sort(my_obj, descending=True) #sort by score
                tensor_taken = my_obj_sort_idx[:self.n_top_bg_to_keep]
                tensor_taken_offset = tensor_taken + tot_offset
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                #grab IoU of every anchor with each gt
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
                
                #My method          
                n_gt = len(gt_boxes)
                device = anchors_per_image.device
                tensor_taken=torch.empty(0, dtype=torch.int64).to(device) #empty tensors
                matched_gt_taken=torch.empty(0, dtype=torch.float32).to(device)
                labels_taken=torch.empty(0, dtype=torch.float32).to(device)
                
                #IDEA 1. Activating positive anchors. Order all of them based on their IoU with GT; keep first n.
                labels_indexes = torch.arange(len(labels_per_image), device=device)
                matched_vals, matches_idx = match_quality_matrix.max(dim=0) #now know gt for every anchor
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True)  #ordered by IoU
                my_pos_sort = labels_indexes[sort_ma_val_idx] #order label by IoU
                matches_idx_sort = matches_idx[sort_ma_val_idx] #order id by IoU    
                #e.g.  matched_vals -> tensor([0.7189, 0.6593, 0.6593, 0.7189, 0.7069], device='cuda:0')
                # matches_idx -> tensor([0, 1, 1, 0, 0], device='cuda:0') 
                # this shows that between the 5 positive anchors, 2 matches better with gt 1; 3 with gt 0
                # sort_ma_val_idx -> tensor([0, 3, 4, 1, 2], device='cuda:0')
                #this is the ordered index based on the input tensor. Useful for ordering another tensor with same order
                for val in range(0, n_gt): #for every gt
                   index = (matches_idx_sort == val) #search in matched id the corresponding one to current gt; boolean mask
                   true_idx = torch.where(index)[0] #keep only the true one
                   true_idx = true_idx[:self.n_top_pos_to_keep]  #keep only first n (highest IoU) 
                   tensor_taken = torch.cat([tensor_taken, my_pos_sort[true_idx]]) #I use index instead of values
                   labels_per_image[my_pos_sort[true_idx]] = 1.0 #Update anchors label to consider them foreground; even if they didn't satisfy the minimoum thresh
                   for i in range(0, self.n_top_pos_to_keep): #add n matched gt and labels based on how much anchors I take. 
                      # This is needed for negative, where rpn.py by default assign to negative anchors the first gt
                      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[val].unsqueeze(0)])
                      labels_taken= torch.cat([labels_taken, torch.tensor(1.0, dtype=torch.float32, device=device).unsqueeze(0)])

                #IDEA 2. Suppressing negative anchors. Pick negative anchors (under IoU), sort them by obj score; select top n
                #For selecting which are negative, I exploit the parameter of negative Matcher of bg_iou_thresh.
                #It's impossible to take again the anchors picked up at step before
                #When I do this line of code, I won't pick previously taken proposals as I have updated their label at step before!
                my_neg = torch.where(labels_per_image == 0.0)[0] 
                only_neg = match_quality_matrix[:, my_neg] #this are in matrix NxM where N is number of gt
                matched_vals, matches_idx = only_neg.max(dim=0)
                sort_ma_val, sort_ma_val_idx = torch.sort(matched_vals, descending=True) 
                objectness = objectness.flatten() #extract objectness
                my_obj = objectness[sort_ma_val_idx+tot_offset]
                my_obj_sort, my_obj_sort_idx = torch.sort(my_obj, descending=True) #sort by score
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
                
                #IDEA 3. Remove totally background anchors (with IoU=0 with all GTs). Ordered by score, above a certain thresh
                if self.n_top_bg_to_keep>0:
                   
                   bg_mask = (sort_ma_val == 0.0)
                   bg_ma_val_idx = sort_ma_val_idx[bg_mask]
                   #remove already taken negative anchors
                   neg_taken_mask = torch.isin(bg_ma_val_idx, neg_already_taken)
                   bg_ma_val_idx = bg_ma_val_idx[~neg_taken_mask]
                   
                   my_obj = objectness[bg_ma_val_idx+tot_offset]
                   my_obj_sort, my_obj_sort_idx = torch.sort(my_obj, descending=True) #sort by score
                   my_bg_sort = my_neg[my_obj_sort_idx]
                   bg_matches_idx = matches_idx[my_obj_sort_idx]
                   #keep all values above thresh
                   thresh = torch.nonzero(my_obj_sort>=self.objectness_bg_thresh).flatten()
                   true_idx = thresh[:self.n_top_bg_to_keep*n_gt] #background proposal are all associate to first gt; pick for each gt the ones with high score.
                   tensor_taken = torch.cat([tensor_taken, my_bg_sort[true_idx]])
                   for i in range(0, self.n_top_bg_to_keep*n_gt): #I assign first gt
                      matched_gt_taken = torch.cat([matched_gt_taken, gt_boxes[0].unsqueeze(0)])
                      labels_taken= torch.cat([labels_taken, torch.tensor(0.0, dtype=torch.float32, device=device).unsqueeze(0)])
                              
                #order back everything with original order
                tensor_taken, orig_order = torch.sort(tensor_taken, descending=False)
                my_matched_gt_boxes = matched_gt_taken[orig_order]
                my_labels = labels_taken[orig_order]     
                tensor_taken_offset = tensor_taken + tot_offset #offset for objectness score

            n_offset=n_offset+1 #increment offset for next image in batch
            labels.append(my_labels) #my appends
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
        # RPN uses all feature maps that are available
        # STEP 0: grab features from backbone backbone
        features = list(features.values())
        
        # STEP 1: compute obj score and bbox regression
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        #Anchors are first generated over features, than trasferred on images
        
        num_images = len(anchors)    
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        #e.g. [torch.Size([3, 232, 200]), torch.Size([3, 116, 100]), torch.Size([3, 58, 50]), torch.Size([3, 29, 25]), torch.Size([3, 15, 13])]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        #e.g.: [139200, 34800, 8700, 2175, 585]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through the proposals
        # this class under encodes bbox in format for training regression
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)        
        proposals = proposals.view(num_images, -1, 4)
        
        #STEP 2: filter proposals to keep only best one; remove small bbox with low score; apply nms
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)      
        losses = {}
        #STEP 3: only in training, train the anchors
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")           
            if self.use_custom_filter_anchors:
               #labels, matched_gt_boxes = self.assign_targets_to_anchors_custom(anchors, targets) #original method
               labels, matched_gt_boxes, indexes, indexes_offset = self.assign_targets_to_anchors_custom_v2(anchors, targets, objectness)
               device = anchors[0].device
               my_anchors=[]
               for anchors_in_image, indexes_in_image in zip(anchors, indexes):
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
               #assign to each anchor a gt
               labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
               regression_targets = self.box_coder.encode(matched_gt_boxes, anchors) #encode proposal for training
               loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, scores, losses #I added scores
